# MM-Conv × ParaRNN integration

Combines the MM-Conv spectral framework with Apple's ParaRNN parallel-in-time
RNN cells. The recurrence is parallelized along T via Newton + diagonal scan
(ParaRNN does the heavy lifting); the spatial decomposition is parameterized by
MM-Conv (we plug in here).

## Architecture (one picture)

```
  x  (B, T, H, W, D)
   │
   │   ──► α-twisted DST lift  (one batched einsum, outside the time loop)
   │       Ψ̃_H[d,k,j] = α_H[d]^(-j/2)·Ψ_H[k,j]
   │       Ψ̃_W[d,l,j] = α_W[d]^(-j/2)·Ψ_W[l,j]
   ▼
  z_spec  (B, T, D·H·W)  — flat spectral state
   │
   │   ──► ParaRNN diagonal GRU / CIFG-LSTM cell
   │       Diagonal A vector = a + b·ξ_H + c·ξ_W + d·ξ_H·ξ_W  (per gate, per (k,l), per d)
   │       Newton + parallel_reduce_diag (PyTorch or CUDA mode)
   ▼
  h_spec  (B, T, D·H·W)
   │
   │   ──► α-twisted iDST unlift
   │       Φ̃_H[d,i,k] = α_H[d]^( i/2)·Φ_H[i,k]
   │       Φ̃_W[d,j,l] = α_W[d]^( j/2)·Φ_W[j,l]
   ▼
  y  (B, T, H, W, D)
```

## Three knobs, three places

| Lives in | Knob | Shape | Controls |
|---|---|---|---|
| `SpectralMMConv{GRU,LSTM}` wrapper | `α_H`, `α_W` | (D,) each | Spatial decomposition geometry (σ, ρ twists) |
| `MMConvGRUDiagCell._build_A` | `a, b, c, d` | (3, D) each | Per-gate spectral gain shape |
| ParaRNN's `GRUDiagMHImpl` (unchanged) | gates / nonlinearities | — | The actual recurrence |

## Files

- `mmconv_gru_diag.py` — `MMConvGRUDiagCell`. Replaces ParaRNN's free `(3, state_dim)` A parameter with a `(3, D)` MM-Conv parameterization. Reuses `GRUDiagMHImpl` unchanged.
- `mmconv_lstm_cifg_diag.py` — `MMConvLSTMCIFGDiagCell`. Same for CIFG-LSTM, applied to both `A` (state mixing) and `C` (peephole) vectors. Reuses `LSTMCIFGDiagMHImpl` unchanged.
- `spectral_wrapper.py` — `SpectralMMConvGRU`, `SpectralMMConvLSTM`. `nn.Module` wrappers that own `α_H`, `α_W`, build the per-channel twisted DST matrices on each forward, and sandwich the cell between lift / unlift einsums.
- `test_integration.py` — sequential vs parallel correctness + speedup sanity check.

## Modes available

| Mode | Status | Notes |
|---|---|---|
| `sequential` | ✅ Free — for debugging | T-step Python loop |
| `parallel` | ✅ Newton + diagonal scan in PyTorch | Default; pure PyTorch |
| `parallel_CUDA` | ✅ Custom diag-scan kernel | Works at any state-dim |
| `parallel_FUSED` | 🛠️ Not wired up | Would need a CUDA kernel that knows about the MM-Conv A parameterization |

Switch via `model.cell.mode = 'parallel_CUDA'` (or set `mode='parallel_CUDA'`
when constructing the wrapper).

## Installing ParaRNN

The cells `import pararnn.…`, so you need ParaRNN on PYTHONPATH. Either install
it editable, or just put its repo root on PYTHONPATH:

```bash
# Editable install (will build CUDA extensions):
cd /home/sky/ml-pararnn
/home/sky/GenT3/.venv/bin/pip install -e . --no-build-isolation

# OR no-install path:
export PYTHONPATH=/home/sky/ml-pararnn:$PYTHONPATH
```

The CUDA extensions are only required for `parallel_CUDA` and `parallel_FUSED`
modes. The pure-PyTorch `parallel` mode works without them, though `import
pararnn.rnn_cell.gru_diag_mh` does `torch.ops.load_library(...)` at import,
which fails if the `.so` isn't built — so for now CUDA build is effectively
required to even import. (If we want a CPU-only path we'd need a small monkey-
patch around that import; not currently included.)

## Running tests

```bash
cd /home/sky/GenT3
.venv/bin/python -m pararnn_integration.test_integration
```

Expect:
- `test_lift_unlift_roundtrip`: max error ≈ 1e-6 (DST round-trip with α=1)
- `test_mmconv_gru` / `test_mmconv_lstm`: max |Δfwd|, |Δgrad| growing as
  `eps · seq_length` (Newton convergence + sequence accumulation; ParaRNN's own
  tests have the same property)
- `test_spectral_gru_endtoend`: confirms gradients flow into `log_alpha_H` and
  `cell.a_mm` (i.e. the new MM-Conv knobs are learnable)

## Open items / extensions

- Input-channel ≠ state-channel: the wrapper currently assumes `D_in == D`.
  Add a 1×1 spatial conv (or a per-mode linear projection) before the lift if
  you need a different input channel count.
- `parallel_FUSED`: ParaRNN's fused GRU/LSTM kernels assume a free `(3, D)` A
  vector. They'd still work — you just hand them the materialized `A` tensor.
  Already implemented under-the-hood since we reuse `GRUDiagMHImpl` unchanged
  (which has a `fused_parallel_forward`). Should work if the `.so` is built; not
  yet tested.
- α-twist initialization: defaults to `α=1` (σ=ρ=1, untwisted DST). To explore
  non-uniform spatial decompositions, pass `alpha_H_init` / `alpha_W_init` to
  the wrapper constructor.
