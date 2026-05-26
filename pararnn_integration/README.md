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
- `spectral_wrapper.py` — `SpectralMMConvGRU`, `SpectralMMConvLSTM`. `nn.Module` wrappers that own `α_H`, `α_W`, build the per-channel twisted DST matrices on each forward, and sandwich the cell between lift / unlift einsums. **`num_heads` defaults to `H·W`** (each spatial position gets its own D×D input projection, mirroring a 1×1 conv); see the discussion at the bottom.
- `test_integration.py` — sequential vs parallel correctness + speedup sanity check.
- `benchmark_pararnn.py` — full sweep vs sequential ConvGRU (dense and depthwise variants), measuring fwd+bwd time and peak GPU memory across T, D, H=W, B.
- `test_lift_only.py` — standalone DST roundtrip check (no ParaRNN needed; useful for sanity-checking the math).

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

## Speedup vs sequential ConvGRU

`benchmark_pararnn.py` measures forward + backward time and peak GPU memory
against two sequential baselines: a dense 3×3 ConvGRU (full D² channel-mixing
state) and a depthwise 3×3 ConvGRU (channel-diagonal state — the fair
comparison, matching our framework's structure). At `T=512, D=32, H=W=8, B=4`:

| Config | dense | depthwise | par_cuda | vs dense | vs depthwise |
|---|---:|---:|---:|---:|---:|
| T=128    |  174 ms |  154 ms | 9.7 ms | **18×** | **16×** |
| T=2048   | 2784 ms | 2566 ms | 54 ms  | **51×** | **47×** |
| D=8      |  704 ms |  612 ms | 9.6 ms | **74×** | **64×** |
| D=64     |  743 ms |  632 ms | 30 ms  | **25×** | **21×** |
| D=256    |  927 ms |  687 ms | 129 ms | **7.2×**| **5.3×** |
| H=W=4    |  700 ms |  627 ms | 9.4 ms | **74×** | **66×** |
| H=W=16   |  666 ms |  606 ms | 57 ms  | **12×** | **11×** |
| H=W=32   | 1100 ms | 1031 ms | 266 ms | **4.1×**| **3.9×** |
| B=1      |  616 ms |  572 ms | 9.5 ms | **65×** | **60×** |
| B=128    | 1730 ms | 1630 ms | 450 ms | **3.8×**| **3.6×** |

**Two non-obvious takeaways:**

1. *Dense and depthwise ConvGRU run in nearly identical wall time* — both are
   cuDNN-launch-overhead-bound at these tensor sizes, so depthwise's 9·D
   state-mix FLOPs vs dense's 9·D² is invisible. The channel-diagonal
   restriction in our model class buys ~10% wall-time, not 10×.

2. *The remaining ~90% of the speedup is from parallel-in-T training itself.*
   Doing one big batched Newton + diag scan instead of T sequential cudnn
   convs is the actual mechanism. Wins are biggest at long T, small batch,
   small spatial, modest channel — the SSM regime.

The framework still beats both baselines at every config tested. Spatial
size is where it degrades fastest (DST cost is quadratic in H or W) — at
H=W=32 the speedup drops to 4× from 70+ at H=W=4.

## Why `num_heads = H·W` (the default)

This is non-obvious enough to warrant a callout. ParaRNN's input projection
matrix `B` has shape `(num_heads, head_input_dim, 3, head_state_dim)` — it's
block-diagonal in the input/state with `num_heads` independent blocks.

- With `num_heads=1` (the obvious default): one giant `(state_dim, 3·state_dim)`
  matrix. For state_dim = D·H·W this scales as `(D·H·W)²` in parameters and
  FLOPs — quartic in spatial and quadratic in D. Wrecks scaling.
- With `num_heads=H·W` (our default): one D×D projection per spatial position,
  exactly matching how a 1×1 conv treats channels. `H·W · D²` params total,
  recurrence scales linearly in D.

You can override `num_heads=` in the wrapper constructor if you want; the
default is set up for the "spatial input" use case where each `(k_h, k_w)`
spectral mode is an independent unit.

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
