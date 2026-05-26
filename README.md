# GenT3 — MM-Conv spectral framework

A PyTorch implementation of **MM-Conv**: a parameterized linear operator that simultaneously performs

- a **D×D matrix multiplication** across the channel dimension, and
- a **3×3 convolution** across the spatial dimensions,

with the two coupled through a shared spectral decomposition (Discrete Sine Transform). MM-Conv is mathematically equivalent to applying a 3×3 conv with a tridiagonal-Toeplitz-structured kernel, but its spectral form admits an `O(log T)` parallel scan when used as the linear step of an SSM-style recurrence — much faster than rolling out `T` sequential `F.conv2d` calls.

See [`instructions.md`](instructions.md) for the full mathematical derivation.

## What's in this repo

| File / dir | Purpose |
|---|---|
| [`mmconv.py`](mmconv.py) | The `MMConv` `nn.Module` — forward in spectral form, plus `to_conv_kernel()` to materialize the equivalent 3×3 conv weight. |
| [`test_mmconv.py`](test_mmconv.py) | Unit tests: 1×1-conv equivalence (b=c=d=0), 3×3-kernel equivalence, linearity, autograd gradcheck. |
| [`benchmark.py`](benchmark.py) | Benchmark suite — sequential conv2d vs sequential spectral vs parallel spectral scan. Sweeps over `T`, `D`, `H=W`, `B`; tracks both time and peak GPU memory. |
| [`pararnn_integration/`](pararnn_integration/) | Plug MM-Conv into Apple's [ParaRNN](https://github.com/apple/ml-pararnn) framework, getting parallel-in-T training of *nonlinear* RNN cells (GRU / CIFG-LSTM) with MM-Conv-parameterized state mixing. |
| [`instructions.md`](instructions.md) | Original specification — the math of MM-Conv, TT eigendecomposition, the lift / spectral-gain / unlift three-stage forward, and the SSM extension. |

## Quickstart

```python
from mmconv import MMConv

block = MMConv(d_in=32, d_out=32, H=8, W=8).cuda()
x = torch.randn(B, H, W, d_in, device='cuda')
y = block(x)                       # (B, H, W, d_out) — equivalent to a 3×3 conv with a TT-structured kernel
K = block.to_conv_kernel()         # (d_out, d_in, 3, 3) — the materialized equivalent
```

## When the parallel scan wins

`benchmark.py` runs three implementations of `h[t] = MMConv(h[t−1]) + x[t]`:

1. **`seq_conv2d`** — sequential `F.conv2d` with the materialized 3×3 kernel.
2. **`seq_mmconv`** — sequential spectral form: lift once, loop `Λ·h + z[t]`, unlift once.
3. **`par_mmconv`** — `O(log T)` parallel prefix scan in the spectral domain.

Headline result on an A100, channel-diagonal MM-Conv, `D=32, H=W=8, B=4`:

| T | `seq_conv2d` | `par_mmconv` | speedup |
|---|---|---|---|
| 32 | 0.9 ms | 0.8 ms | 1.1× |
| 512 | 13.7 ms | 1.1 ms | 12.9× |
| 2048 | 55.9 ms | 2.6 ms | **21×** |

Full dense MM-Conv (D×D channel mixing, uniform α) is harder — the parallel scan's transfer matrices cost `O(D³)` per level, so the gain is more modest (~1.3× at T=2048). The diagonal case is where parallelism pays.

The scan also loses when `H·W`, `B`, or `D` grow large enough that conv2d becomes compute-bound rather than launch-overhead-bound — see `benchmark.py` for the full sweeps.

## ParaRNN integration — parallel-in-T training of nonlinear RNNs

The spectral form of MM-Conv is diagonal in (channel × spectral-mode), which fits exactly into ParaRNN's `RNNCellDiagImpl`. That means you can wrap MM-Conv inside a GRU or CIFG-LSTM cell, train it in **parallel along the sequence dimension** via Newton + parallel reduction, and get ParaRNN's CUDA diag-scan kernel for free.

```python
from pararnn_integration import SpectralMMConvGRU

# Input: (B, T, H, W, D_in) — spatial feature-map sequence
# Output: (B, T, H, W, D)   — same shape, spatially convolved + recurrent
model = SpectralMMConvGRU(D=32, H=8, W=8, mode='parallel_CUDA').cuda()
y = model(x)                       # ParaRNN handles parallel-in-T training
```

Three independent learnable knobs:

- **α_H, α_W** — per-channel spatial-decomposition geometry (σ/ρ twists)
- **(a, b, c, d)** — per-gate, per-channel MM-Conv spectral gain coefficients
- gates / nonlinearities — same as ParaRNN's GRUDiagMH / LSTMCifgDiagMH

### Speedup over sequential ConvGRU baselines

`pararnn_integration/benchmark_pararnn.py` compares forward + backward time against two sequential baselines: a standard `ConvGRU` (full 3×3 channel-mixing state conv) and a `DepthwiseConvGRU` (3×3 depthwise state conv, same channel-diagonal structure as our framework — the fair comparison). Headline on A100, `T=512, D=32, H=W=8, B=4`:

| Config | dense ConvGRU | depthwise ConvGRU | par_cuda | vs dense | vs depthwise |
|---|---:|---:|---:|---:|---:|
| T=2048   | 2784 ms | 2566 ms | 54 ms  | **51×** | **47×** |
| D=8      |  704 ms |  612 ms | 9.6 ms | **74×** | **64×** |
| D=256    |  927 ms |  687 ms | 129 ms | **7×**  | **5×** |
| H=W=4    |  700 ms |  627 ms | 9.4 ms | **74×** | **66×** |
| H=W=32   | 1100 ms | 1031 ms | 266 ms | **4×**  | **4×** |
| B=1      |  616 ms |  572 ms | 9.5 ms | **65×** | **60×** |
| B=128    | 1730 ms | 1630 ms | 450 ms | **4×**  | **4×** |

**The depthwise and dense ConvGRU baselines are nearly identical in wall time** (both launch-overhead-bound at these sizes) — so the channel-diagonal restriction in our framework accounts for only ~10% of the speedup. The other ~90% comes from the parallel-in-T training itself (Newton + diag scan vs T sequential cudnn calls). The framework wins decisively at every config tested.

See [`pararnn_integration/README.md`](pararnn_integration/README.md) for the full layout, file-by-file responsibilities, and installation notes (the CUDA-toolchain build hint at the bottom).

## Installation

```bash
# core (mmconv.py, benchmark.py, test_mmconv.py) — just needs PyTorch with CUDA
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# optional: ParaRNN integration (needs nvcc + g++ + the apple/ml-pararnn repo)
git clone https://github.com/apple/ml-pararnn ../ml-pararnn
# ... build with CUDA flags — see pararnn_integration/README.md for the exact incantation
```

## Running tests / benchmarks

```bash
.venv/bin/python test_mmconv.py                                    # unit tests
.venv/bin/python benchmark.py                                      # full sweep (~10 minutes on A100)
.venv/bin/python pararnn_integration/test_lift_only.py             # standalone DST roundtrip — no ParaRNN
.venv/bin/python -m pararnn_integration.test_integration           # parallel/sequential/CUDA correctness
```

## License

See `LICENSE` (TODO: add).
