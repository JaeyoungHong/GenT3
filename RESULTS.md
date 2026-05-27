# Results Summary

Consolidated experimental findings for the MM-Conv spectral framework and its
ParaRNN integration. All benchmarks on **NVIDIA A100 80GB PCIe**, PyTorch +
CUDA 12.1, float32.

---

## Part 1 — Linear MM-Conv scan (`benchmark.py`)

Task: the SSM-style linear recurrence `h[t] = MMConv(h[t−1]) + x[t]`, run three ways:

- **`seq_conv2d`** — sequential `F.conv2d` with the materialized 3×3 kernel (T launches)
- **`seq_mmconv`** — sequential spectral recurrence (lift once, loop `Λ·h + z[t]`, unlift once)
- **`par_mmconv`** — `O(log T)` parallel prefix scan in the spectral domain

### Diagonal MM-Conv (channel-independent), D=32, H=W=8, B=4

| T | seq_conv2d | seq_mmconv (hoisted) | par_mmconv | par speedup |
|---:|---:|---:|---:|---:|
| 32   | 0.9 ms  | 1.1 ms  | 0.8 ms | 1.1× |
| 512  | 13.7 ms | 10.1 ms | 1.1 ms | **12.9×** |
| 2048 | 55.9 ms | 38.1 ms | 2.6 ms | **21×** |

Two findings:
- The **parallel scan** reaches ~21× at T=2048 — the `O(log T)` depth advantage.
- The **hoisted sequential spectral form already beats seq_conv2d** (38 ms vs 56 ms at T=2048): the spectral recurrence step `Λ·h + z[t]` is a fused multiply-add, cheaper than conv2d's 9-tap stencil. Hoisting the DST out of the loop (one batched lift instead of T) was the key optimization.

### Dense MM-Conv (D×D channel mixing, uniform α), D=32, H=W=8, B=4

| T | seq_conv2d | seq_mmconv | par_mmconv | par speedup |
|---:|---:|---:|---:|---:|
| 2048 | 85 ms | 170 ms | 67 ms | 1.3× |

Dense is much harder: the parallel scan composes D×D transfer matrices, costing `O(D³)` per scan level. The dense `seq_mmconv` was sped up ~17% by expressing the per-mode matmul as a single fused `(4D, D, 1, 1)` 1×1 conv (cuDNN) instead of an einsum.

**Takeaway:** the diagonal case is where the spectral parallel scan pays off; dense is dominated by the `D³` matrix-composition cost.

---

## Part 2 — ParaRNN integration: parallel-in-T nonlinear RNN training

Task: a gated **nonlinear** recurrence (GRU) operating on spatial feature-map
sequences, `x: (B, T, H, W, D) → y: (B, T, H, W, D)`. The MM-Conv spectral form
parameterizes the diagonal state mixing; ParaRNN parallelizes the recurrence
along T via Newton + diagonal parallel scan.

Compared (forward **+ backward** time) against two sequential baselines:

- **dense** — standard ConvGRU, full 3×3 channel-mixing state conv (D² per tap)
- **depthwise** — ConvGRU with 3×3 *depthwise* state conv (channel-diagonal — the *fair* comparison, same model structure as ours)
- **par_cuda** — `SpectralMMConvGRU` in `parallel_CUDA` mode (Newton + custom diag-scan kernel)

### Vary T (D=32, H=W=8, B=4)

| T | dense | depthwise | par_cuda | vs dense | vs depthwise |
|---:|---:|---:|---:|---:|---:|
| 32   |  42 ms  |  39 ms  | 9.7 ms | 4.3× | 4.0× |
| 128  | 174 ms  | 154 ms  | 9.7 ms | 18× | 16× |
| 512  | 694 ms  | 635 ms  | 14.9 ms | **47×** | **43×** |
| 2048 | 2784 ms | 2566 ms | 54 ms  | **51×** | **47×** |

### Vary D (T=512, H=W=8, B=4)

| D | dense | depthwise | par_cuda | vs depthwise | par mem |
|---:|---:|---:|---:|---:|---:|
| 8   | 704 ms | 612 ms | 9.6 ms | **64×** | 108 MB |
| 32  | 694 ms | 621 ms | 15.5 ms | **40×** | 433 MB |
| 64  | 743 ms | 632 ms | 30 ms  | **21×** | 833 MB |
| 128 | 771 ms | 641 ms | 57 ms  | **11×** | 1.7 GB |
| 256 | 927 ms | 687 ms | 129 ms | **5.3×** | 3.3 GB |

### Vary H=W (T=512, D=32, B=4)

| H=W | dense | depthwise | par_cuda | vs depthwise | par mem |
|---:|---:|---:|---:|---:|---:|
| 4  | 700 ms  | 627 ms  | 9.4 ms | **66×** | 108 MB |
| 8  | 695 ms  | 622 ms  | 14.4 ms | **43×** | 433 MB |
| 16 | 666 ms  | 606 ms  | 56 ms  | **11×** | 1.7 GB |
| 24 | 818 ms  | 738 ms  | 137 ms | **5.4×** | 3.8 GB |
| 32 | 1100 ms | 1031 ms | 266 ms | **3.9×** | 6.8 GB |

### Vary B (T=512, D=32, H=W=8)

| B | dense | depthwise | par_cuda | vs depthwise |
|---:|---:|---:|---:|---:|
| 1   | 616 ms  | 572 ms  | 9.5 ms | **60×** |
| 4   | 697 ms  | 636 ms  | 15.5 ms | **41×** |
| 16  | 692 ms  | 627 ms  | 57 ms  | **11×** |
| 64  | 1100 ms | 1042 ms | 223 ms | **4.7×** |
| 128 | 1730 ms | 1630 ms | 450 ms | **3.6×** |

---

## Key insights

### 1. The win is parallel-in-T, not the channel-diagonal restriction
Dense and depthwise ConvGRU run in **nearly identical wall time** (within 5–15%) — both are cuDNN-launch-overhead-bound at these tensor sizes, so depthwise's `9·D` state-mix FLOPs vs dense's `9·D²` is invisible. Consequently the speedup against depthwise is ~90% of the speedup against dense. **~10% of the speedup comes from the model-class restriction; ~90% from the parallel scan.**

### 2. `num_heads = H·W` is essential (was a configuration trap)
ParaRNN's input projection `B` has shape `(num_heads, head_input_dim, 3, head_state_dim)`.
- `num_heads=1`: one `(state_dim, 3·state_dim)` matrix → `O((D·H·W)²)` work — quartic in spatial, quadratic in D. Kills scaling.
- `num_heads=H·W`: one D×D projection per spatial position (mirrors a 1×1 conv) → `O(H·W·D²)`, recurrence linear in D.

Fixing this took D=256 from **2.7× slower** than ConvGRU to **6.8× faster** (an 18× improvement), and dropped memory from 12.4 GB to 3.3 GB.

### 3. Scaling behavior (after the fix)
| Dimension | par_cuda scaling | conv2d scaling | Net |
|---|---|---|---|
| T | sub-linear (`log T` scan depth) | linear (T launches) | speedup *grows* with T |
| D | linear (diagonal state) | D² (but launch-bound until large D) | speedup shrinks but stays >1 |
| H·W | **quadratic** (DST cost) | linear (but launch-bound) | speedup shrinks fastest |
| B | linear | linear (launch-bound until large B) | speedup shrinks |

**The framework's one genuine algorithmic weakness is the DST: cost is quadratic in H or W.** That's intrinsic to the spectral decomposition, not an implementation choice. Everything else scales linearly.

### 4. Where the framework is the right choice
```
long sequence (T ≥ 100)   AND
small spatial (H·W ≤ 16×16)   AND
modest channel (D ≤ 256)   AND
modest batch (B ≤ 128)
```
That's the SSM training regime. In it, you get 4–70× faster training (forward + backward) than a sequential ConvGRU, and parallel-in-T training becomes feasible where it wasn't. The framework beat both baselines at **every config tested** (worst case 3.6× at B=128).

### 5. Correctness
All three application modes (`sequential`, `parallel`, `parallel_CUDA`) match to **~1e-6** on forward output and gradients (within the expected `eps · seq_length` Newton-accumulation tolerance). Gradients flow into all learnable knobs: α (DST twist), (a,b,c,d) (spectral gains), gates, and input projection.

### 6. `parallel_CUDA` vs `parallel` (PyTorch)
The custom CUDA diag-scan kernel is consistently 5–15% faster than the pure-PyTorch parallel mode. The modest gain (vs 5×+ for the *linear* MM-Conv scan) is because in the nonlinear case the Newton iteration + Jacobian assembly runs in PyTorch for both modes; only the inner linear solve differs.

---

## Method comparison: parallel scan vs ParaRNN

| | Linear recurrence | Nonlinear recurrence (ParaRNN) |
|---|---|---|
| Primitive | parallel scan | parallel scan |
| Wrapper | none | Newton iteration |
| Passes / step | 1 (exact) | 3–4 Newton iters (exact at convergence) |
| Our examples | linear MM-Conv (`benchmark.py`) | SpectralMMConvGRU/LSTM (`pararnn_integration/`) |

ParaRNN = Newton outer loop + parallel scan inner solver. The scan is the same
primitive Mamba/S4/S5 use; the Newton wrapper is what extends it to arbitrary
nonlinear cells (GRU, LSTM, ...). The diagonal state structure (shared by
ParaRNN's stock GRUDiagMH and our MM-Conv parameterization) is what makes the
Jacobian bandable and the fast diag-scan kernel applicable.
