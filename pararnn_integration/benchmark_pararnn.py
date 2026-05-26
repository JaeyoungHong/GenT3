"""Compare sequential ConvGRU vs SpectralMMConvGRU parallel modes.

Sequential conv2d baseline: a standard ConvGRU — three 3×3 convs (one per
gate) for state mixing + three 1×1 convs for input projection, looped over T.

Parallel side: SpectralMMConvGRU from pararnn_integration, in both `parallel`
(PyTorch Newton + diag scan) and `parallel_CUDA` (ParaRNN's custom CUDA kernel)
modes.

Measures forward + backward time (since parallel-in-T training is the whole
point — backward matters as much as forward) and peak GPU memory above baseline.
"""

import time
import torch
import torch.nn as nn

from pararnn_integration import SpectralMMConvGRU


# ── Baseline: standard ConvGRU ─────────────────────────────────────────────

class ConvGRU(nn.Module):
    """Sequential ConvGRU. State mixing = full 3×3 conv per gate."""

    def __init__(self, D, H, W, device='cuda', dtype=torch.float32):
        super().__init__()
        self.D, self.H, self.W = D, H, W
        # State mixing: 3×3 conv per gate (z, r, h)
        self.U_z = nn.Conv2d(D, D, 3, padding=1, bias=False, device=device, dtype=dtype)
        self.U_r = nn.Conv2d(D, D, 3, padding=1, bias=False, device=device, dtype=dtype)
        self.U_h = nn.Conv2d(D, D, 3, padding=1, bias=False, device=device, dtype=dtype)
        # Input projection: 1×1 conv per gate
        self.W_z = nn.Conv2d(D, D, 1, bias=True, device=device, dtype=dtype)
        self.W_r = nn.Conv2d(D, D, 1, bias=True, device=device, dtype=dtype)
        self.W_h = nn.Conv2d(D, D, 1, bias=True, device=device, dtype=dtype)

    def forward(self, x):
        # x: (B, T, H, W, D)
        x_chw = x.permute(0, 1, 4, 2, 3).contiguous()        # (B, T, D, H, W)
        B, T = x_chw.shape[:2]
        h = torch.zeros(B, self.D, self.H, self.W, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(T):
            xt = x_chw[:, t]
            z = torch.sigmoid(self.W_z(xt) + self.U_z(h))
            r = torch.sigmoid(self.W_r(xt) + self.U_r(h))
            h_tilde = torch.tanh(self.W_h(xt) + self.U_h(r * h))
            h = (1 - z) * h + z * h_tilde
            outs.append(h)
        out = torch.stack(outs, dim=1)                       # (B, T, D, H, W)
        return out.permute(0, 1, 3, 4, 2)                    # (B, T, H, W, D)


# ── Measurement utility ───────────────────────────────────────────────────

def measure(model, x, warmup=2, iters=5):
    """Returns (fwd_ms, bwd_ms, peak_MB).

    fwd_ms and bwd_ms are averaged across iters; peak_MB is the peak GPU
    memory above baseline (so each method's marginal cost is comparable).
    """
    for _ in range(warmup):
        y = model(x)
        loss = (y ** 2).sum()
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None
        if x.grad is not None:
            x.grad = None

    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    fwd_total = 0.0
    bwd_total = 0.0
    for _ in range(iters):
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e2 = torch.cuda.Event(enable_timing=True)
        e0.record()
        y = model(x)
        loss = (y ** 2).sum()
        e1.record()
        loss.backward()
        e2.record()
        torch.cuda.synchronize()
        fwd_total += e0.elapsed_time(e1)
        bwd_total += e1.elapsed_time(e2)
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None
        if x.grad is not None:
            x.grad = None

    peak = torch.cuda.max_memory_allocated()
    return fwd_total / iters, bwd_total / iters, (peak - base) / (1024 ** 2)


# ── One-row runner ─────────────────────────────────────────────────────────

def run(B, T, D, H, W, label):
    device = torch.device('cuda')
    torch.manual_seed(42)

    def make_x():
        return torch.randn(B, T, H, W, D, device=device, requires_grad=True)

    # Baseline: sequential ConvGRU
    torch.manual_seed(0)
    convgru = ConvGRU(D, H, W, device=device)
    x = make_x()
    fwd_c, bwd_c, mem_c = measure(convgru, x)
    total_c = fwd_c + bwd_c
    del convgru, x
    torch.cuda.empty_cache()

    # SpectralMMConvGRU — parallel (PyTorch Newton + scan)
    torch.manual_seed(0)
    par_model = SpectralMMConvGRU(D=D, H=H, W=W, num_heads=1, mode='parallel', device=device)
    x = make_x()
    fwd_p, bwd_p, mem_p = measure(par_model, x)
    total_p = fwd_p + bwd_p
    del par_model, x
    torch.cuda.empty_cache()

    # SpectralMMConvGRU — parallel_CUDA (custom kernel)
    torch.manual_seed(0)
    cuda_model = SpectralMMConvGRU(D=D, H=H, W=W, num_heads=1, mode='parallel_CUDA', device=device)
    x = make_x()
    fwd_pc, bwd_pc, mem_pc = measure(cuda_model, x)
    total_pc = fwd_pc + bwd_pc
    del cuda_model, x
    torch.cuda.empty_cache()

    print(f"{label}  "
          f"convgru={total_c:7.1f}ms /{mem_c:6.0f}MB  "
          f"par_py={total_p:7.1f}ms /{mem_p:6.0f}MB  "
          f"par_cuda={total_pc:7.1f}ms /{mem_pc:6.0f}MB  "
          f"spd_py={total_c/total_p:5.2f}x  spd_cuda={total_c/total_pc:5.2f}x")


# ── Sweeps ─────────────────────────────────────────────────────────────────

def section(title):
    print("\n" + "═" * 130)
    print(title)
    print("═" * 130)


def subsection(title):
    print(f"\n── {title} ──")


if __name__ == '__main__':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("\nMethods compared (forward + backward time / peak memory above baseline):")
    print("  convgru   : standard ConvGRU, 3×3 conv per gate, T-step sequential loop")
    print("  par_py    : SpectralMMConvGRU, ParaRNN 'parallel' mode (Newton + scan, PyTorch)")
    print("  par_cuda  : SpectralMMConvGRU, ParaRNN 'parallel_CUDA' mode (custom CUDA kernel)")

    T0, D0, K0, B0 = 512, 32, 8, 4

    section("Vary T  (D=32, H=W=8, B=4)")
    for T in [32, 128, 512, 2048]:
        run(B0, T, D0, K0, K0, f"T={T:5d}")

    section("Vary D  (T=512, H=W=8, B=4)")
    for D in [8, 16, 32, 64, 128, 256]:
        run(B0, T0, D, K0, K0, f"D={D:5d}")

    section("Vary H=W  (T=512, D=32, B=4)")
    for K in [4, 8, 16, 24, 32]:
        run(B0, T0, D0, K, K, f"H=W={K:3d}")

    section("Vary B  (T=512, D=32, H=W=8)")
    for B in [1, 4, 16, 32, 64, 128]:
        run(B, T0, D0, K0, K0, f"B={B:5d}")
