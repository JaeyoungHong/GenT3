"""
Benchmark: sequential 3x3 conv recurrence vs parallel MM-Conv spectral scan.

SSM recurrence: h[t] = A(h[t-1]) + input[t]

Methods compared:
  1. Sequential depthwise Conv2d     — naive baseline, T GPU kernels
  2. Sequential MM-Conv (spectral)   — same math, spectral path, T Python iterations
  3. Parallel MM-Conv scan           — O(log T) depth via prefix scan in spectral domain
"""

import torch
import torch.nn.functional as F
from mmconv import MMConv

DEVICE = torch.device('cuda')
torch.backends.cudnn.benchmark = True


# ── Parallel prefix scan ──────────────────────────────────────────────────────

def parallel_scan(Lambda, Z, h0):
    """
    Solve h[t] = Lambda * h[t-1] + Z[t] for all t simultaneously.

    Binary-tree prefix scan, O(log T) depth.  Uses a double-buffer scheme:
    two fixed (T,...) tensors are pre-allocated and swapped each level so
    there are zero runtime allocations inside the loop.

    Lambda : (D, H, W)        constant per-mode decay
    Z      : (T, B, D, H, W)  inputs in spectral domain
    h0     : (B, D, H, W)     initial spectral state
    Returns: (T, B, D, H, W)  all spectral states h[0..T-1]
    """
    T, _, D, H, W = Z.shape

    # ── pre-allocate double buffers (one-time cost) ──
    # A tracks the cumulative Lambda^k factor; shape (T, 1, D, H, W) broadcasts over B
    A = [Lambda.unsqueeze(0).unsqueeze(0).expand(T, 1, D, H, W).clone(),
         torch.empty(T, 1, D, H, W, device=Z.device, dtype=Z.dtype)]
    # B tracks the running state accumulation
    B = [Z.clone(), torch.empty_like(Z)]
    B[0][0] = B[0][0] + Lambda * h0   # absorb initial state into first slot

    src, dst = 0, 1
    step = 1
    while step < T:
        n = T - step
        # Copy unchanged prefix so dst is fully self-contained after the swap
        if step > 0:
            A[dst][:step].copy_(A[src][:step])
            B[dst][:step].copy_(B[src][:step])
        # A[dst][step:] = A[src][step:] * A[src][:n]   — no temp via out=
        torch.mul(A[src][step:], A[src][:n], out=A[dst][step:])
        # B[dst][step:] = A[src][step:] * B[src][:n] + B[src][step:]  — no temps via addcmul
        torch.addcmul(B[src][step:], A[src][step:], B[src][:n], out=B[dst][step:])
        src, dst = dst, src
        step *= 2

    return B[src]


# ── Spectral lift / unlift for the channel-diagonal case ─────────────────────

def make_ops(block, D):
    with torch.no_grad():
        rho_H, sigma_H, rho_W, sigma_W = block._derive_twists()
        rH = rho_H  [range(D), range(D)]   # (D, H)
        sH = sigma_H[range(D), range(D)]   # (D, H)
        rW = rho_W  [range(D), range(D)]   # (D, W)
        sW = sigma_W[range(D), range(D)]   # (D, W)
        PhiH, PsiH = block.Phi_H, block.Psi_H
        PhiW, PsiW = block.Phi_W, block.Psi_W
        Lam = block._spectral_gain()[range(D), range(D)]   # (D, H, W)
        K   = block.to_conv_kernel()
        dw  = K[range(D), range(D)].unsqueeze(1)           # (D, 1, 3, 3)
    return rH, sH, rW, sW, PhiH, PsiH, PhiW, PsiW, Lam, dw


def make_lift_unlift(sH, sW, rH, rW, PhiH, PsiH, PhiW, PsiW):
    def lift(X):
        # (..., H, W, D) → (..., D, H, W)
        Xt = torch.einsum('ph,pw,...hwp->...phw', sH, sW, X)
        return torch.einsum('kh,lw,...phw->...pkl', PsiH, PsiW, Xt)

    def unlift(Om):
        # (..., D, H, W) → (..., H, W, D)
        Os = torch.einsum('ik,jl,...pkl->...pij', PhiH, PhiW, Om)
        return torch.einsum('ph,pw,...phw->...hwp', rH, rW, Os)

    return lift, unlift


# ── Timing + memory utility ───────────────────────────────────────────────────

def measure(fn, warmup=3, iters=10):
    """Run fn and return (avg_ms, peak_extra_MB).

    peak_extra_MB is the peak GPU memory allocated above the pre-call baseline,
    so it isolates each method's working-set cost from setup tensors.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    torch.cuda.synchronize()
    ms = e0.elapsed_time(e1) / iters
    peak = torch.cuda.max_memory_allocated()
    return ms, (peak - base) / (1024 ** 2)


# ── Main benchmark ────────────────────────────────────────────────────────────

def run(B=4, T=256, D=32, H=8, W=8, label=None):
    if label is None:
        label = f"T={T:5d}"
    # Channel-diagonal MM-Conv: a=0.7, b=c=0.15, d=0 (stable: |0.7|+2*0.15=1.0, borderline)
    # Use a=0.6 to stay safely inside |Lambda|<1
    block = MMConv(D, D, H, W).to(DEVICE).eval()
    with torch.no_grad():
        I = torch.eye(D, device=DEVICE)
        block.a.data           = I * 0.60
        block.b.data           = I * 0.15
        block.c.data           = I * 0.15
        block.d.data           = I * 0.0
        block.log_alpha_H.data = I * 0.0   # alpha_H = exp(0) = 1
        block.log_alpha_W.data = I * 0.0

    rH, sH, rW, sW, PhiH, PsiH, PhiW, PsiW, Lam, dw = make_ops(block, D)
    lift, unlift = make_lift_unlift(sH, sW, rH, rW, PhiH, PsiH, PhiW, PsiW)

    inp_hwc = torch.randn(T, B, H, W, D, device=DEVICE)
    inp_chw = inp_hwc.permute(0, 1, 4, 2, 3).contiguous()
    h0_hwc  = torch.zeros(B, H, W, D, device=DEVICE)
    h0_chw  = torch.zeros(B, D, H, W, device=DEVICE)

    # ── Method 1: sequential depthwise Conv2d ─────────────────────────────────
    def seq_conv2d():
        h = h0_chw.clone()
        out = []
        for t in range(T):
            h = F.conv2d(h, dw, padding=1, groups=D) + inp_chw[t]
            out.append(h)
        return torch.stack(out)   # (T, B, D, H, W)

    # ── Method 2: sequential spectral MM-Conv ─────────────────────────────────
    def seq_mmconv():
        # Hoisted DST: one batched lift over all T inputs, then loop is just elementwise.
        Z = lift(inp_hwc.reshape(T * B, H, W, D)).reshape(T, B, D, H, W)
        h = lift(h0_hwc)   # (B, D, H, W) spectral
        out = []
        for t in range(T):
            h = Lam * h + Z[t]
            out.append(h)
        # batch-unlift at end
        stacked = torch.stack(out)                              # (T, B, D, H, W)
        return unlift(stacked.reshape(T * B, D, H, W)).reshape(T, B, H, W, D)

    # ── Method 3: parallel MM-Conv scan ───────────────────────────────────────
    def par_mmconv():
        Z = lift(inp_hwc.reshape(T * B, H, W, D)).reshape(T, B, D, H, W)
        H_all = parallel_scan(Lam, Z, lift(h0_hwc))            # (T, B, D, H, W)
        return unlift(H_all.reshape(T * B, D, H, W)).reshape(T, B, H, W, D)

    # correctness (compare last time step in spatial domain)
    with torch.no_grad():
        yc = seq_conv2d()   # (T, B, D, H, W)
        ys = seq_mmconv()   # (T, B, H, W, D)
        yp = par_mmconv()   # (T, B, H, W, D)

    err_p = (yp.permute(0,1,4,2,3) - yc).abs().max().item()
    del yc, ys, yp
    torch.cuda.empty_cache()

    ms1, m1 = measure(seq_conv2d)
    ms2, m2 = measure(seq_mmconv)
    ms3, m3 = measure(par_mmconv)

    print(f"{label}  "
          f"conv2d={ms1:7.2f}ms /{m1:6.0f}MB  "
          f"seq_mm={ms2:7.2f}ms /{m2:6.0f}MB  "
          f"par_mm={ms3:7.2f}ms /{m3:6.0f}MB  "
          f"spd={ms1/ms3:5.2f}x  err={err_p:.1e}")


# ── Full dense: matrix recurrence scan ───────────────────────────────────────

def par_matrix_scan(Lam_hw, Z_flat, h0_flat):
    """
    Parallel scan for the full dense matrix recurrence:
      h[t, b, k, :] = Lam_hw[k] @ h[t-1, b, k, :] + Z[t, b, k, :]

    With CONSTANT transfer matrix (same Lam at every t), the associative
    operation is (A2, b2) o (A1, b1) = (A2@A1, A2@b1 + b2), giving a
    binary-tree scan of depth O(log T) using batched matmuls.

    Lam_hw  : (HW, D, D)   one D×D matrix per spectral mode
    Z_flat  : (T, B, HW, D)
    h0_flat : (B, HW, D)
    Returns : (T, B, HW, D)
    """
    T, Bs, HW, D = Z_flat.shape

    # Absorb h0 into Z[0]: Z[0] += Lam @ h0
    Z_flat = Z_flat.clone()
    Z_flat[0] += torch.einsum('hqr,bhr->bhq', Lam_hw, h0_flat)

    # Pre-allocate double buffers — no runtime allocs inside the loop.
    # .contiguous() ensures reshape(n_el,D,D) returns a view (not a copy) so
    # out= writes actually land in the buffer.
    # A: (T, HW, D, D), B: (T, B, HW, D)
    A = [Lam_hw.unsqueeze(0).expand(T, -1, -1, -1).contiguous(),
         torch.empty(T, HW, D, D, device=Z_flat.device, dtype=Z_flat.dtype)]
    B = [Z_flat, torch.empty_like(Z_flat)]

    src, dst = 0, 1
    step = 1
    while step < T:
        n = T - step
        # Copy unchanged prefix to dst
        A[dst][:step].copy_(A[src][:step])
        B[dst][:step].copy_(B[src][:step])

        # A[dst][step:] = A[src][step:] @ A[src][:n]   (batched matmul, no alloc)
        n_el = n * HW
        torch.bmm(A[src][step:].reshape(n_el, D, D),
                  A[src][:n  ].reshape(n_el, D, D),
                  out=A[dst][step:].reshape(n_el, D, D))

        # B[dst][step:] = A[src][step:] @ B[src][:n] + B[src][step:]
        # A[src][step:]: (n, HW, D, D) → unsqueeze batch dim
        # B[src][:n]   : (n, B, HW, D) → treat last dim as column vector
        torch.matmul(
            A[src][step:].unsqueeze(1),          # (n, 1, HW, D, D)
            B[src][:n].unsqueeze(-1),             # (n, B, HW, D, 1)
            out=B[dst][step:].unsqueeze(-1)       # (n, B, HW, D, 1) — view of dst
        )
        B[dst][step:].add_(B[src][step:])         # + B_curr, in-place

        src, dst = dst, src
        step *= 2

    return B[src]


def run_dense(B=4, T=256, D=32, H=8, W=8, label=None):
    """
    Full dense MM-Conv (all D×D channel pairs active) with uniform alpha=1.

    With uniform alpha the shared DST reduces the spectral state from
    (D,D,H,W) to (D,H,W): the D-vector at each spectral mode (k_h,k_w)
    evolves under the D×D matrix Lambda[k_h,k_w], enabling a matrix
    recurrence parallel scan via batched matmul.
    """
    if label is None:
        label = f"T={T:5d}"
    block = MMConv(D, D, H, W).to(DEVICE).eval()
    torch.manual_seed(0)
    with torch.no_grad():
        block.log_alpha_H.data.zero_()    # alpha = exp(0) = 1 (uniform)
        block.log_alpha_W.data.zero_()
        # Scale random part by 1/sqrt(D) so spectral norm stays bounded as D grows.
        # Without this, |Lambda| ~ 1 + sqrt(D)*sigma  →  T=512 powers overflow float32.
        s = (D ** -0.5)
        block.a.data = torch.eye(D, device=DEVICE) * 0.5 \
                     + torch.randn(D, D, device=DEVICE) * (0.05 * s)
        block.b.data = torch.randn(D, D, device=DEVICE) * (0.05 * s)
        block.c.data = torch.randn(D, D, device=DEVICE) * (0.05 * s)
        block.d.data.zero_()

    with torch.no_grad():
        # Lambda (D,D,H,W), kernel (D,D,3,3)
        Lambda = block._spectral_gain()           # (D, D, H, W)
        K      = block.to_conv_kernel()           # (D, D, 3, 3)
        PhiH, PsiH = block.Phi_H, block.Psi_H
        PhiW, PsiW = block.Phi_W, block.Psi_W
        # Reshape Lambda for matrix scan: (HW, D, D), must be contiguous so
        # par_matrix_scan's reshape(n_el,D,D) returns a view, not a copy
        Lam_hw = Lambda.permute(2, 3, 0, 1).reshape(H * W, D, D).contiguous()
        # Stacked 1×1-conv weights for the seq_mmconv loop (detached from autograd).
        W_stack = torch.stack([block.a, block.b, block.c, block.d]).reshape(4 * D, D, 1, 1)
        scale = torch.stack([
            torch.ones(H, W, device=DEVICE),
            block.xi_H.unsqueeze(-1).expand(H, W),
            block.xi_W.unsqueeze(0).expand(H, W),
            block.xi_H.unsqueeze(-1) * block.xi_W.unsqueeze(0),
        ]).view(1, 4, 1, H, W)

    # ── Lift / unlift for uniform alpha=1 (shared DST, no per-(q,r) twist) ──
    # X (..., H, W, D_in) → Z (..., D_in, H, W)
    def lift(X):
        return torch.einsum('kj,lm,...jmr->...rkl', PsiH, PsiW, X)

    # Inverse DST only — converts spectral state (q-indexed) back to spatial.
    # The recurrence already applies Lambda at each step; no Lambda here.
    def unlift_raw(Z):
        return torch.einsum('ik,jl,...qkl->...ijq', PhiH, PhiW, Z)

    inp_hwc = torch.randn(T, B, H, W, D, device=DEVICE)
    inp_chw = inp_hwc.permute(0, 1, 4, 2, 3).contiguous()
    h0_hwc  = torch.zeros(B, H, W, D, device=DEVICE)
    h0_chw  = torch.zeros(B, D, H, W, device=DEVICE)

    # ── Sequential dense Conv2d ───────────────────────────────────────────────
    def seq_conv2d():
        h = h0_chw.clone()
        out = []
        for t in range(T):
            h = F.conv2d(h, K, padding=1) + inp_chw[t]
            out.append(h)
        return torch.stack(out)   # (T, B, D, H, W)

    # ── Sequential dense MM-Conv spectral ─────────────────────────────────────
    # Lambda factors separably:  Lam = a + b·ξ_H + c·ξ_W + d·ξ_H·ξ_W .
    # So the spectral step  h_new[q,k,l] = Σ_r Lam[q,r,k,l] · h[r,k,l]  becomes a
    # single (4D, D, 1, 1) cuDNN 1×1 conv followed by a per-pixel weighted fold of
    # its four channel blocks — strictly faster than einsum/bmm at these sizes.
    # (W_stack and scale are precomputed inside the no_grad block above.)
    def seq_mmconv():
        # Hoisted DST: one batched lift over all T inputs, loop is recurrence only.
        Z = lift(inp_hwc.reshape(T * B, H, W, D)).reshape(T, B, D, H, W)
        h = lift(h0_hwc)   # (B, D, H, W) spectral
        out = []
        for t in range(T):
            y = F.conv2d(h, W_stack).view(B, 4, D, H, W) * scale  # (B, 4, D, H, W)
            h = y.sum(dim=1) + Z[t]
            out.append(h)
        stacked = torch.stack(out)   # (T, B, D, H, W)
        return unlift_raw(stacked.reshape(T * B, D, H, W)).reshape(T, B, H, W, D)

    # ── Parallel dense MM-Conv matrix scan ────────────────────────────────────
    def par_mmconv():
        # Batch-lift all T inputs at once (shared DST applied once)
        Z_all   = lift(inp_hwc.reshape(T * B, H, W, D)).reshape(T, B, D, H, W)
        h0_spec = lift(h0_hwc)   # (B, D, H, W)

        # Reshape to (T, B, HW, D) for the matrix scan
        Z_flat  = Z_all.permute(0, 1, 3, 4, 2).reshape(T, B, H * W, D).contiguous()
        h0_flat = h0_spec.permute(0, 2, 3, 1).reshape(B, H * W, D).contiguous()

        H_flat = par_matrix_scan(Lam_hw, Z_flat, h0_flat)  # (T, B, HW, D)

        # Reshape back and batch-unlift
        H_all = H_flat.reshape(T * B, H, W, D).permute(0, 3, 1, 2)  # (T*B, D, H, W)
        return unlift_raw(H_all).reshape(T, B, H, W, D)

    # ── Correctness ───────────────────────────────────────────────────────────
    with torch.no_grad():
        yc = seq_conv2d()
        ys = seq_mmconv()
        yp = par_mmconv()

    err_p = (yp.permute(0,1,4,2,3) - yc).abs().max().item()
    del yc, ys, yp
    torch.cuda.empty_cache()

    ms1, m1 = measure(seq_conv2d)
    ms2, m2 = measure(seq_mmconv)
    ms3, m3 = measure(par_mmconv)

    print(f"{label}  "
          f"conv2d={ms1:7.2f}ms /{m1:6.0f}MB  "
          f"seq_mm={ms2:7.2f}ms /{m2:6.0f}MB  "
          f"par_mm={ms3:7.2f}ms /{m3:6.0f}MB  "
          f"spd={ms1/ms3:5.2f}x  err={err_p:.1e}")


def section(title):
    print("\n" + "═" * 110)
    print(title)
    print("═" * 110)


def subsection(title):
    print(f"\n── {title} ──")


if __name__ == '__main__':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Defaults for fixed dims when sweeping another
    T0, D0, K0, B0 = 512, 32, 8, 4

    # ── Diagonal ────────────────────────────────────────────────────────────
    section("DIAGONAL (channel-independent depthwise, parallel scalar scan)")

    subsection(f"vary T  (D={D0}, H=W={K0}, B={B0})")
    for T in [32, 128, 512, 2048]:
        run(B=B0, T=T, D=D0, H=K0, W=K0, label=f"T={T:5d}")

    subsection(f"vary D  (T={T0}, H=W={K0}, B={B0})")
    for D in [8, 16, 32, 64, 128]:
        run(B=B0, T=T0, D=D, H=K0, W=K0, label=f"D={D:5d}")

    subsection(f"vary H=W  (T={T0}, D={D0}, B={B0})")
    for K in [4, 8, 16, 32]:
        run(B=B0, T=T0, D=D0, H=K, W=K, label=f"H=W={K:3d}")

    subsection(f"vary B  (T={T0}, D={D0}, H=W={K0})")
    for B in [1, 4, 16, 64]:
        run(B=B, T=T0, D=D0, H=K0, W=K0, label=f"B={B:5d}")

    # ── Dense ──────────────────────────────────────────────────────────────
    section("DENSE (D×D channel mixing, uniform alpha, parallel matrix scan)")

    subsection(f"vary T  (D={D0}, H=W={K0}, B={B0})")
    for T in [32, 128, 512, 2048]:
        run_dense(B=B0, T=T, D=D0, H=K0, W=K0, label=f"T={T:5d}")

    subsection(f"vary D  (T={T0}, H=W={K0}, B={B0})")
    for D in [8, 16, 32, 64]:
        run_dense(B=B0, T=T0, D=D, H=K0, W=K0, label=f"D={D:5d}")

    subsection(f"vary H=W  (T={T0}, D={D0}, B={B0})")
    for K in [4, 8, 16, 32]:
        run_dense(B=B0, T=T0, D=D0, H=K, W=K, label=f"H=W={K:3d}")

    subsection(f"vary B  (T={T0}, D={D0}, H=W={K0})")
    for B in [1, 4, 16, 64]:
        run_dense(B=B, T=T0, D=D0, H=K0, W=K0, label=f"B={B:5d}")
