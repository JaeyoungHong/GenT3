"""Unit tests for MMConv following the implementation plan in instructions.md Section 8."""

import math
import torch
import torch.nn.functional as F
from mmconv import MMConv

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running tests on: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == 'cuda' else ""))


def make_block(d_in, d_out, H, W, alpha_init=1.0, **kw):
    return MMConv(d_in, d_out, H, W, alpha_init=alpha_init, **kw).to(DEVICE).eval()


# ── Test 1: pure 1×1 conv when b=c=d=0 ──────────────────────────────────────

def test_pointwise_equivalence():
    H, W, d_in, d_out = 4, 5, 3, 7
    block = make_block(d_in, d_out, H, W)  # alpha_init=1.0, twists are all 1
    with torch.no_grad():
        block.b.zero_()
        block.c.zero_()
        block.d.zero_()

    X = torch.randn(2, H, W, d_in, device=DEVICE)
    with torch.no_grad():
        Y = block(X)
        # Expected: Y[..., q] = sum_r a[q, r] * X[..., r]
        Y_ref = torch.einsum('qr,...r->...q', block.a, X)

    assert torch.allclose(Y, Y_ref, atol=1e-5), \
        f"1×1 conv equiv failed: max err {(Y - Y_ref).abs().max():.2e}"
    print("PASS  test_pointwise_equivalence")


# ── Test 2: equivalence to materialized 3×3 kernel ───────────────────────────

def test_conv_kernel_equivalence():
    H, W, d_in, d_out = 5, 6, 4, 8
    # Use float64: float32 accumulates ~1e-3 error across the multi-step DST chain
    block = make_block(d_in, d_out, H, W).double()

    X = torch.randn(2, H, W, d_in, device=DEVICE, dtype=torch.float64)
    with torch.no_grad():
        Y_mm = block(X)

        K = block.to_conv_kernel()  # (D_o, D_i, 3, 3)
        Xc = X.permute(0, 3, 1, 2)  # (B, D_i, H, W)
        Yc = F.conv2d(Xc, K, padding=1)  # (B, D_o, H, W)
        Y_ref = Yc.permute(0, 2, 3, 1)  # (B, H, W, D_o)

    assert torch.allclose(Y_mm, Y_ref, atol=1e-5), \
        f"3×3 kernel equiv failed: max err {(Y_mm - Y_ref).abs().max():.2e}"
    print("PASS  test_conv_kernel_equivalence")


# ── Test 3: linearity in X ───────────────────────────────────────────────────

def test_linearity():
    H, W, d_in, d_out = 4, 4, 3, 5
    block = make_block(d_in, d_out, H, W)
    X1 = torch.randn(2, H, W, d_in, device=DEVICE)
    X2 = torch.randn(2, H, W, d_in, device=DEVICE)
    s = 3.14

    with torch.no_grad():
        Y_sum = block(X1 + X2)
        Y_scale = block(s * X1)
        Y1 = block(X1)
        Y2 = block(X2)

    assert torch.allclose(Y_sum, Y1 + Y2, atol=1e-5), \
        f"Additivity failed: max err {(Y_sum - Y1 - Y2).abs().max():.2e}"
    assert torch.allclose(Y_scale, s * Y1, atol=1e-5), \
        f"Homogeneity failed: max err {(Y_scale - s * Y1).abs().max():.2e}"
    print("PASS  test_linearity")


# ── Test 4: gradient check ────────────────────────────────────────────────────

def test_gradcheck():
    H, W, d_in, d_out = 3, 3, 2, 2
    block = MMConv(d_in, d_out, H, W).to(DEVICE).double()

    X = torch.randn(1, H, W, d_in, dtype=torch.float64, device=DEVICE, requires_grad=True)

    def fn(x):
        return block(x)

    result = torch.autograd.gradcheck(fn, (X,), eps=1e-5, atol=1e-4, rtol=1e-3,
                                      raise_exception=True)
    assert result
    print("PASS  test_gradcheck")


if __name__ == '__main__':
    test_pointwise_equivalence()
    test_conv_kernel_equivalence()
    test_linearity()
    test_gradcheck()
    print("\nAll tests passed.")
