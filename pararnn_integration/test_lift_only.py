"""Standalone lift/unlift math check — doesn't import pararnn.

Verifies the α-twisted DST round-trip is identity. Useful for confirming the
spatial-decomposition math works in environments where ParaRNN's CUDA extension
hasn't been built.
"""
import math
import torch


def _build_dst_matrices(N, device, dtype):
    idx = torch.arange(1, N + 1, device=device, dtype=dtype)
    Phi = torch.sin(idx[:, None] * idx[None, :] * math.pi / (N + 1))
    Psi = (2.0 / (N + 1)) * Phi
    return Phi, Psi


def twisted_lift(x, alpha_H, alpha_W, Psi_H, Psi_W):
    """x: (B, T, H, W, D) → z: (B, T, D, H, W) spectral."""
    D, H, W = alpha_H.shape[0], Psi_H.shape[0], Psi_W.shape[0]
    jH = torch.arange(1, H + 1, device=x.device, dtype=x.dtype)
    jW = torch.arange(1, W + 1, device=x.device, dtype=x.dtype)
    sigma_H = alpha_H[:, None] ** (-jH / 2)
    sigma_W = alpha_W[:, None] ** (-jW / 2)
    Psi_H_tw = sigma_H[:, None, :] * Psi_H[None, :, :]   # (D, H, H)
    Psi_W_tw = sigma_W[:, None, :] * Psi_W[None, :, :]   # (D, W, W)
    return torch.einsum('dkj,dlm,btjmd->btdkl', Psi_H_tw, Psi_W_tw, x)


def twisted_unlift(z, alpha_H, alpha_W, Phi_H, Phi_W):
    """z: (B, T, D, H, W) → y: (B, T, H, W, D) spatial."""
    D, H, W = alpha_H.shape[0], Phi_H.shape[0], Phi_W.shape[0]
    jH = torch.arange(1, H + 1, device=z.device, dtype=z.dtype)
    jW = torch.arange(1, W + 1, device=z.device, dtype=z.dtype)
    rho_H = alpha_H[:, None] ** (jH / 2)
    rho_W = alpha_W[:, None] ** (jW / 2)
    Phi_H_tw = rho_H[:, :, None] * Phi_H[None, :, :]
    Phi_W_tw = rho_W[:, :, None] * Phi_W[None, :, :]
    return torch.einsum('dik,djl,btdkl->btijd', Phi_H_tw, Phi_W_tw, z)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")
    torch.manual_seed(0)
    for D, H, W in [(4, 5, 6), (8, 8, 8), (16, 16, 16)]:
        Phi_H, Psi_H = _build_dst_matrices(H, device, torch.float32)
        Phi_W, Psi_W = _build_dst_matrices(W, device, torch.float32)
        for alpha_init in [1.0, 0.5, 2.0]:
            alpha_H = torch.full([D], alpha_init, device=device)
            alpha_W = torch.full([D], alpha_init, device=device)
            x = torch.randn(2, 3, H, W, D, device=device)
            z = twisted_lift(x, alpha_H, alpha_W, Psi_H, Psi_W)
            y = twisted_unlift(z, alpha_H, alpha_W, Phi_H, Phi_W)
            err = (y - x).abs().max().item()
            tag = "OK" if err < 1e-4 else "FAIL"
            print(f"  D={D:3d} H={H:3d} W={W:3d} α={alpha_init:.1f}  "
                  f"max|y−x| = {err:.2e}  [{tag}]")


if __name__ == '__main__':
    main()
