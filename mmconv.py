"""MM-Conv: Spectral Framework for Coupled Channel-Mixing and 3×3 Convolution."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_dst_matrices(N: int, device, dtype):
    """Build Phi (forward DST) and Psi (inverse DST) for size N.

    Phi[i, k] = sin(i * k * pi / (N + 1))   i, k in 1..N  (0-indexed: +1 shift applied)
    Psi = (2 / (N + 1)) * Phi
    xi[k] = cos(k * pi / (N + 1))
    """
    idx = torch.arange(1, N + 1, device=device, dtype=dtype)  # (N,)
    Phi = torch.sin(idx[:, None] * idx[None, :] * math.pi / (N + 1))  # (N, N)
    Psi = (2.0 / (N + 1)) * Phi
    xi = torch.cos(idx * math.pi / (N + 1))  # (N,)
    return Phi, Psi, xi


class MMConv(nn.Module):
    """MM-Conv block: simultaneous channel mixing and 3×3 spatial convolution.

    Args:
        d_in:       input channel dimension D_i
        d_out:      output channel dimension D_o
        H:          spatial height
        W:          spatial width
        alpha_init: initial value for alpha_H and alpha_W (default -1.0)
    """

    def __init__(self, d_in: int, d_out: int, H: int, W: int, alpha_init: float = -1.0):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.H = H
        self.W = W

        shape = (d_out, d_in)

        # alpha is stored in log-space so exp(log_alpha) > 0 always (real-valued path).
        # For negative or complex alpha, switch to complex tensors externally.
        log_alpha_init = math.log(abs(alpha_init)) if alpha_init != 0 else 0.0
        self.log_alpha_H = nn.Parameter(torch.full(shape, log_alpha_init))
        self.log_alpha_W = nn.Parameter(torch.full(shape, log_alpha_init))
        # Init: b=c=d=0 so block starts as pure 1×1 conv; a ~ Xavier
        self.a = nn.Parameter(torch.empty(shape))
        self.b = nn.Parameter(torch.zeros(shape))
        self.c = nn.Parameter(torch.zeros(shape))
        self.d = nn.Parameter(torch.zeros(shape))
        nn.init.xavier_uniform_(self.a)

        # Precomputed, fixed buffers
        Phi_H, Psi_H, xi_H = _build_dst_matrices(H, device='cpu', dtype=torch.float32)
        Phi_W, Psi_W, xi_W = _build_dst_matrices(W, device='cpu', dtype=torch.float32)
        self.register_buffer('Phi_H', Phi_H)
        self.register_buffer('Psi_H', Psi_H)
        self.register_buffer('xi_H', xi_H)
        self.register_buffer('Phi_W', Phi_W)
        self.register_buffer('Psi_W', Psi_W)
        self.register_buffer('xi_W', xi_W)

    @property
    def alpha_H(self):
        return self.log_alpha_H.exp()

    @property
    def alpha_W(self):
        return self.log_alpha_W.exp()

    def _derive_twists(self):
        """Compute rho and sigma tensors from alpha parameters."""
        alpha_H = self.alpha_H  # (D_o, D_i), always positive real
        alpha_W = self.alpha_W
        jH = torch.arange(1, self.H + 1, device=alpha_H.device,
                          dtype=alpha_H.dtype)  # (H,)
        jW = torch.arange(1, self.W + 1, device=alpha_W.device,
                          dtype=alpha_W.dtype)  # (W,)
        # rho_H[q,r,j] = alpha_H[q,r]^(j/2),  always real and finite for alpha>0
        rho_H = alpha_H[..., None] ** (jH / 2)    # (D_o, D_i, H)
        sigma_H = alpha_H[..., None] ** (-jH / 2)  # (D_o, D_i, H)
        rho_W = alpha_W[..., None] ** (jW / 2)    # (D_o, D_i, W)
        sigma_W = alpha_W[..., None] ** (-jW / 2)  # (D_o, D_i, W)
        return rho_H, sigma_H, rho_W, sigma_W

    def _spectral_gain(self):
        """Compute Lambda (D_o, D_i, H, W)."""
        xi_H = self.xi_H  # (H,)
        xi_W = self.xi_W  # (W,)
        Lambda = (
            self.a[..., None, None]
            + self.b[..., None, None] * xi_H[None, None, :, None]
            + self.c[..., None, None] * xi_W[None, None, None, :]
            + self.d[..., None, None] * xi_H[None, None, :, None] * xi_W[None, None, None, :]
        )  # (D_o, D_i, H, W)
        return Lambda

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Apply MM-Conv to X.

        Args:
            X: (..., H, W, D_i)
        Returns:
            Y: (..., H, W, D_o)
        """
        rho_H, sigma_H, rho_W, sigma_W = self._derive_twists()
        Lambda = self._spectral_gain()

        Psi_H = self.Psi_H.to(X.dtype)
        Psi_W = self.Psi_W.to(X.dtype)
        Phi_H = self.Phi_H.to(X.dtype)
        Phi_W = self.Phi_W.to(X.dtype)

        # Stage 1: lift
        # Xtwisted[..., q, r, j_h, j_w] = sigma_H[q,r,j_h] * sigma_W[q,r,j_w] * X[...,j_h,j_w,r]
        Xtwisted = torch.einsum('qrh,qrw,...hwr->...qrhw', sigma_H, sigma_W, X)
        # Z[..., q, r, k_h, k_w] = sum_{j_h,j_w} Psi_H[k_h,j_h] * Psi_W[k_w,j_w] * Xtwisted
        Z = torch.einsum('kh,lw,...qrhw->...qrkl', Psi_H, Psi_W, Xtwisted)

        # Stage 2: spectral gain
        Omega = Lambda * Z  # (..., D_o, D_i, H, W)

        # Stage 3: unlift + contract r
        Omega_spatial = torch.einsum('ik,jl,...qrkl->...qrij', Phi_H, Phi_W, Omega)
        Y = torch.einsum('qrh,qrw,...qrhw->...hwq', rho_H, rho_W, Omega_spatial)

        return Y

    def to_conv_kernel(self) -> torch.Tensor:
        """Materialize the equivalent 3×3 convolution kernel.

        Derived from the spectral form: Lambda = a + b*xi_H + c*xi_W + d*xi_H*xi_W.
        xi_H acts on H (row) axis, xi_W on W (col) axis. The twist prefactors produce
        sqrt(alpha) factors (not alpha) in the off-diagonal entries.

        PyTorch conv2d kernel K[p,q] multiplies X[i+p-1, j+q-1], so:
          K[2,1] = coeff of X[i+1, j]  (dh=+1, upper H band)
          K[0,1] = coeff of X[i-1, j]  (dh=-1, lower H band)
          K[1,2] = coeff of X[i, j+1]  (dw=+1, upper W band)
          K[1,0] = coeff of X[i, j-1]  (dw=-1, lower W band)

        Returns:
            K: (D_o, D_i, 3, 3)  — equivalent nn.Conv2d weight tensor
        """
        a, b, c, d = self.a, self.b, self.c, self.d
        sH = self.alpha_H.sqrt()   # sqrt(alpha_H), shape (D_o, D_i)
        sW = self.alpha_W.sqrt()   # sqrt(alpha_W)
        K = torch.zeros(self.d_out, self.d_in, 3, 3, device=a.device, dtype=a.dtype)
        K[:, :, 1, 1] = a
        # b*xi_H → H-direction off-diagonals
        K[:, :, 2, 1] = b / (2 * sH)      # dh=+1, dw=0
        K[:, :, 0, 1] = b * sH / 2        # dh=-1, dw=0
        # c*xi_W → W-direction off-diagonals
        K[:, :, 1, 2] = c / (2 * sW)      # dh=0, dw=+1
        K[:, :, 1, 0] = c * sW / 2        # dh=0, dw=-1
        # d*xi_H*xi_W → cross off-diagonals (product of H and W factors)
        K[:, :, 2, 2] = d / (4 * sH * sW)
        K[:, :, 2, 0] = d * sW / (4 * sH)
        K[:, :, 0, 2] = d * sH / (4 * sW)
        K[:, :, 0, 0] = d * sH * sW / 4
        return K
