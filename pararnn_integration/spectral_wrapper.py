"""nn.Module wrappers: α-twisted DST  →  ParaRNN cell  →  iDST.

The MM-Conv "ratio parameters" α_H, α_W live here (one per channel, learnable
via log-space for positivity). They build per-channel twisted DST matrices

    Ψ̃_H[d, k, j] = α_H[d]^(-j/2) · Ψ_H[k, j]      # σ-twist, for lift
    Ψ̃_W[d, l, j] = α_W[d]^(-j/2) · Ψ_W[l, j]
    Φ̃_H[d, i, k] = α_H[d]^( i/2) · Φ_H[i, k]      # ρ-twist, for unlift
    Φ̃_W[d, j, l] = α_W[d]^( j/2) · Φ_W[j, l]

and apply them around the inner ParaRNN cell, which sees pure flat spectral state.
The DST/iDST are batched over the whole T sequence and live *outside* the time loop,
so they're a one-shot cost per forward pass.
"""

import math
import typing as typ

import torch
import torch.nn as nn

from .mmconv_gru_diag import MMConvGRUDiagConfig, MMConvGRUDiagCell
from .mmconv_lstm_cifg_diag import MMConvLSTMCIFGDiagConfig, MMConvLSTMCIFGDiagCell


def _build_dst_matrices(N: int, device, dtype):
    """Phi (inverse DST), Psi (forward DST) for size N, matching mmconv.py."""
    idx = torch.arange(1, N + 1, device=device, dtype=dtype)
    Phi = torch.sin(idx[:, None] * idx[None, :] * math.pi / (N + 1))     # (N, N)
    Psi = (2.0 / (N + 1)) * Phi
    return Phi, Psi


def _dst2d_gemm(x: torch.Tensor, M_H: torch.Tensor, M_W: torch.Tensor) -> torch.Tensor:
    """Shared 2D DST as two fused GEMMs (≈3× faster than the 3-operand einsum).

    y[b, k, l] = Σ_{j,m} M_H[k,j] · M_W[l,m] · x[b, j, m]
    x: (batch, H, W), M_H: (H, H), M_W: (W, W) → (batch, H, W).
    """
    batch, H, W = x.shape
    z = x @ M_W.t()                                  # contract W: one batched cuBLAS GEMM
    z = z.transpose(0, 1).reshape(H, batch * W)      # (H, batch·W)
    z = M_H @ z                                      # contract H: one big cuBLAS GEMM
    z = z.reshape(H, batch, W).transpose(0, 1)
    return z.contiguous()


class _SpectralMMConvBase(nn.Module):
    """Shared lift/unlift logic. Subclasses bind a specific ParaRNN cell."""

    cell: nn.Module       # set by subclass

    def __init__(self, D: int, H: int, W: int, device, dtype,
                 alpha_H_init: float = 1.0, alpha_W_init: float = 1.0):
        super().__init__()
        self.D, self.H, self.W = D, H, W

        # Per-channel α (positive via log-space). α=1  ⇒  σ=ρ=1  ⇒  shared DST.
        log_aH = math.log(alpha_H_init) if alpha_H_init > 0 else 0.0
        log_aW = math.log(alpha_W_init) if alpha_W_init > 0 else 0.0
        self.log_alpha_H = nn.Parameter(torch.full([D], log_aH, device=device, dtype=dtype))
        self.log_alpha_W = nn.Parameter(torch.full([D], log_aW, device=device, dtype=dtype))

        # Fixed (untwisted) DST buffers.
        Phi_H, Psi_H = _build_dst_matrices(H, device, dtype)
        Phi_W, Psi_W = _build_dst_matrices(W, device, dtype)
        self.register_buffer('Psi_H', Psi_H)              # (H, H)
        self.register_buffer('Psi_W', Psi_W)              # (W, W)
        self.register_buffer('Phi_H', Phi_H)              # (H, H)
        self.register_buffer('Phi_W', Phi_W)              # (W, W)
        # Spatial-position indices, used to build the per-channel α^(j/2) twists.
        self.register_buffer('jH', torch.arange(1, H + 1, device=device, dtype=dtype))
        self.register_buffer('jW', torch.arange(1, W + 1, device=device, dtype=dtype))

    def _twist_vectors(self) -> typ.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-channel twist VECTORS (σ_H, σ_W, ρ_H, ρ_W), each (D, ·).

        The α-twist is diagonal in the spatial domain, so it factors out of the
        DST as an elementwise pre-scale (lift) / post-scale (unlift), leaving a
        *shared* (channel-independent) DST — which is a single fused GEMM rather
        than a batched per-channel one.
        """
        aH = self.log_alpha_H.exp()                       # (D,)
        aW = self.log_alpha_W.exp()
        sigma_H = aH[:, None] ** (-self.jH / 2)           # (D, H)
        sigma_W = aW[:, None] ** (-self.jW / 2)           # (D, W)
        rho_H   = aH[:, None] ** ( self.jH / 2)
        rho_W   = aW[:, None] ** ( self.jW / 2)
        return sigma_H, sigma_W, rho_H, rho_W

    def _lift(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, H, W, D)  →  z: (B, T, D·H·W) — spectral state, flat.

        z[b,t,d,k,l] = Σ_{j,m} Ψ_H[k,j] Ψ_W[l,m] · (σ_H[d,j] σ_W[d,m] x[b,t,j,m,d])
        — twist as elementwise pre-scale, then shared 2D DST via fused GEMM.
        """
        B, T, H, W, D = x.shape
        sigma_H, sigma_W, _, _ = self._twist_vectors()
        # elementwise twist: multiply x[...,j,m,d] by σ_H[d,j]·σ_W[d,m]
        x = x * sigma_H.t()[None, None, :, None, :] * sigma_W.t()[None, None, None, :, :]
        xr = x.permute(0, 1, 4, 2, 3).reshape(B * T * D, H, W)   # (B·T·D, H, W)
        z = _dst2d_gemm(xr, self.Psi_H, self.Psi_W)              # shared DST
        return z.reshape(B, T, D, H, W).reshape(B, T, D * H * W)

    def _unlift(self, h_spec: torch.Tensor) -> torch.Tensor:
        """h_spec: (B, T, D·H·W)  →  y: (B, T, H, W, D) — spatial.

        y[b,t,i,j,d] = ρ_H[d,i] ρ_W[d,j] · Σ_{k,l} Φ_H[i,k] Φ_W[j,l] h[b,t,d,k,l]
        — shared inverse 2D DST via fused GEMM, then twist as elementwise post-scale.
        """
        B, T, _ = h_spec.shape
        D, H, W = self.D, self.H, self.W
        _, _, rho_H, rho_W = self._twist_vectors()
        h = h_spec.reshape(B * T * D, H, W)
        y = _dst2d_gemm(h, self.Phi_H, self.Phi_W)               # shared inverse DST
        y = y.reshape(B, T, D, H, W).permute(0, 1, 3, 4, 2)      # (B, T, H, W, D)
        y = y * rho_H.t()[None, None, :, None, :] * rho_W.t()[None, None, None, :, :]
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, H, W, D_in)  →  y: (B, T, H, W, D_out)."""
        z = self._lift(x)              # (B, T, D·H·W)  — note: D_in == D for this baseline
        h_spec = self.cell(z)          # ParaRNN handles parallel-in-T
        y = self._unlift(h_spec)
        return y


class SpectralMMConvGRU(_SpectralMMConvBase):
    """Spectral MM-Conv GRU. Lift → ParaRNN MMConvGRUDiag cell → unlift.

    `num_heads` defaults to H·W: one D×D input projection per spatial position,
    which mirrors how a 1×1 conv treats channels. With num_heads=1 you instead
    get a fully-connected (D·H·W)² input projection — quadratic in state-dim,
    which kills scaling and wastes memory. Don't do that unless you have a
    reason.
    """

    def __init__(
            self,
            D: int, H: int, W: int,
            num_heads: int = None,
            nonlin_update: str = 'sigmoid',
            nonlin_reset: str = 'sigmoid',
            nonlin_state: str = 'tanh',
            alpha_H_init: float = 1.0,
            alpha_W_init: float = 1.0,
            mode: str = 'parallel',
            device: torch.device = torch.device('cpu'),
            dtype: torch.dtype = torch.float32,
    ):
        super().__init__(D, H, W, device, dtype, alpha_H_init, alpha_W_init)
        state_dim = D * H * W
        if num_heads is None:
            num_heads = H * W
        # input_dim equals state_dim here: input is spectrally-lifted with same D
        # channels per spatial position. To accept a different D_in, add an extra
        # 1×1 channel projection before the lift.
        config = MMConvGRUDiagConfig(
            state_dim=state_dim,
            input_dim=state_dim,
            device=device, dtype=dtype,
            mode=mode,
            num_heads=num_heads,
            nonlin_update=nonlin_update,
            nonlin_reset=nonlin_reset,
            nonlin_state=nonlin_state,
            H=H, W=W,
        )
        self.cell = MMConvGRUDiagCell(config)


class SpectralMMConvLSTM(_SpectralMMConvBase):
    """Spectral MM-Conv CIFG-LSTM. Lift → ParaRNN MMConvLSTMCIFGDiag cell → unlift.

    See `SpectralMMConvGRU` for the `num_heads` discussion — default H·W gives
    per-spatial-position D×D input projections (analogous to a 1×1 conv) and
    linear-in-D scaling. num_heads=1 wastes memory and ruins scaling.
    """

    def __init__(
            self,
            D: int, H: int, W: int,
            num_heads: int = None,
            nonlin_f: str = 'sigmoid',
            nonlin_o: str = 'sigmoid',
            nonlin_c: str = 'tanh',
            nonlin_state: str = 'tanh',
            alpha_H_init: float = 1.0,
            alpha_W_init: float = 1.0,
            mode: str = 'parallel',
            device: torch.device = torch.device('cpu'),
            dtype: torch.dtype = torch.float32,
    ):
        super().__init__(D, H, W, device, dtype, alpha_H_init, alpha_W_init)
        state_dim = D * H * W
        if num_heads is None:
            num_heads = H * W
        config = MMConvLSTMCIFGDiagConfig(
            state_dim=state_dim,
            input_dim=state_dim,
            device=device, dtype=dtype,
            mode=mode,
            num_heads=num_heads,
            nonlin_f=nonlin_f,
            nonlin_o=nonlin_o,
            nonlin_c=nonlin_c,
            nonlin_state=nonlin_state,
            H=H, W=W,
        )
        self.cell = MMConvLSTMCIFGDiagCell(config)
