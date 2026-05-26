"""Diagonal-multihead GRU whose state-mixing vector A is MM-Conv-parameterized.

Reuses ParaRNN's GRUDiagMHImpl unchanged. Only the cell's `_specific_init`
and `_system_parameters` differ: instead of a free (3, D·H·W) `A` parameter,
A is materialized each forward from four small (3, D) tensors (a, b, c, d)
via the MM-Conv spectral form

    A[v, d, k, l] = a[v,d]
                  + b[v,d] · ξ_H[k]
                  + c[v,d] · ξ_W[l]
                  + d[v,d] · ξ_H[k] · ξ_W[l]                          (v = z, r, h gate)

and flattened to (3, D·H·W) before being handed to the unchanged GRU impl.
The diagonal-Jacobian parallel reduction (and `parallel_reduce_diag_cuda`)
works at any state size, so D·H·W is fine.
"""

from dataclasses import dataclass
import math
import typing as typ

import torch
import torch.nn as nn

from pararnn.rnn_cell.rnn_cell import BaseRNNCell
from pararnn.rnn_cell.rnn_cell_utils import Config
from pararnn.rnn_cell.gru_diag_mh import (
    GRUDiagMHTrait,
    GRUDiagMHSystemParameters,
    GRUDiagMHImpl,
)
from pararnn.utils.init import INIT_REGISTRY


@dataclass
class MMConvGRUDiagConfig(Config[GRUDiagMHTrait]):
    """Config for a GRU-diag whose A vector is MM-Conv-parameterized.

    `state_dim` must equal `D * H * W` where D is the channel count and (H, W)
    is the spatial size of the underlying feature map. `num_heads` partitions
    the input projection in the usual GRUDiagMH way and must divide both
    `state_dim` and `input_dim`.
    """
    # GRUDiagMH fields (duplicated so the dataclass picks them up cleanly):
    nonlin_update: str = 'sigmoid'
    nonlin_reset: str = 'sigmoid'
    nonlin_state: str = 'tanh'
    num_heads: int = 2
    a_init_fn: str = 'xlstm'
    b_init_fn: str = 'bias_minus_linspace'
    w_init_fn: str = 'xavier_uniform'
    # MM-Conv additions:
    H: int = 8
    W: int = 8


class MMConvGRUDiagCell(BaseRNNCell[
    MMConvGRUDiagConfig, GRUDiagMHSystemParameters, GRUDiagMHImpl
]):
    """ParaRNN cell with MM-Conv-parameterized diagonal state mixing.

    Learnable parameters
    --------------------
    a, b, c, d : (3, D)   per-gate MM-Conv spectral coefficients
    B          : input projection (same as GRUDiagMH)
    bias       : (3, D·H·W)
    """

    def _specific_init(self, config: MMConvGRUDiagConfig):
        self.H = config.H
        self.W = config.W
        assert config.state_dim % (config.H * config.W) == 0, (
            f"state_dim {config.state_dim} must be a multiple of H*W = {config.H*config.W}"
        )
        self.D = config.state_dim // (config.H * config.W)

        assert self.input_dim % config.num_heads == 0, \
            "num_heads must exactly divide input_dim"
        assert self.state_dim % config.num_heads == 0, \
            "num_heads must exactly divide state_dim"
        self.num_heads = config.num_heads
        self.head_input_dim = self.input_dim // self.num_heads
        self.head_state_dim = self.state_dim // self.num_heads

        # MM-Conv spectral coefficients — 4·D scalars per gate, three gates (z, r, h).
        self.a_mm = nn.Parameter(torch.empty([3, self.D], device=self.device, dtype=self.dtype))
        self.b_mm = nn.Parameter(torch.empty([3, self.D], device=self.device, dtype=self.dtype))
        self.c_mm = nn.Parameter(torch.empty([3, self.D], device=self.device, dtype=self.dtype))
        self.d_mm = nn.Parameter(torch.empty([3, self.D], device=self.device, dtype=self.dtype))

        # ξ_H, ξ_W: cosine factors from the DST eigendecomposition (fixed buffers).
        idxH = torch.arange(1, config.H + 1, device=self.device, dtype=self.dtype)
        idxW = torch.arange(1, config.W + 1, device=self.device, dtype=self.dtype)
        self.register_buffer('xi_H', torch.cos(idxH * math.pi / (config.H + 1)))
        self.register_buffer('xi_W', torch.cos(idxW * math.pi / (config.W + 1)))

        # Input projection & bias — same structure as GRUDiagMH.
        self.B = nn.Parameter(torch.empty(
            [self.num_heads, self.head_input_dim, 3, self.head_state_dim],
            device=self.device, dtype=self.dtype
        ))
        self.bias = nn.Parameter(torch.empty(
            [3, self.state_dim], device=self.device, dtype=self.dtype
        ))

        # Gate nonlinearities (sigmoid/sigmoid/tanh by default).
        self.nonlin_update, self.derivative_nonlin_update = \
            self._set_nonlinearity_and_derivative(config.nonlin_update)
        self.nonlin_reset, self.derivative_nonlin_reset = \
            self._set_nonlinearity_and_derivative(config.nonlin_reset)
        self.nonlin_state, self.derivative_nonlin_state = \
            self._set_nonlinearity_and_derivative(config.nonlin_state)

        self.reset_parameters()

    def _build_A(self) -> torch.Tensor:
        """Materialize A = a + b·ξ_H + c·ξ_W + d·ξ_H·ξ_W  →  (3, D·H·W)."""
        xh = self.xi_H[None, None, :, None]            # (1, 1, H, 1)
        xw = self.xi_W[None, None, None, :]            # (1, 1, 1, W)
        a = self.a_mm[..., None, None]                 # (3, D, 1, 1)
        b = self.b_mm[..., None, None] * xh
        c = self.c_mm[..., None, None] * xw
        d = self.d_mm[..., None, None] * (xh * xw)
        A = a + b + c + d                              # (3, D, H, W)
        return A.reshape(3, self.D * self.H * self.W)

    @property
    def _system_parameters(self) -> GRUDiagMHSystemParameters:
        return GRUDiagMHSystemParameters(
            A=self._build_A(),
            B=self.B,
            b=self.bias,
            nonlin_update=self.nonlin_update,
            nonlin_reset=self.nonlin_reset,
            nonlin_state=self.nonlin_state,
            derivative_nonlin_update=self.derivative_nonlin_update,
            derivative_nonlin_reset=self.derivative_nonlin_reset,
            derivative_nonlin_state=self.derivative_nonlin_state,
        )

    @torch.no_grad()
    def reset_parameters(self):
        super().reset_parameters()
        # Init a_mm with GRUDiagMH's `a_init_fn` so that at start A ≈ that init
        # (b, c, d start at zero → no spectral modulation initially).
        INIT_REGISTRY[self._config.a_init_fn](self.a_mm.data, fan_in=self.D, fan_out=None)
        self.b_mm.data.zero_()
        self.c_mm.data.zero_()
        self.d_mm.data.zero_()
        INIT_REGISTRY[self._config.w_init_fn](
            self.B.data, fan_in=self.head_input_dim, fan_out=self.state_dim
        )
        INIT_REGISTRY[self._config.b_init_fn](
            self.bias.data, fan_in=None, fan_out=self.bias.numel()
        )
