"""CIFG-LSTM diagonal-multihead with MM-Conv-parameterized A and C vectors.

Reuses ParaRNN's LSTMCIFGDiagMHImpl unchanged. Two diagonal vectors get the
MM-Conv spectral parameterization:
    A : (3, D·H·W)  — state mixing for f, c-candidate, o gates
    C : (2, D·H·W)  — peephole coefficients for f and o gates on the cell state

Each is materialized at forward time from small (3,D) / (2,D) tensors
(a, b, c, d) via the MM-Conv spectral form

    X[v, d, k, l] = a[v,d] + b[v,d]·ξ_H[k] + c[v,d]·ξ_W[l] + d[v,d]·ξ_H[k]·ξ_W[l]

and flattened. Note: LSTMCIFGDiagMHImpl is a *block-diag* impl with 2 blocks
(cell-state cc and hidden-state hh); ParaRNN handles that internally.
"""

from dataclasses import dataclass
import math
import typing as typ

import torch
import torch.nn as nn

from pararnn.rnn_cell.rnn_cell import BaseRNNCell
from pararnn.rnn_cell.rnn_cell_utils import Config
from pararnn.rnn_cell.lstm_cifg_diag_mh import (
    LSTMCIFGDiagMHTrait,
    LSTMCIFGDiagMHSystemParameters,
    LSTMCIFGDiagMHImpl,
)
from pararnn.utils.init import INIT_REGISTRY


@dataclass
class MMConvLSTMCIFGDiagConfig(Config[LSTMCIFGDiagMHTrait]):
    """Config for a CIFG-LSTM-diag whose A and C vectors are MM-Conv-parameterized."""
    # LSTMCIFGDiagMH fields (duplicated for clean dataclass binding):
    nonlin_f: str = 'sigmoid'
    nonlin_o: str = 'sigmoid'
    nonlin_c: str = 'tanh'
    nonlin_state: str = 'tanh'
    num_heads: int = 2
    a_init_fn: str = 'xlstm'
    w_init_fn: str = 'xavier_uniform'
    b_init_fn: str = 'bias_minus_linspace'
    # MM-Conv additions:
    H: int = 8
    W: int = 8


class MMConvLSTMCIFGDiagCell(BaseRNNCell[
    MMConvLSTMCIFGDiagConfig, LSTMCIFGDiagMHSystemParameters, LSTMCIFGDiagMHImpl
]):
    """CIFG-LSTM cell with MM-Conv-parameterized A (state mixing) and C (peephole).

    Learnable parameters
    --------------------
    a_A, b_A, c_A, d_A : (3, D)   per-gate (f, c, o) MM-Conv coeffs for A
    a_C, b_C, c_C, d_C : (2, D)   per-gate (f, o) MM-Conv coeffs for C
    B                  : input projection (same as LSTMCIFGDiagMH)
    bias               : (3, D·H·W)
    """

    def _specific_init(self, config: MMConvLSTMCIFGDiagConfig):
        self.H = config.H
        self.W = config.W
        # config.state_dim describes the *per-block* (h-block) dim. The total stored
        # state is 2·state_dim (cell + hidden), but that doubling is handled inside
        # the parent class — see LSTMCIFGDiagMH._specific_init at the bottom.
        assert config.state_dim % (config.H * config.W) == 0, (
            f"state_dim {config.state_dim} must be a multiple of H*W = {config.H*config.W}"
        )
        self.D = config.state_dim // (config.H * config.W)

        assert self.input_dim % config.num_heads == 0
        assert self.state_dim % config.num_heads == 0
        self.num_heads = config.num_heads
        self.head_input_dim = self.input_dim // self.num_heads
        self.head_state_dim = self.state_dim // self.num_heads

        # MM-Conv coefficients for A (3 gates: f, c, o)
        self.a_A = nn.Parameter(torch.empty([3, self.D], device=self.device, dtype=self.dtype))
        self.b_A = nn.Parameter(torch.empty([3, self.D], device=self.device, dtype=self.dtype))
        self.c_A = nn.Parameter(torch.empty([3, self.D], device=self.device, dtype=self.dtype))
        self.d_A = nn.Parameter(torch.empty([3, self.D], device=self.device, dtype=self.dtype))

        # MM-Conv coefficients for C (2 peepholes: f, o)
        self.a_C = nn.Parameter(torch.empty([2, self.D], device=self.device, dtype=self.dtype))
        self.b_C = nn.Parameter(torch.empty([2, self.D], device=self.device, dtype=self.dtype))
        self.c_C = nn.Parameter(torch.empty([2, self.D], device=self.device, dtype=self.dtype))
        self.d_C = nn.Parameter(torch.empty([2, self.D], device=self.device, dtype=self.dtype))

        # ξ buffers
        idxH = torch.arange(1, config.H + 1, device=self.device, dtype=self.dtype)
        idxW = torch.arange(1, config.W + 1, device=self.device, dtype=self.dtype)
        self.register_buffer('xi_H', torch.cos(idxH * math.pi / (config.H + 1)))
        self.register_buffer('xi_W', torch.cos(idxW * math.pi / (config.W + 1)))

        # Input projection & bias
        self.B = nn.Parameter(torch.empty(
            [self.num_heads, self.head_input_dim, 3, self.head_state_dim],
            device=self.device, dtype=self.dtype
        ))
        self.bias = nn.Parameter(torch.empty(
            [3, self.state_dim], device=self.device, dtype=self.dtype
        ))

        self.nonlin_f, self.derivative_nonlin_f = \
            self._set_nonlinearity_and_derivative(config.nonlin_f)
        self.nonlin_o, self.derivative_nonlin_o = \
            self._set_nonlinearity_and_derivative(config.nonlin_o)
        self.nonlin_c, self.derivative_nonlin_c = \
            self._set_nonlinearity_and_derivative(config.nonlin_c)
        self.nonlin_state, self.derivative_nonlin_state = \
            self._set_nonlinearity_and_derivative(config.nonlin_state)

        self.reset_parameters()

        # Match the parent class trick: hidden state stores both cell & hidden, so
        # the actually-stored state dim is 2·state_dim. ParaRNN consumes this.
        self.state_dim = 2 * self.state_dim

    def _build_vec(self, a, b, c, d) -> torch.Tensor:
        """a, b, c, d each shape (V, D)  →  (V, D·H·W)."""
        xh = self.xi_H[None, None, :, None]
        xw = self.xi_W[None, None, None, :]
        out = (a[..., None, None]
             + b[..., None, None] * xh
             + c[..., None, None] * xw
             + d[..., None, None] * (xh * xw))                # (V, D, H, W)
        V = a.shape[0]
        return out.reshape(V, self.D * self.H * self.W)

    @property
    def _system_parameters(self) -> LSTMCIFGDiagMHSystemParameters:
        A = self._build_vec(self.a_A, self.b_A, self.c_A, self.d_A)   # (3, D·H·W)
        C = self._build_vec(self.a_C, self.b_C, self.c_C, self.d_C)   # (2, D·H·W)
        return LSTMCIFGDiagMHSystemParameters(
            A=A,
            B=self.B,
            C=C,
            b=self.bias,
            nonlin_f=self.nonlin_f,
            nonlin_o=self.nonlin_o,
            nonlin_c=self.nonlin_c,
            nonlin_state=self.nonlin_state,
            derivative_nonlin_f=self.derivative_nonlin_f,
            derivative_nonlin_o=self.derivative_nonlin_o,
            derivative_nonlin_c=self.derivative_nonlin_c,
            derivative_nonlin_state=self.derivative_nonlin_state,
        )

    @torch.no_grad()
    def reset_parameters(self):
        super().reset_parameters()
        INIT_REGISTRY[self._config.a_init_fn](self.a_A.data, fan_in=self.D, fan_out=None)
        INIT_REGISTRY[self._config.a_init_fn](self.a_C.data, fan_in=self.D, fan_out=None)
        for p in (self.b_A, self.c_A, self.d_A, self.b_C, self.c_C, self.d_C):
            p.data.zero_()
        INIT_REGISTRY[self._config.w_init_fn](
            self.B.data, fan_in=self.head_input_dim, fan_out=self.head_state_dim
        )
        INIT_REGISTRY[self._config.b_init_fn](
            self.bias.data, fan_in=None, fan_out=self.bias.numel()
        )
