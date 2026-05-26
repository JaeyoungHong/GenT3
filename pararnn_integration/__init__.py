"""MM-Conv × ParaRNN integration.

Provides ParaRNN cells whose diagonal state-mixing vector A is parameterized by
MM-Conv's spectral form (a, b, c, d, ξ_H, ξ_W), plus thin nn.Module wrappers
that handle the α-twisted DST lift / unlift around the cell.
"""

from .mmconv_gru_diag import (
    MMConvGRUDiagConfig,
    MMConvGRUDiagCell,
)
from .mmconv_lstm_cifg_diag import (
    MMConvLSTMCIFGDiagConfig,
    MMConvLSTMCIFGDiagCell,
)
from .spectral_wrapper import (
    SpectralMMConvGRU,
    SpectralMMConvLSTM,
)

__all__ = [
    "MMConvGRUDiagConfig",
    "MMConvGRUDiagCell",
    "MMConvLSTMCIFGDiagConfig",
    "MMConvLSTMCIFGDiagCell",
    "SpectralMMConvGRU",
    "SpectralMMConvLSTM",
]
