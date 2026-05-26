"""Integration tests for the MM-Conv × ParaRNN combination.

Run with the GenT3 .venv after pararnn is installed:
    .venv/bin/python -m pararnn_integration.test_integration
"""

import time
import torch

from pararnn.rnn_cell.rnn_cell_application import RNNCellApplicationMode

from pararnn_integration import (
    MMConvGRUDiagConfig, MMConvGRUDiagCell,
    MMConvLSTMCIFGDiagConfig, MMConvLSTMCIFGDiagCell,
    SpectralMMConvGRU, SpectralMMConvLSTM,
)

torch.manual_seed(42)


def _sequential_vs_parallel(model: torch.nn.Module, x: torch.Tensor, *, label: str):
    """Run model in parallel mode, then sequential, comparing fwd outputs and grads.

    Assumes `model` is a BaseRNNCell or anything with a `mode` attribute that
    accepts RNNCellApplicationMode.{PARALLEL, SEQUENTIAL, PARALLEL_CUDA}.
    """
    print(f"\n=== {label} ===")
    model.zero_grad()
    x_par = x.detach().clone().requires_grad_(True)

    model.mode = RNNCellApplicationMode.PARALLEL
    t = time.time()
    y_par = model(x_par)
    l_par = (y_par ** 2).sum()
    l_par.backward()
    par_t = time.time() - t
    par_x_grad = x_par.grad.detach().clone()
    par_param_grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    y_par_detached = y_par.detach().clone()
    print(f"parallel:    {par_t*1000:7.1f} ms   fwd output norm = {y_par_detached.norm():.3e}")

    model.zero_grad()
    x_seq = x.detach().clone().requires_grad_(True)
    model.mode = RNNCellApplicationMode.SEQUENTIAL
    t = time.time()
    y_seq = model(x_seq)
    l_seq = (y_seq ** 2).sum()
    l_seq.backward()
    seq_t = time.time() - t
    seq_x_grad = x_seq.grad.detach().clone()
    seq_param_grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    y_seq_detached = y_seq.detach().clone()
    print(f"sequential:  {seq_t*1000:7.1f} ms   fwd output norm = {y_seq_detached.norm():.3e}")
    print(f"speedup:     {seq_t/par_t:5.2f}×")

    fwd_err = (y_par_detached - y_seq_detached).abs().max().item()
    x_grad_err = (par_x_grad - seq_x_grad).abs().max().item()
    print(f"max |Δfwd|      = {fwd_err:.2e}")
    print(f"max |Δgrad_x|   = {x_grad_err:.2e}")
    for n in par_param_grads:
        if n in seq_param_grads:
            e = (par_param_grads[n] - seq_param_grads[n]).abs().max().item()
            print(f"max |Δgrad {n}| = {e:.2e}")

    if torch.cuda.is_available() and x.device.type == 'cuda':
        model.zero_grad()
        x_cuda = x.detach().clone().requires_grad_(True)
        model.mode = RNNCellApplicationMode.PARALLEL_CUDA
        t = time.time()
        y_pc = model(x_cuda)
        l_pc = (y_pc ** 2).sum()
        l_pc.backward()
        pc_t = time.time() - t
        print(f"parallel_CUDA: {pc_t*1000:5.1f} ms   max |Δvs seq| = "
              f"{(y_pc.detach() - y_seq_detached).abs().max().item():.2e}")


def test_lift_unlift_roundtrip():
    """With α=1 and the cell replaced by identity, lift+unlift should be ≈ identity."""
    print("\n=== α=1 lift→unlift roundtrip ===")
    D, H, W = 4, 5, 6
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m = SpectralMMConvGRU(D=D, H=H, W=W, mode='sequential', device=device)
    x = torch.randn(2, 3, H, W, D, device=device)
    z = m._lift(x)
    y = m._unlift(z)
    err = (y - x).abs().max().item()
    print(f"max |y - x| = {err:.2e}  (should be ~1e-6, DST is a unitary involution up to (2/(N+1)) norm)")


def test_mmconv_gru():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    D, H, W = 4, 4, 4
    config = MMConvGRUDiagConfig(
        state_dim=D*H*W, input_dim=D*H*W,
        device=device, mode='parallel',
        H=H, W=W, num_heads=4,
    )
    model = MMConvGRUDiagCell(config)
    x = torch.randn(2, 32, config.input_dim, device=device)
    _sequential_vs_parallel(model, x, label=f"MMConvGRUDiagCell  D={D} H=W={H} T=32")


def test_mmconv_lstm():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    D, H, W = 4, 4, 4
    config = MMConvLSTMCIFGDiagConfig(
        state_dim=D*H*W, input_dim=D*H*W,
        device=device, mode='parallel',
        H=H, W=W, num_heads=4,
    )
    model = MMConvLSTMCIFGDiagCell(config)
    x = torch.randn(2, 32, config.input_dim, device=device)
    _sequential_vs_parallel(model, x, label=f"MMConvLSTMCIFGDiagCell  D={D} H=W={H} T=32")


def test_spectral_gru_endtoend():
    """Full SpectralMMConvGRU: spatial input → lift → cell → unlift → spatial output."""
    print("\n=== SpectralMMConvGRU end-to-end (spatial in / spatial out) ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    D, H, W = 4, 4, 4
    model = SpectralMMConvGRU(D=D, H=H, W=W, num_heads=4, mode='parallel', device=device)
    x = torch.randn(2, 32, H, W, D, device=device, requires_grad=True)
    y = model(x)
    loss = (y ** 2).sum()
    loss.backward()
    print(f"output shape = {tuple(y.shape)}, output norm = {y.norm():.3e}")
    print(f"alpha_H gradient norm = {model.log_alpha_H.grad.norm():.3e}  "
          f"(non-zero ⇒ α is learning)")
    print(f"cell.a_mm gradient norm = {model.cell.a_mm.grad.norm():.3e}")


if __name__ == '__main__':
    print(f"device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    test_lift_unlift_roundtrip()
    test_mmconv_gru()
    test_mmconv_lstm()
    test_spectral_gru_endtoend()
