r"""
Shallow Neural Network
======================

We illustrate how to use a simple shallow neural network.

"""

from typing import Iterable
import matplotlib.pyplot as plt
import torch as pt
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from skwdro.wrap_problem import dualize_primal_loss
from skwdro.solvers.oracle_torch import DualLoss
from skwdro.base.losses_torch.wrapper import WrappedPrimalLoss

class MyShallowNet(nn.Module):
    def __init__(self, spec: list[int]) -> None:
        super(MyShallowNet, self).__init__()
        assert len(spec) > 1
        self.layers = pt.compile(nn.Sequential(
                    *([
                        nn.Sequential( # N layers
                            nn.Linear(fan_in, fan_out), # A linear layer from k to k+1
                            nn.BatchNorm1d(fan_out),
                            nn.LeakyReLU(), # A Rectified linear unit for activation
                        ) for fan_out, fan_in in zip(spec[1:-1], spec[:-2])
                    ] + [nn.Dropout1d(p=.01), nn.Linear(spec[-2], spec[-1])])
                ))
    def forward(self, signal: pt.Tensor) -> pt.Tensor:
        if signal.dim() == 2:
            return self.layers(signal)
        elif signal.dim() == 3:
            n, m, d = signal.shape
            return self.layers(signal.flatten(start_dim=0, end_dim=1)).reshape(n, m, d)
        else:
            raise NotImplementedError("Please flatten your data appropriately")


def train(dual_loss: DualLoss, dataset: Iterable[tuple[pt.Tensor, pt.Tensor]], epochs: int=10):
    optimizer = pt.optim.AdamW(dual_loss.parameters(), lr=1e-2)
    pbar = tqdm(range(epochs))

    for _ in pbar:
        # Every now and then, try to rectify the dual parameter (e.g. once per epoch).
        dual_loss.get_initial_guess_at_dual(*next(iter(dataset))) # *

        # Main train loop
        inpbar = tqdm(dataset, leave=False)
        for xi, xi_label in inpbar:
            optimizer.zero_grad()

            # Forward the batch
            loss = dual_loss(xi, xi_label).mean()

            # Backward pass
            loss.backward()
            optimizer.step()

            inpbar.set_postfix({"loss": f"{loss.item():.2f}"})
        pbar.set_postfix({"lambda": f"{dual_loss.lam.item():.2f}"})
    assert isinstance(dual_loss.primal_loss, WrappedPrimalLoss)
    return dual_loss.primal_loss.transform

def f(x): return pt.sin(2. * pt.pi * x)

def main():
    device = "cuda" if pt.cuda.is_available() else "cpu"
    model = MyShallowNet([1, 10, 5, 1]).to(device)

    rho = pt.tensor(1e-1).to(device)

    x = pt.sort(pt.flatten(
        pt.linspace(0., 1., 10, device=device).unsqueeze(0)\
        + pt.randn(10, 10, device=device) * 1e-1
    ))[0]
    y = f(x) + pt.randn(100, device=device) * 2e-2
    dataset = DataLoader(TensorDataset(x.unsqueeze(-1), y.unsqueeze(-1)), batch_size=50, shuffle=True)

    # New line: "dualize" the loss
    dual_loss = dualize_primal_loss(
            nn.MSELoss(reduction='none'),
            model,
            rho,
            x.unsqueeze(-1),
            y.unsqueeze(-1)
        )

    model = train(dual_loss, dataset, 10) # type: ignore
    model.eval()

    fig, ax = plt.subplots()
    ax.scatter(x.cpu(), y.cpu(), c='g', label='train data')
    ax.plot(x.cpu(), f(x).cpu(), 'k', label='ground truth')
    ax.scatter(x.cpu(), model(x.unsqueeze(-1)).detach().cpu().squeeze(), marker='+', c='r', label='outputs')

    fig.legend()
    plt.show()

if __name__ == '__main__':
    pt.set_float32_matmul_precision('high')
    main()
