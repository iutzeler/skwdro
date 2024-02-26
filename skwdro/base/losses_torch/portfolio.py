import torch as pt
import torch.nn as nn
import torch.nn.functional as fl
import torch.nn.utils.parametrize as pz

class SimplePortfolio(nn.Module):

    def __init__(self, d: int, risk_aversion: float, risk_level: float) -> None:
        super(SimplePortfolio, self).__init__()
        self.assets = nn.Linear(d, 1, bias=False)#nn.Parameter(pt.zeros(1, d))
        pz.register_parametrization(self.assets, "weight", nn.Softmax(dim=-1))
        self.eta = pt.tensor(risk_aversion)
        self.alpha = pt.tensor(risk_level)
        self.tau = nn.Parameter(pt.tensor(0.))

    def forward(self, xi: pt.Tensor) -> pt.Tensor:
        returns = self.assets(xi)
        cvar = self.tau + fl.relu(-returns - self.tau) / self.alpha
        return -returns + self.eta * cvar
