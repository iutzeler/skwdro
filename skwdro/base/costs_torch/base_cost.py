from typing import Optional, Tuple
import torch as pt
import torch.nn as nn
import torch.distributions as dst

from skwdro.base.costs import ENGINES_NAMES

class TorchCost(nn.Module):
    """ Base class for transport functions """
    def __init__(self, name: str="", engine: str=""):
        super(TorchCost, self).__init__()
        self.name = name
        self.engine = engine

    def forward(
            self,
            xi: pt.Tensor,
            zeta: pt.Tensor,
            xi_labels: Optional[pt.Tensor]=None,
            zeta_labels: Optional[pt.Tensor]=None
            ) -> pt.Tensor:
        return self.value(xi, zeta, xi_labels, zeta_labels)

    def value(
            self,
            xi: pt.Tensor,
            zeta: pt.Tensor,
            xi_labels: Optional[pt.Tensor]=None,
            zeta_labels: Optional[pt.Tensor]=None
            ) -> pt.Tensor:
        raise NotImplementedError("Please Implement this method")

    def sampler(
            self,
            xi: pt.Tensor,
            xi_labels: pt.Tensor,
            epsilon: pt.Tensor
            ) -> Tuple[dst.Distribution, Optional[dst.Distribution]]:
        return self._sampler_data(xi, epsilon), self._sampler_labels(xi_labels, epsilon)

    def _sampler_data(
            self,
            xi: pt.Tensor,
            epsilon: pt.Tensor
            ) -> dst.Distribution:
        raise NotImplementedError()

    def _sampler_labels(
            self,
            xi_labels: pt.Tensor,
            epsilon: pt.Tensor
            ) -> Optional[dst.Distribution]:
        raise NotImplementedError()

    def __str__(self) -> str:
        return "Cost named " + self.name + " using as data: " + ENGINES_NAMES[self.engine]

    def solve_max_series_exp(
            self,
            xi: pt.Tensor,
            xi_labels: Optional[pt.Tensor],
            rhs: pt.Tensor,
            rhs_labels: Optional[pt.Tensor]
            ) -> pt.Tensor:
        raise NotImplementedError()
