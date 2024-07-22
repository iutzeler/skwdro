from abc import ABC, abstractmethod
from typing import Optional, Tuple
import torch as pt
import torch.nn as nn
import skwdro.distributions as dst

ENGINES_NAMES = {
    "pt": "PyTorch tensors",
    "jx": "Jax arrays"
}


class TorchCost(nn.Module, ABC):
    """ Base class for transport functions """

    def __init__(
        self,
        name: str = "",
        engine: str = ""
    ):
        super(TorchCost, self).__init__()
        self.name = name
        self.engine = engine
        # Default power needs to be overwritten
        self.power = 1.0

    def forward(
        self,
        xi: pt.Tensor,
        zeta: pt.Tensor,
        xi_labels: Optional[pt.Tensor] = None,
        zeta_labels: Optional[pt.Tensor] = None
    ) -> pt.Tensor:
        return self.value(xi, zeta, xi_labels, zeta_labels)

    @abstractmethod
    def value(
        self,
        xi: pt.Tensor,
        zeta: pt.Tensor,
        xi_labels: Optional[pt.Tensor] = None,
        zeta_labels: Optional[pt.Tensor] = None
    ) -> pt.Tensor:
        del xi, zeta, xi_labels, zeta_labels
        raise NotImplementedError("Please Implement this method")

    def sampler(
        self,
        xi: pt.Tensor,
        xi_labels: pt.Tensor,
        epsilon: pt.Tensor
    ) -> Tuple[dst.Distribution, Optional[dst.Distribution]]:
        return (
            self._sampler_data(xi, epsilon),
            self._sampler_labels(xi_labels, epsilon)
        )

    @abstractmethod
    def _sampler_data(
        self,
        xi: pt.Tensor,
        epsilon: pt.Tensor
    ) -> dst.Distribution:
        del xi, epsilon
        raise NotImplementedError()

    @abstractmethod
    def _sampler_labels(
        self,
        xi_labels: pt.Tensor,
        epsilon: pt.Tensor
    ) -> Optional[dst.Distribution]:
        del xi_labels, epsilon
        raise NotImplementedError()

    def __str__(self) -> str:
        return ' '.join([
            "Cost named",
            self.name,
            "using as data:",
            ENGINES_NAMES[self.engine]
        ])

    @abstractmethod
    def solve_max_series_exp(
        self,
        xi: pt.Tensor,
        xi_labels: Optional[pt.Tensor],
        rhs: pt.Tensor,
        rhs_labels: Optional[pt.Tensor]
    ) -> Tuple[pt.Tensor, Optional[pt.Tensor]]:
        del xi, rhs, xi_labels, rhs_labels
        raise NotImplementedError()
