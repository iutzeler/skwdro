from typing import Optional


import torch as pt
import torch.nn as nn

from .base_loss import Loss
from skwdro.base.samplers.torch.base_samplers import (
    BaseSampler
)
from skwdro.base.samplers.torch.classif_sampler import (
    ClassificationNormalNormalSampler
)
from skwdro.base.samplers.torch.base_samplers import LabeledSampler


class QuadraticLoss(Loss):
    def __init__(
        self,
        sampler: LabeledSampler,
        *,
        d: int = 0,
        l2reg: Optional[float] = None,
        fit_intercept: bool = False
    ) -> None:
        super(QuadraticLoss, self).__init__(sampler, l2reg=l2reg)
        assert d > 0, "Please provide a valid data dimension d>0"
        self.linear = nn.Linear(d, 1, bias=fit_intercept)
        self.L = nn.MSELoss(reduction='none')

    def regression(self, X) -> pt.Tensor:
        coefs = self.linear(X)
        assert isinstance(coefs, pt.Tensor)
        return coefs

    def value(
        self, xi: pt.Tensor, xi_labels: Optional[pt.Tensor]
    ) -> pt.Tensor:
        assert xi_labels is not None
        coefs = self.regression(xi)
        return self.regularize(self.L(coefs, xi_labels))

    @classmethod
    def default_sampler(cls, xi, xi_labels, epsilon, seed: int) -> BaseSampler:
        return ClassificationNormalNormalSampler(
            xi, xi_labels, seed,
            sigma=epsilon, l_sigma=epsilon
        )

    @property
    def theta(self) -> pt.Tensor:
        return self.linear.weight

    @property
    def intercept(self) -> pt.Tensor:
        return self.linear.bias
