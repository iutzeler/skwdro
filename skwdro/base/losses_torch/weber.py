from typing import Optional

import torch as pt
import torch.nn as nn


from .base_loss import Loss
from skwdro.base.samplers.torch.base_samplers import LabeledSampler
from skwdro.base.samplers.torch.classif_sampler import (
    ClassificationNormalNormalSampler
)


class SimpleWeber(nn.Module):
    def __init__(self, d: int) -> None:
        super(SimpleWeber, self).__init__()
        self.pos = nn.Parameter(pt.zeros(d))
        self.d = d

    def forward(self, xi: pt.Tensor, xi_labels: pt.Tensor) -> pt.Tensor:
        distances = pt.linalg.norm(
            xi - self.pos.unsqueeze(0), dim=-1, keepdims=True
        )
        val = xi_labels * distances * xi_labels.shape[1]
        assert isinstance(val, pt.Tensor)
        return val


class WeberLoss(Loss):
    def __init__(
            self,
            sampler: LabeledSampler,
            *,
            d: int = 2,
            l2reg: Optional[float] = None
    ):
        super(WeberLoss, self).__init__(sampler, l2reg=l2reg)
        self.d = d
        self.pos = nn.Parameter(pt.zeros(d))

    def value(self, xi: pt.Tensor, xi_labels: Optional[pt.Tensor]):
        assert xi_labels is not None
        distances = pt.linalg.norm(
            xi - self.pos.unsqueeze(0), dim=-1, keepdims=True)
        val = xi_labels * distances * xi_labels.shape[0]
        return val

    @classmethod
    def default_sampler(
        cls, xi, xi_labels, epsilon, seed: int
    ) -> LabeledSampler:
        assert xi_labels is not None
        return ClassificationNormalNormalSampler(
            xi,
            xi_labels,
            seed,
            sigma=epsilon,
            l_sigma=epsilon
        )

    @property
    def theta(self) -> pt.Tensor:
        return self.pos

    @property
    def intercept(self) -> None:
        return None
