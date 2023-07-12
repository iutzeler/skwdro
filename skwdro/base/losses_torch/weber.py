from typing import Optional
from types import NoneType

import torch as pt
import torch.nn as nn


from .base_loss import Loss
from skwdro.base.samplers.torch.base_samplers import LabeledSampler
from skwdro.base.samplers.torch.classif_sampler import ClassificationNormalNormalSampler


class WeberLoss(Loss):

    def __init__(
            self,
            sampler: LabeledSampler,
            *,
            name="Weber loss"):
        super(WeberLoss, self).__init__(sampler)
        self.pos = nn.Parameter(pt.tensor([0.0,0.0]))
        self.name = name


    def value(self, xi: pt.Tensor, xi_labels: pt.Tensor):
        distances = pt.linalg.norm(xi - self.pos, dim=-1)[:,:,None]
        val = xi_labels * distances
        return val

    @classmethod
    def default_sampler(cls, xi, xi_labels, epsilon, seed: int):
        return ClassificationNormalNormalSampler(xi, xi_labels, seed, sigma=epsilon, l_sigma=epsilon)

    @property
    def theta(self) -> pt.Tensor:
        return self.pos

    @property
    def intercept(self) -> NoneType:
        return None
