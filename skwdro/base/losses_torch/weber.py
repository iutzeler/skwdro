from typing import Optional
from types import NoneType

import torch as pt
import torch.nn as nn


from .base_loss import Loss
from skwdro.base.samplers.torch.base_samplers import LabeledSampler
from skwdro.base.samplers.torch.classif_sampler import ClassificationNormalNormalSampler

class SimpleWeber(nn.Module):
    """
    Exemple of a short code to write a Weber loss.
    """
    def __init__(self, d: int) -> None:
        super(SimpleWeber, self).__init__()
        self.pos = nn.Parameter(pt.zeros(d))
        self.d = d

    def forward(self, xi: pt.Tensor, xi_labels: pt.Tensor) -> pt.Tensor:
        distances = pt.linalg.norm(xi - self.pos.unsqueeze(0), dim=-1, keepdims=True)
        val = xi_labels * distances * xi_labels.shape[1]
        return val


class WeberLoss(Loss):
    r"""
    The "data" dimension (first) represents the various anchor points of the problem specification
    (e.g the various factories).
    The "x" tensor is for the d-dimensional coordinates of said points,
    and the "y"/"label" tensor is for the ground costs (e.g. the train cost per unit distance).
    The problem writes:

    ..math::
        \min_\theta \mathbb{E}\left[\xi_\text{labels}.\|\xi_\text{data}-theta\|\right]
    """
    def __init__(
            self,
            sampler: LabeledSampler,
            *,
            d: int=2,
            l2reg: Optional[float]=None
            ):
        super(WeberLoss, self).__init__(sampler, l2reg=l2reg)
        self.d = d
        self.pos = nn.Parameter(pt.zeros(d))

    def value(self, xi: pt.Tensor, xi_labels: Optional[pt.Tensor]):
        """
        The function the :py:method:`~Loss.forward` method will call internally.
        """
        assert xi_labels is not None
        distances = pt.linalg.norm(xi - self.pos.unsqueeze(0), dim=-1, keepdims=True)
        val = xi_labels * distances * xi_labels.shape[0]
        return val

    @classmethod
    def default_sampler(cls, xi, xi_labels, epsilon, seed: int) -> LabeledSampler:
        assert xi_labels is not None
        return ClassificationNormalNormalSampler(xi, xi_labels, seed, sigma=epsilon, l_sigma=epsilon)

    @property
    def theta(self) -> pt.Tensor:
        return self.pos

    @property
    def intercept(self) -> NoneType:
        return None
