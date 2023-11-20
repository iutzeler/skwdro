from typing import Optional
from itertools import chain
import torch as pt
import torch.nn as nn

from skwdro.base.losses_torch import Loss
from skwdro.base.samplers.torch.base_samplers import BaseSampler

class WrappedPrimalLoss(Loss):
    def __init__(
            self,
            loss: nn.Module,
            transform: Optional[nn.Module],
            sampler: BaseSampler,
            has_labels: bool,
            *,
            l2reg: Optional[float]=None
            ) -> None:
        super(WrappedPrimalLoss, self).__init__(sampler, l2reg=l2reg)
        self.loss = loss
        self.transform = transform if transform is not None else nn.Identity()
        self.has_labels = has_labels

    @property
    def theta(self):
        return pt.concat(list(
            map(
                pt.flatten,
                chain(self.loss.parameters(), self.transform.parameters())
            )))

    @property
    def intercept(self):
        return pt.tensor(0.)

    def value(self, xi: pt.Tensor, xi_labels: Optional[pt.Tensor]=None):
        if self.has_labels:
            assert xi_labels is not None
            return self.regularize(self.loss(self.transform(xi), xi_labels))
        else:
            assert xi_labels is None
            return self.regularize(self.loss(self.transform(xi)))
