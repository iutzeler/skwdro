from typing import Optional
from itertools import chain
import torch as pt
import torch.nn as nn

from skwdro.base.losses_torch import Loss
from skwdro.base.samplers.torch.base_samplers import BaseSampler


class WrappingError(ValueError):
    pass


class WrappedPrimalLoss(Loss):
    def __init__(
            self,
            loss: nn.Module,
            transform: Optional[nn.Module],
            sampler: BaseSampler,
            has_labels: bool,
            *,
            l2reg: Optional[float] = None
    ) -> None:
        super(WrappedPrimalLoss, self).__init__(sampler, l2reg=l2reg)
        self.loss = loss
        self.transform = transform if transform is not None else nn.Identity()
        self.has_labels = has_labels

    @classmethod
    def default_sampler(cls, xi, xi_labels, epsilon, seed: int):
        raise WrappingError(
            "No default sampler can be attributed by default by a wrapped loss.")

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

    def _flat_value_w_labels(self, xi, xi_labels):
        return self.regularize(self.loss(self.transform(xi), xi_labels))

    def _flat_value_wo_labels(self, xi):
        return self.regularize(self.loss(self.transform(xi)))

    def value(self, xi: pt.Tensor, xi_labels: Optional[pt.Tensor] = None):
        if self.has_labels:
            assert xi_labels is not None
            if xi.dim() > 2 and xi_labels.dim() > 2:
                *b, _ = xi.size()
                return self._flat_value_w_labels(xi.flatten(start_dim=0, end_dim=-2), xi_labels.flatten(start_dim=0, end_dim=-2)).view(*b, 1)
            elif xi.dim() > 2 and xi_labels.dim() == 2:
                *b, _ = xi.size()
                return self._flat_value_w_labels(xi.flatten(start_dim=0, end_dim=-2), xi_labels.squeeze()).view(*b, 1)
            elif xi.dim() == 2 and xi_labels.dim() <= 2:
                return self._flat_value_w_labels(xi, xi_labels).unsqueeze(-1)
            elif xi.dim() == xi_labels.dim() == 1:
                return self._flat_value_w_labels(xi, xi_labels)
            else:
                raise NotImplementedError()
        else:
            assert xi_labels is None
            if xi.dim() > 2:
                *b, _ = xi.size()
                return self._flat_value_wo_labels(xi.flatten(start_dim=0, end_dim=-2)).view(*b, 1)
            elif xi.dim() == 2:
                return self._flat_value_wo_labels(xi).unsqueeze(-1)
            elif xi.dim() == 1:
                return self._flat_value_wo_labels(xi)
            else:
                raise NotImplementedError()
