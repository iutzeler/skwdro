from typing import Callable, Optional, Union
from itertools import chain
import torch as pt
import torch.nn as nn

from skwdro.base.losses_torch import Loss
from skwdro.base.samplers.torch.base_samplers import BaseSampler


class WrappingError(ValueError):
    pass


class WrappedPrimalLoss(Loss):
    loss_oop_interface: bool = True
    reduce_spatial_dims: bool = True

    def __init__(
        self,
        loss: Union[
            nn.Module,
            Callable[..., pt.Tensor]
        ],
        transform: Optional[nn.Module],
        sampler: BaseSampler,
        has_labels: bool,
        reduce_spatial_dims: bool = True,
        *,
        l2reg: Optional[float] = None
    ) -> None:
        super(WrappedPrimalLoss, self).__init__(sampler, l2reg=l2reg)
        self.loss = loss

        if isinstance(loss, pt.nn.Module):
            assert loss.reduction == 'none', " ".join([
                'If you are using the OOP interface of PyTorch to define the',
                'main loss functional, please set its reduction method to',
                '\"none\"'
            ])
            self.loss_oop_interface = True
        else:
            assert callable(loss), " ".join([
                'If you are not using the OOP interface of PyTorch to define',
                'the main loss functional, please use the functional interface',
                'so that loss is at least callable, with a signature accepting',
                'either: my_loss(input: Tensor, target: Tensor, reduction: str)',
                'or my_loss(input: Tensor, reduction: str).'
            ])
            self.loss_oop_interface = False
        self.reduce_spatial_dims = reduce_spatial_dims
        self.transform = transform if transform is not None else nn.Identity()
        self.has_labels = has_labels

    @classmethod
    def default_sampler(cls, xi, xi_labels, epsilon, seed: int):
        del xi, xi_labels, epsilon, seed
        raise WrappingError(
            "No default sampler can be attributed by default by a wrapped loss.")

    @property
    def theta(self):
        if self.loss_oop_interface:
            assert isinstance(self.loss, nn.Module)
            return pt.concat(list(
                map(
                    pt.flatten,
                    chain(self.loss.parameters(), self.transform.parameters())
                )))
        else:
            assert callable(self.loss)
            return self.transform.parameters

    @property
    def intercept(self):
        return pt.tensor(0.)

    def _flat_value_w_labels(self, xi, xi_labels):
        if self.loss_oop_interface:
            return self.regularize(self.loss(
                self.transform(xi),
                xi_labels
            ))
        else:
            return self.regularize(self.loss(
                self.transform(xi),
                xi_labels,
                reduction='none'
            ))

    def _flat_value_wo_labels(self, xi):
        if self.loss_oop_interface:
            return self.regularize(self.loss(
                self.transform(xi)
            ))
        else:
            return self.regularize(self.loss(
                self.transform(xi),
                reduction='none'
            ))

    def _reduce_flat_spatial_dims_loss(self, losses: pt.Tensor) -> pt.Tensor:
        if self.reduce_spatial_dims:
            return losses.mean(dim=-1, keepdim=True)
        else:
            return losses.unsqueeze(-1)

    def value(self, xi: pt.Tensor, xi_labels: Optional[pt.Tensor] = None):
        if self.has_labels:
            assert xi_labels is not None
            if xi.dim() > 2 and xi_labels.dim() > 2:
                *b, _ = xi.size()
                flat_loss = self._flat_value_w_labels(
                    xi.flatten(start_dim=0, end_dim=-2),
                    xi_labels.flatten(start_dim=0, end_dim=-2)
                )
                return self._reduce_flat_spatial_dims_loss(flat_loss).view(*b, 1)
            elif xi.dim() > 2 and xi_labels.dim() == 2:
                *b, _ = xi.size()
                flat_loss = self._flat_value_w_labels(
                    xi.flatten(start_dim=0, end_dim=-2),
                    xi_labels
                )
                return self._reduce_flat_spatial_dims_loss(flat_loss).view(*b, 1)
            elif xi.dim() == 2 and xi_labels.dim() <= 2:
                flat_loss = self._flat_value_w_labels(
                    xi, xi_labels
                ).squeeze()
                return self._reduce_flat_spatial_dims_loss(flat_loss)
            elif xi.dim() == xi_labels.dim() == 1:
                flat_loss = self._flat_value_w_labels(xi, xi_labels)
                return self._reduce_flat_spatial_dims_loss(flat_loss)
            else:
                raise NotImplementedError()
        else:
            assert xi_labels is None
            if xi.dim() > 2:
                *b, _ = xi.size()
                flat_loss = self._flat_value_wo_labels(
                    xi.flatten(start_dim=0, end_dim=-2)
                )
                return self._reduce_flat_spatial_dims_loss(flat_loss).view(*b, 1)
            elif xi.dim() == 2:
                return self._reduce_flat_spatial_dims_loss(
                    self._flat_value_wo_labels(xi)
                ).squeeze()
            elif xi.dim() == 1:
                return self._reduce_flat_spatial_dims_loss(
                    self._flat_value_wo_labels(xi)
                )
            else:
                raise NotImplementedError()
