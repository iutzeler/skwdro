from typing import Optional
from abc import abstractmethod, abstractproperty, ABC

import torch as pt
import torch.nn as nn

from skwdro.base.samplers.torch.base_samplers import BaseSampler

class Loss(nn.Module, ABC):
    """ Base class for loss functions """
    _sampler: BaseSampler
    def __init__(
            self,
            sampler: BaseSampler,
            *,
            l2reg: Optional[float]=None
            ):
        super(Loss, self).__init__()
        self._sampler = sampler
        self.l2reg = None if l2reg is None or l2reg <= 0. else pt.tensor(l2reg)

    def regularize(self, loss: pt.Tensor):
        r"""
        Returns the regularized loss, used in the value function.
        Adds a new term :math:`\frac{1}{2}\|\theta\|_2^2
        """
        if self.l2reg is None:
            return loss
        else:
            return loss + .5 * self.l2reg * (self.theta*self.theta).sum()

    def value_old(self,theta,xi):
        """
        DEPRECATED, DO NOT USE
        """
        raise NotImplementedError("Please Implement this method")

    def value(self, xi: pt.Tensor, xi_labels: Optional[pt.Tensor]):
        """
        Perform forward pass.
        """
        raise NotImplementedError("Please Implement this method")

    def sample_pi0(self, n_samples: int):
        return self.sampler.sample(n_samples)

    @property
    def sampler(self) -> BaseSampler:
        if self._sampler is None:
            raise ValueError("The sampler was not initialized properly")
        else:
            return self._sampler

    @sampler.setter
    def sampler(self, sampler: BaseSampler):
        self._sampler = sampler

    @sampler.deleter
    def sampler(self):
        del self._sampler

    @classmethod
    @abstractmethod
    def default_sampler(cls, xi, xi_labels, epsilon, seed: int) -> BaseSampler:
        raise NotImplementedError("Please Implement this method")

    def forward(self, *args):
        """
        Alias for :method:`~Loss.value`, for consistency with usual torch api.
        """
        return self.value(*args)

    @abstractproperty
    def theta(self):
        raise NotImplementedError("Please Implement this property")

    @abstractproperty
    def intercept(self):
        raise NotImplementedError("Please Implement this property")
