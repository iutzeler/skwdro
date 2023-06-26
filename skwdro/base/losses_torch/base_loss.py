from typing import Optional
from abc import abstractclassmethod, abstractproperty

import torch as pt
import torch.nn as nn

from skwdro.base.samplers.torch.base_samplers import BaseSampler

class Loss(nn.Module):
    """ Base class for loss functions """
    def __init__(self, sampler: Optional[BaseSampler]=None):
        super(Loss, self).__init__()
        self._sampler = sampler

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
        self._sampler = None

    @abstractclassmethod
    def default_sampler(cls, xi, xi_labels, epsilon) -> BaseSampler:
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
