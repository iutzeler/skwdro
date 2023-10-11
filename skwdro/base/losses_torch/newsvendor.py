from typing import Optional
from types import NoneType

import torch as pt
import torch.nn as nn

from .base_loss import Loss
from skwdro.base.samplers.torch.base_samplers import NoLabelsSampler
from skwdro.base.samplers.torch.newsvendor_sampler import NewsVendorNormalSampler

class NewsVendorLoss_torch(Loss):
    r""" Loss associated with the newsvendor problem:
    .. math::
        k\theta - \mathbb{E}[u\min(\theta, \xi)]

    Parameters
    ----------
    sampler : Optional[NoLabelsSampler]
        optional sampler to use for the demand
    k : int
        journal price
    u : int
        benefit per journal sold
    name :
        name of the loss, optional

    Attributes
    ----------
    sampler : NoLabelsSampler
    k : nn.Parameter
    u : nn.Parameter
    theta : nn.Parameter
        number of journal stocked
    name : Optional[str]
    """
    def __init__(
            self,
            sampler: NoLabelsSampler,
            *,
            k: float=5,
            u: float=7,
            l2reg: Optional[float]=None,
            name: Optional[str]="NewsVendor loss"):
        super(NewsVendorLoss_torch, self).__init__(sampler, l2reg=l2reg)
        self.k = nn.Parameter(pt.tensor(float(k)), requires_grad=False)
        self.u = nn.Parameter(pt.tensor(float(u)), requires_grad=False)
        self.name = name
        self._theta = nn.Parameter(pt.rand(1))

    def value_old(self,theta,xi):
        return self.k*theta-self.u*pt.minimum(theta,xi)

    def value(self, xi: pt.Tensor, xi_labels: NoneType=None):
        """ Forward pass of the loss on the data

        Parameters
        ----------
        xi : pt.Tensor
            empirical observations of demand
        xi_labels : NoneType
            placeholder, do not touch
        """
        return self.k*self.theta - self.u*pt.minimum(self.theta, xi)

    @property
    def theta(self) -> pt.Tensor:
        return self._theta

    @property
    def intercept(self) -> NoneType:
        return None

    @classmethod
    def default_sampler(cls, xi, xi_labels, epsilon, seed: int):
        return NewsVendorNormalSampler(xi, seed, sigma=epsilon)
