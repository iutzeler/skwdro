from typing import Optional

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
            k: float = 5,
            u: float = 7,
            l2reg: Optional[float] = None,
            name: Optional[str] = "NewsVendor loss"):
        super(NewsVendorLoss_torch, self).__init__(sampler, l2reg=l2reg)
        self.k = pt.tensor(float(k))
        self.u = pt.tensor(float(u))
        self.name = name
        self._theta = nn.Parameter(pt.rand(1))

    def value_old(self, theta, xi):
        return self.k * theta - self.u * pt.minimum(theta, xi)

    def value(
        self, xi: pt.Tensor, xi_labels: Optional[pt.Tensor] = None
    ) -> pt.Tensor:
        """ Forward pass of the loss on the data

        Parameters
        ----------
        xi : pt.Tensor
            empirical observations of demand
        xi_labels : NoneType
            placeholder, do not touch
        """
        assert xi_labels is not None
        returns = self.k * self.theta
        costs = self.u * pt.minimum(self.theta, xi).mean(dim=-1, keepdim=True)
        return returns - costs

    @property
    def theta(self) -> pt.Tensor:
        return self._theta

    @property
    def intercept(self) -> None:
        return None

    @classmethod
    def default_sampler(cls, xi, xi_labels, epsilon, seed: int):
        del xi_labels
        return NewsVendorNormalSampler(xi, seed, sigma=epsilon)
