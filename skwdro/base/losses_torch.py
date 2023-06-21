from abc import ABC, abstractclassmethod, abstractmethod, abstractproperty
from types import NoneType
from typing import Optional

import torch as pt
import torch.nn as nn

from skwdro.base.samplers.torch.base_samplers import LabeledSampler, BaseSampler, NoLabelsSampler
from skwdro.base.samplers.torch.newsvendor_sampler import NewsVendorNormalSampler
from skwdro.base.samplers.torch.classif_sampler import ClassificationNormalNormalSampler

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
            sampler: Optional[NoLabelsSampler]=None,
            *,
            k: int=5,
            u: int=7,
            name: Optional[str]="NewsVendor loss"):
        super(NewsVendorLoss_torch, self).__init__(sampler)
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
    def default_sampler(cls, xi, xi_labels, epsilon):
        return NewsVendorNormalSampler(xi, sigma=epsilon)

class LogisticLoss(Loss):
    r""" Logisic regression with classes :math:`\{-1, 1\}`

    Parameters
    ----------
    sampler : Optional[LabeledSampler]
        sampler for the adversarial samples
    d : int
        dimension of the data (``xi.size(-1)``)
    fit_intercept : bool
        model has an affine dimension

    Attributes
    ----------
    L : nn.SoftMarginLoss
        torch solution to implement the soft margin in ``]-1, 1[``
    classif : nn.Tanh
        activation function to project tensor in ``]-1, 1[``
    linear : nn.Linear
        linear combination containing the relevant parameters
    """
    def __init__(
            self,
            sampler: Optional[LabeledSampler]=None,
            *,
            d: int=0,
            fit_intercept: bool=False) -> None:
        super(LogisticLoss, self).__init__(sampler)
        assert d > 0, "Please provide a valid data dimension d>0"
        self.linear = nn.Linear(d, 1, bias=fit_intercept)
        self.classif = nn.Tanh()
        self.L = nn.SoftMarginLoss(reduction='none')

    def predict(self, X: pt.Tensor) -> pt.Tensor:
        """ Predict the label of the argument tensor

        Parameters
        ----------
        self :
            self
        X : pt.Tensor
            X

        Returns
        -------
        pt.Tensor

        """
        coefs = self.linear(X)
        return self.classif(coefs)

    def value(self, xi: pt.Tensor, xi_labels: pt.Tensor):
        """ Forward pass of the loss

        Parameters
        ----------
        xi : pt.Tensor
            data
        xi_labels : pt.Tensor
            labels
        """
        coefs = self.linear(xi)
        return self.L(coefs, xi_labels)

    @classmethod
    def default_sampler(cls, xi, xi_labels, epsilon):
        return ClassificationNormalNormalSampler(xi, xi_labels, sigma=epsilon, l_sigma=epsilon)

    @property
    def theta(self) -> pt.Tensor:
        return self.linear.weight

    @property
    def intercept(self) -> pt.Tensor:
        return self.linear.bias


class QuadraticLoss(Loss):
    def __init__(
            self,
            sampler: Optional[LabeledSampler]=None,
            *,
            d: int=0,
            fit_intercept: bool=False) -> None:
        super(QuadraticLoss, self).__init__(sampler)
        assert d > 0, "Please provide a valid data dimension d>0"
        self.linear = nn.Linear(d, 1, bias=fit_intercept)
        self.L = nn.MSELoss(reduction='none')

    def regression(self, X):
        coefs = self.linear(X)
        return coefs

    def value(self, xi: pt.Tensor, xi_labels: pt.Tensor):
        coefs = self.regression(xi)
        return self.L(
                coefs,
                xi_labels)

    @classmethod
    def default_sampler(cls, xi, xi_labels, epsilon):
        return ClassificationNormalNormalSampler(xi, xi_labels, sigma=epsilon, l_sigma=epsilon)

    @property
    def theta(self) -> pt.Tensor:
        return self.linear.weight

    @property
    def intercept(self) -> pt.Tensor:
        return self.linear.bias
