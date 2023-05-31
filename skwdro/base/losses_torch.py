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
        raise NotImplementedError("Please Implement this method")

    def value(self, xi: pt.Tensor, xi_labels: pt.Tensor):
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
        return self.value(*args)

    @abstractproperty
    def theta(self):
        raise NotImplementedError("Please Implement this property")

    @abstractproperty
    def intercept(self):
        raise NotImplementedError("Please Implement this property")

class NewsVendorLoss_torch(Loss):
    def __init__(
            self,
            sampler: Optional[NoLabelsSampler]=None,
            *,
            k=5, u=7,
            name="NewsVendor loss"):
        super(NewsVendorLoss_torch, self).__init__(sampler)
        self.k = nn.Parameter(pt.tensor(float(k)), requires_grad=False)
        self.u = nn.Parameter(pt.tensor(float(u)), requires_grad=False)
        self.name = name
        self._theta = nn.Parameter(pt.rand(1))

    def value_old(self,theta,xi):
        return self.k*theta-self.u*pt.minimum(theta,xi)

    def value(self, xi: pt.Tensor, xi_labels: NoneType=None):
        return self.k*self.theta - self.u*pt.minimum(self.theta, xi).squeeze(dim=-1)

    @classmethod
    def default_sampler(cls, xi: pt.Tensor, xi_labels: NoneType, epsilon):
        return NewsVendorNormalSampler(xi, sigma=epsilon)

    @property
    def theta(self) -> pt.Tensor:
        return self._theta

    @property
    def intercept(self) -> NoneType:
        return None

    @classmethod
    def default_sampler(cls, xi, xi_labels, epsilon):
        return NewsVendorNormalSampler(xi, sigma=epsilon)

class WeberLoss_torch(Loss):

    def __init__(
            self,
            sampler: Optional[LabeledSampler]=None,
            *,
            name="Weber loss"):
        super(WeberLoss_torch, self).__init__(sampler)
        self.w = nn.Parameter(pt.rand(1))
        self.name = name

    def value_old(self,y,x,w):
        return w*pt.linalg.norm(x-y)

    def value(self, xi: pt.Tensor, xi_labels: pt.Tensor):
        return self.w * pt.linalg.norm(xi - xi_labels, dim=-1)

    @classmethod
    def default_sampler(cls, xi, xi_labels, epsilon):
        return ClassificationNormalNormalSampler(xi, xi_labels, sigma=epsilon, l_sigma=epsilon)

    @property
    def theta(self) -> pt.Tensor:
        return self.w

    @property
    def intercept(self) -> NoneType:
        return None

class LogisticLoss(Loss):
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
        self.L = nn.BCEWithLogitsLoss(reduction='none')

    def logprobs(self, X):
        coefs = self.linear(X)
        return self.classif(coefs), coefs

    def value(self, xi: pt.Tensor, xi_labels: pt.Tensor):
        _, coefs = self.logprobs(xi)
        return self.L(
                coefs,
                (xi_labels + 1.) * .5)

    @classmethod
    def default_sampler(cls, xi, xi_labels, epsilon):
        return ClassificationNormalNormalSampler(xi, xi_labels, sigma=epsilon, l_sigma=epsilon)

    @property
    def theta(self) -> pt.Tensor:
        return self.linear.weight

    @property
    def intercept(self) -> pt.Tensor:
        return self.linear.bias

class PortfolioLoss_torch(Loss):

    def __init__(self, eta, alpha, name="Portfolio loss"):
        self.eta = eta
        self.alpha = alpha
        self.name = name

    def value(self, theta, X):
        #Conversion np.array to torch.tensor if necessary
        if isinstance(theta, (np.ndarray,np.generic)):
            theta = torch.from_numpy(theta)
        if isinstance(X, (np.ndarray,np.generic)):
            X = torch.from_numpy(X)

        N = X.size()[0]

        #We add a double cast in the dot product to solve torch type issues for torch.dot
        in_sample_products = torch.tensor([torch.dot(theta, X[i].double()) for i in range(N)]) 
        expected_value = -(1/N) * torch.sum(in_sample_products)
        reducer = SuperquantileReducer(superquantile_tail_fraction=self.alpha)
        reduce_loss = reducer(in_sample_products)

        return expected_value + self.eta*reduce_loss
    @classmethod
    def default_sampler(cls, xi, xi_labels, epsilon) -> BaseSampler:
        raise NotImplementedError("Please Implement this method")
