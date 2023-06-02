from abc import abstractclassmethod, abstractmethod
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

    def value(self,theta,xi):
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

class NewsVendorLoss_torch(Loss):

    def __init__(
            self,
            sampler: Optional[NoLabelsSampler]=None,
            *,
            k=5, u=7,
            name="NewsVendor loss"):
        super(NewsVendorLoss_torch, self).__init__(sampler)
        self.k = k
        self.u = u
        self.name = name

    def value(self,theta,xi):
        return self.k*theta-self.u*pt.minimum(theta,xi)

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
        self.name = name

    def value(self,y,x,w):
        return w*pt.linalg.norm(x-y)

    @classmethod
    def default_sampler(cls, xi, xi_labels, epsilon):
        return ClassificationNormalNormalSampler(xi, xi_labels, sigma=epsilon, l_sigma=epsilon)

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
        self.L = nn.BCEWithLogitsLoss()

    def forward(self, X):
        coefs = self.linear(X)
        return self.classif(coefs), coefs

    def value(self, X, y):
        _, coefs = self.__call__(X)
        return self.L(
                coefs,
                (y == 1).long(),
                reduction='none')

    @classmethod
    def default_sampler(cls, xi, xi_labels, epsilon):
        return ClassificationNormalNormalSampler(xi, xi_labels, sigma=epsilon)
