from abc import ABC, abstractclassmethod, abstractmethod, abstractproperty
from types import NoneType
from typing import Optional

import torch as pt
import torch.nn as nn

from skwdro.base.samplers.torch.base_samplers import LabeledSampler, BaseSampler, NoLabelsSampler
from skwdro.base.samplers.torch.newsvendor_sampler import NewsVendorNormalSampler
from skwdro.base.samplers.torch.classif_sampler import ClassificationNormalNormalSampler

from skwdro.base.losses_torch import Loss

class ShallowNetLoss(Loss):
    def __init__(
            self,
            sampler: Optional[LabeledSampler]=None,
            *,
            d: int=0,
            nbneurone: int=0,
            fit_intercept: bool=False) -> None:
        super(ShallowNetLoss, self).__init__(sampler)
        assert nbneurone is not None and nbneurone > 0, "Please provide a valid layer height nbneurone>0"
        assert d > 0, "Please provide a valid data dimension d>0"
        self.L = nn.MSELoss(reduction='none')

        #self.linear1 = nn.Linear(d, nbneurone, bias=fit_intercept) # nbneurone -> 1
        #self.linear2 = nn.Linear(nbneurone, 1, bias=fit_intercept) # nbneurone -> 1

        self.linear1 = nn.Linear(d, 1, bias=fit_intercept) # debug=linearreg

    def pred(self, X):
        li = pt.relu(self.linear1(X))
        return self.linear2(li)

    def value(self, xi: pt.Tensor, xi_labels: pt.Tensor):
        #xi_labels_pred = self.pred(xi)

        xi_labels_pred = self.linear1(xi) # debug=linearreg

        return self.L(
                xi_labels_pred,
                xi_labels)

    @classmethod
    def default_sampler(cls, xi, xi_labels, epsilon):
        return ClassificationNormalNormalSampler(xi, xi_labels, sigma=epsilon, l_sigma=epsilon)

    @property
    def theta(self) -> pt.Tensor:
        #return [self.linear1.weight, self.linear2.weight]
        return self.linear1.weight

    @property
    def intercept(self) -> pt.Tensor:
        return self.linear1.bias
