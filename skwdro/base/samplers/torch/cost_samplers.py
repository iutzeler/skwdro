from typing import Optional
import torch as pt
from abc import ABC, abstractmethod

from skwdro.base.samplers.torch.base_samplers import LabeledSampler, NoLabelsSampler
from skwdro.base.costs_torch import Cost

class NoLabelsCostSampler(NoLabelsSampler):
    def __init__(
            self,
            cost: Cost,
            xi: pt.Tensor,
            epsilon,
            seed: int,
            ):
        super(NoLabelsCostSampler, self).__init__(cost._sampler_data(xi, epsilon), seed)
        self.generating_cost = cost
        self.epsilon = epsilon

    def reset_mean(self, xi, xi_labels):
        self.__init__(self.generating_cost, xi, self.epsilon, self.seed)

class LabeledCostSampler(LabeledSampler):
    def __init__(
            self,
            cost: Cost,
            xi: pt.Tensor,
            xi_labels: pt.Tensor,
            epsilon,
            seed: int
            ):
        super(LabeledCostSampler, self).__init__(cost._sampler_data(xi, epsilon), cost._sampler_labels(xi_labels, epsilon), seed)
        self.generating_cost = cost
        self.epsilon = epsilon

    def reset_mean(self, xi, xi_labels):
        self.__init__(self.generating_cost, xi, xi_labels, self.epsilon, self.seed)
