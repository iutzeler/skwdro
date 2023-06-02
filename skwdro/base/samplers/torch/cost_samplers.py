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
            epsilon
            ):
        super(NoLabelsCostSampler, self).__init__(cost._sampler_data(xi, epsilon))
        self.generating_cost = cost

class LabeledCostSampler(LabeledSampler):
    def __init__(
            self,
            cost: Cost,
            xi: pt.Tensor,
            xi_labels: pt.Tensor,
            epsilon
            ):
        super(LabeledCostSampler, self).__init__(cost._sampler_data(xi, epsilon), cost._sampler_labels(xi_labels, epsilon))
        self.generating_cost = cost
