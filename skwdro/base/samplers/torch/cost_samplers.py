from typing import Optional
import torch as pt

from skwdro.base.samplers.torch.base_samplers import LabeledSampler, NoLabelsSampler
from skwdro.base.costs_torch import Cost


class NoLabelsCostSampler(NoLabelsSampler):
    def __init__(
        self,
        cost: Cost,
        xi: pt.Tensor,
        sigma,
        seed: Optional[int] = None,
    ):
        """
        Parent class of all samplers that only sample inputs, with a
        specification drawn from a cost functional
        (:py:class:`skwdro.base.costs_torch.Cost`).

        Parameters
        ----------
        cost: Cost
            cost functional specifying the samp;ling behaviour through its
            :py:method:`skwdro.base.costs_torch.Cost.sampler` method.
        xi: pt.Tensor
            mean for inputs
        sigma: float|Tensor
            scalar standard deviation shared through dimensions, for inputs.

        See :py:class:`skwdro.base.samplers.torch.base_samplers.IsOptionalCovarianceSampler`
        for other arguments.
        """
        super(NoLabelsCostSampler, self).__init__(
            cost._sampler_data(xi, sigma), seed
        )
        self.generating_cost = cost
        self.sigma = sigma

    def reset_mean(self, xi, xi_labels):
        del xi_labels
        self.__init__(self.generating_cost, xi, self.sigma, self.seed)


class LabeledCostSampler(LabeledSampler):
    def __init__(
        self,
        cost: Cost,
        xi: pt.Tensor,
        xi_labels: pt.Tensor,
        sigma,
        seed: Optional[int] = None
    ):
        """
        Parent class of all samplers that sample both inputs and labels, with a
        specification drawn from a cost functional
        (:py:class:`skwdro.base.costs_torch.Cost`).

        Parameters
        ----------
        cost: Cost
            cost functional specifying the samp;ling behaviour through its
            :py:method:`skwdro.base.costs_torch.Cost.sampler` method.
        xi: pt.Tensor
            mean for inputs
        xi_labels: pt.Tensor
            mean for targets
        sigma: float|Tensor
            scalar standard deviation shared through dimensions, for inputs.

        See :py:class:`skwdro.base.samplers.torch.base_samplers.IsOptionalCovarianceSampler`
        for other arguments.
        """
        sd, sl = (
            cost._sampler_data(xi, sigma),
            cost._sampler_labels(xi_labels, sigma)
        )
        if sl is None:
            raise ValueError("Please choose a cost that can sample labels")
        super(LabeledCostSampler, self).__init__(sd, sl, seed)
        self.generating_cost = cost
        self.sigma = sigma

    def reset_mean(self, xi, xi_labels):
        self.__init__(
            self.generating_cost, xi,
            xi_labels, self.sigma, self.seed
        )
