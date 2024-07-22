import torch as pt

from skwdro.base.samplers.torch.base_samplers import LabeledSampler, NoLabelsSampler
from skwdro.base.costs_torch import Cost


class NoLabelsCostSampler(NoLabelsSampler):
    def __init__(
        self,
        cost: Cost,
        xi: pt.Tensor,
        sigma,
        seed: int,
    ):
        super(NoLabelsCostSampler, self).__init__(
            cost._sampler_data(xi, sigma), seed
        )
        self.generating_cost = cost
        self.sigma = sigma

    def reset_mean(self, xi, xi_labels):
        del xi_labels  # https://pycodequ.al/docs/pylint-messages/w0613-unused-argument.html#how-to-fix
        self.__init__(self.generating_cost, xi, self.sigma, self.seed)


class LabeledCostSampler(LabeledSampler):
    def __init__(
        self,
        cost: Cost,
        xi: pt.Tensor,
        xi_labels: pt.Tensor,
        sigma,
        seed: int
    ):
        sd, sl = cost._sampler_data(
            xi, sigma), cost._sampler_labels(xi_labels, sigma)
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
