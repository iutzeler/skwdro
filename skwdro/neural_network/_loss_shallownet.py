from typing import Optional

import torch as pt
import torch.nn as nn

from skwdro.base.samplers.torch.base_samplers import LabeledSampler
from skwdro.base.samplers.torch.classif_sampler import ClassificationNormalNormalSampler

from skwdro.base.losses_torch import Loss


class ShallowNetLoss(Loss):
    def __init__(
            self,
            sampler: Optional[LabeledSampler] = None,
            *,
            d: int = 0,
            n_neurons: int = 0,
            ly1=None,
            ly2=None,
            fit_intercept: bool = False) -> None:
        assert sampler is not None
        super(ShallowNetLoss, self).__init__(sampler)
        assert n_neurons is not None and n_neurons > 0, "Please provide a valid layer height n_neurons>0"
        assert d > 0, "Please provide a valid data dimension d>0"
        if ly1 is not None:
            assert len(ly1) == n_neurons  # would be weird
        self.L = nn.MSELoss(reduction='none')

        self.linear1 = nn.Linear(
            d, n_neurons, bias=fit_intercept)  # d -> n_neurons
        self.linear2 = nn.Linear(n_neurons, 1, bias=False)  # n_neurons -> 1

        dtype, device = pt.float32, "cpu"  # maybe put in parameters, todo?
        if ly1 is not None and ly2 is not None:
            self.linear1.weight.data = pt.tensor(
                ly1[:, :-1], dtype=dtype, device=device, requires_grad=True)
            self.linear1.bias.data = pt.tensor(
                ly1[:, -1:].flatten(), dtype=dtype, device=device, requires_grad=True)
            self.linear2.weight.data = pt.tensor(
                ly2, dtype=dtype, device=device, requires_grad=True)

    def pred(self, X):
        li = pt.relu(self.linear1(X))
        return self.linear2(li)

    def value(self, xi: pt.Tensor, xi_labels: Optional[pt.Tensor]):
        assert xi_labels is not None
        xi_labels_pred = self.pred(xi)

        return self.L(
            xi_labels_pred,
            xi_labels
        )

    @classmethod
    def default_sampler(cls, xi, xi_labels, epsilon, seed=0):
        return ClassificationNormalNormalSampler(xi, xi_labels, sigma=epsilon, l_sigma=epsilon, seed=seed)

    @property
    def theta(self) -> pt.Tensor:
        return pt.concatenate((self.linear1.weight.flatten(), self.linear2.weight.flatten()))

    @property
    def intercept(self) -> pt.Tensor:
        return self.linear1.bias
        # return pt.concatenate((self.linear1.bias, self.linear2.bias))

    @property
    def parameters_iter(self):
        return self.state_dict()
