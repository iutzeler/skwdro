from typing import Optional

import torch as pt
import torch.nn as nn

from .base_loss import Loss
from skwdro.base.samplers.torch.base_samplers import LabeledSampler
from skwdro.base.samplers.torch.classif_sampler import ClassificationNormalNormalSampler

class BiDiffSoftMarginLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(BiDiffSoftMarginLoss, self).__init__()

    def forward(self, input, target):
        target = pt.reshape(target, input.shape)
        return - nn.functional.logsigmoid(target * input)

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
            sampler: LabeledSampler,
            *,
            d: int=0,
            fit_intercept: bool=False) -> None:
        super(LogisticLoss, self).__init__(sampler)
        assert d > 0, "Please provide a valid data dimension d>0"
        self.linear = nn.Linear(d, 1, bias=fit_intercept)
        self.classif = nn.Tanh()
        self.L = BiDiffSoftMarginLoss(reduction='none')

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
    def default_sampler(cls, xi, xi_labels, epsilon, seed: int):
        return ClassificationNormalNormalSampler(xi, xi_labels, seed, sigma=epsilon, l_sigma=epsilon)

    @property
    def theta(self) -> pt.Tensor:
        return self.linear.weight

    @property
    def intercept(self) -> pt.Tensor:
        return self.linear.bias
