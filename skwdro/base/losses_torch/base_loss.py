from typing import Optional, Tuple, overload
from abc import abstractmethod, ABC

import torch as pt
import torch.nn as nn

from skwdro.base.samplers.torch.base_samplers import BaseSampler


class Loss(nn.Module, ABC):
    """ Base class for loss functions """
    _sampler: Optional[BaseSampler]
    has_labels: bool

    def __init__(
        self,
        sampler: Optional[BaseSampler],
        has_labels: bool,
        *,
        l2reg: Optional[float] = None,
        xi: Optional[pt.Tensor] = None,
        xi_labels: Optional[pt.Tensor] = None,
        sigma: Optional[float] = None
    ) -> None:
        r"""Base class for loss functions.

        This class provides a base implementation for various types of loss
        functions. It includes attributes to handle sampler initialization and L2
        regularization parameters.

        sampler: BaseSampler|None
            An optional BaseSampler instance used by the loss function. If not
            provided, default sampler parameters are used based on other inputs
            if available. (default is None)
        has_labels: bool
            set to ``True`` if the loss accepts two inputs: a prediction and some
            kind of target. Otherwise, set to ``False``.

            .. warning:: It is *your* job to check that the :py:attr:`loss`,
                :py:attr:`_sampler`, and :py:attr:`has_labels` parameters are
                compatible with one another.

        l2reg: float|None
            An optional float for L2 regularization parameter. It will be
            converted to a tensor if provided and positive; otherwise, it remains
            `None`. (default is None)
        xi: Tensor|None
            An optional tensor representing the xi value for samplers
            initialization if sampler is set to ``None``. If not provided,
            defaults to `None`. (default is None)
        xi_labels:  Tensor|None
            An optional tensor representing labels associated with the xi value.
            It is not used to determine if the default solver can be set, only
            :math:`\xi` is. If not provided, defaults to `None` (default is None)
        sigma: An optional float for sigma parameter used in sampler
            initialization if no specific sigma is given. If not provided,
            defaults to 0.1. (default is None)
        """
        super(Loss, self).__init__()
        # Try to initialise the sampler in best-effort mode
        if sampler is None and xi is not None:
            self._sampler = self.default_sampler(
                xi, xi_labels,
                0.1 if sigma is None else sigma,
                None
            )
        else:
            self._sampler = sampler
        self.l2reg: Optional[pt.Tensor] = (
            None if l2reg is None or l2reg <= 0.
            else pt.tensor(l2reg)
        )
        self.has_labels = has_labels

    def regularize(self, loss: pt.Tensor) -> pt.Tensor:
        r"""
        Returns the regularized loss, used in the value function.
        Adds a new term :math:`\frac{1}{2}\|\theta\|_2^2`
        """
        if self.l2reg is None:
            return loss
        else:
            reg: pt.Tensor = .5 * self.l2reg * (self.theta * self.theta).sum()
            return loss + reg

    def value_old(self, theta, xi):
        """
        DEPRECATED, DO NOT USE
        """
        del theta, xi
        raise NotImplementedError("Please Implement this method")

    @overload
    def value(self, xi: pt.Tensor, xi_labels: pt.Tensor) -> pt.Tensor:
        pass

    @overload
    def value(self, xi: pt.Tensor, xi_labels: None) -> pt.Tensor:
        pass

    def value(
        self,
        xi: pt.Tensor,
        xi_labels: Optional[pt.Tensor]
    ) -> pt.Tensor:
        """
        Perform forward pass.
        Overload the method to implement your own.
        """
        del xi, xi_labels
        raise NotImplementedError("Please Implement this method")

    def sample_pi0(
        self, n_samples: int
    ) -> Tuple[pt.Tensor, Optional[pt.Tensor]]:
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

    @classmethod
    @abstractmethod
    def default_sampler(
        cls,
        xi,
        xi_labels,
        epsilon,
        seed: Optional[int]
    ) -> Optional[BaseSampler]:
        r"""
        Overload this method if you wish to define a default sampler associated
        to your loss.
        """
        return None

    def forward(self, *args):
        """
        Alias for :py:meth:`value`, for consistency with usual torch api.
        """
        return self.value(*args)

    @property
    @abstractmethod
    def theta(self):
        raise NotImplementedError("Please Implement this property")

    @property
    @abstractmethod
    def intercept(self):
        raise NotImplementedError("Please Implement this property")
