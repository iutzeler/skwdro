from abc import ABC, abstractmethod
from typing import Optional, Tuple, overload
from itertools import chain

import torch as pt
import torch.nn as nn

from skwdro.base.costs_torch import Cost
from skwdro.base.losses_torch import Loss
from skwdro.base.samplers.torch.base_samplers import BaseSampler
from skwdro.base.samplers.torch import NoLabelsCostSampler, LabeledCostSampler
from skwdro.solvers.utils import Steps


class _DualLossBase(nn.Module, ABC):
    r""" Base class to register parameters for a dual loss:
    .. math::
        \begin{align}
            \rho\lambda\\
            + \epsilon\mathbb{E}_{\xi}
                \ln{\left(\mathbb{E}_{\zeta\sim\pi_0}
                    e^{\frac{L_\theta(\zeta)-\lambda c(\zeta,\xi)}{\epsilon}}
                \right)}
        \end{align}

    Parameters
    ----------
    loss : Loss
        regular loss :math:`L_\theta`, with a forward pass
        available with ``__call__``
    cost : Cost
        transport (ground) cost :math:`c(\xi, \zeta)`
    n_samples : int
        number of :math:`zeta` samples to draw from :math:`\pi_0`
    rho_0 : pt.Tensor
        first guess for a good maximal distance between xi distribution
        and adversarial distribution
    epsilon_0 : Optional[pt.tensor], default ``None``
        first guess for a good regularization value :math:`\epsilon`
        for Sinkhorn
    n_iter : int
        number of gradient descent updates
    reduction: str | None
         specifies the reduction to apply to the outer expectation of the
         SkWDRO formula applied: ``'none'`` | ``'mean'`` | ``'sum'``.
         - ``'none'``: no reduction will be applied,
         - ``'mean'``: the sum of the output will be divided by the number of
         elements in the output,
         - ``'sum'``: the output will be summed.
         Default: ``None`` which translates to ``'mean'``
    gradient_hypertuning : bool, default ``False``
        [WIP] set to ``True`` to tune rho and epsilon.
    imp_samp : bool, default ``True``
        kwarg, set to ``True`` to use importance sampling to improve
        the sampling

    Attributes
    ----------
    loss : Loss
    cost : Cost
    epsilon : nn.Parameter
        initialized a ``epsilon_0``, and without requires_grad
    rho : nn.Parameter
        initialized a ``rho_0``, and without requires_grad
    n_samples : int
    n_iter : int
    reduction: str
         specifies the reduction to apply to the outer expectation of the
         SkWDRO formula applied: ``'none'`` | ``'mean'`` | ``'sum'``.
    imp_samp : bool

    Shapes
    ------
    rho_0: (1,)
    epsilon_0: (1,)
    """
    reduction: str

    def __init__(
        self,
        loss: Loss,
        cost: Cost,
        n_samples: int,
        epsilon_0: pt.Tensor,
        rho_0: pt.Tensor,
        n_iter: Steps,
        *,
        reduction: Optional[str] = None,
        gradient_hypertuning: bool = False,
        imp_samp: bool = True,
    ) -> None:
        super(_DualLossBase, self).__init__()
        self.primal_loss = loss
        # TODO: implement __call__ for torch costs
        self.cost = cost

        # epsilon and rho are parameters so that they can be printed if needed.
        # But they are not included in the autograd graph
        # (requires_grad=False).
        self.rho = nn.Parameter(pt.as_tensor(
            rho_0), requires_grad=gradient_hypertuning)
        self.epsilon = nn.Parameter(pt.as_tensor(
            epsilon_0), requires_grad=gradient_hypertuning)

        self.reduction = 'mean' if reduction is None else reduction

        # Lambda is tuned during training, and it requires a proxy in its
        # parameter form.
        # _lam is the tuned variable, and softplus(_lam) is the "proxy"
        # that is accessed via
        # self.lam in the code (see the parameter decorated method).
        self._lam = nn.Parameter(1e-3 / rho_0 if rho_0 > 0. else pt.tensor(0.))
        if rho_0 <= 0.:
            self._lam.requires_grad_(False)

        # Private sampler points to the loss l_theta, even if unset
        # Verification is made lazily at forward pass time
        self._sampler = loss._sampler

        # Number of zeta samples are checked at __init__, but can be
        # dynamically changed
        self.n_samples = n_samples
        self.n_iter = n_iter
        self.imp_samp = imp_samp
        self._opti: Optional[pt.optim.Optimizer] = None
        self.erm_mode: bool = False

    def reduce_loss_batch(self, losses: pt.Tensor) -> pt.Tensor:
        r"""Performs the reduction that the ``torch.nn._C`` API would have done if
        one were to use the OOP/functional API of pytorch instead of SkWDRO.
        If the reduction method was set to ``"mean"``, the batch of losses is
        averaged on all available dimensions. If instead it was set to ``"sum"``
        it is summed over.
        On the other hand if it was set to ``"none"`` (or ``None``), the batch is
        returned **as is**.

        This reduction aims at computing the **outer** expectation of the SkWDRO
        formula (`see more <why_skwdro.html>`__), on the :math:`\xi` samples as
        :math:`\mathbb{E}_{\xi\sim\hat{\mathbb{P}}^N}[L_\theta^\texttt{robust}]`

        .. warning::
            Take care about the way the loss you picked returns batches in the
            good shape if you want the "no reduction" option.

        Parameters
        ----------
        losses: :py:class:`torch.Tensor`
            batch of losses computed for each sample :math:`\xi`

        Returns
        -------
        loss: :py:class:`torch.Tensor`
            squeezed/reduced batch of losses

        Shapes:
        -------
        losses: (m, 1)
        loss:
            - (,) if :py:attr:`reduction` is ``"mean"`` or ``"sum"``,
            - (m,) if :py:attr:`reduction` is ``"none"``.
        """
        assert losses.dim() == 2
        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        elif self.reduction in ('none', None):
            return losses.squeeze(-1)
        else:
            # Unreachable (unless messing up with reduction attribute).
            raise NotImplementedError()

    def init_sampler(
        self,
        xi: pt.Tensor, xi_labels: Optional[pt.Tensor],
        sigma: float, seed: Optional[int]
    ) -> BaseSampler:
        """Initializes a sampler based on the provided parameters and sets it to
        the instance attribute :py:attr:`_sampler`.

        Parameters
        ----------

        xi: pt.Tensor
            A tensor representing the input data points.
        xi_labels: pt.Tensor|None
            An optional tensor representing the labels for the input data points,
            defaults to ``None``.
        sigma: float
            A float representing the scale parameter for the sampler.
        seed: int|None
            An optional integer representing the random seed for reproducibility,
            defaults to None.

        Raises
        ------
        AssertionError:
            If `has_labels` is True and `xi_labels` is None, or if `has_labels`
            is False and `xi_labels` is not None.

        Returns
        -------
        BaseSampler
            An instance of the appropriate subclass of BaseSampler.
        """
        sam: BaseSampler
        assert self.primal_loss.sampler is None
        if self.primal_loss.has_labels:
            assert xi_labels is not None, """
                Do not forward None as labels on a Loss function that has the
                `has_labels` flag set to True.
            """
            sam = LabeledCostSampler(
                self.cost,
                xi, xi_labels,
                sigma, seed=seed
            )
        else:
            assert xi_labels is None, """
                Do not forward labels on a Loss function that has the
                `has_labels` flag set to False.
            """
            sam = NoLabelsCostSampler(
                self.cost,
                xi,
                sigma, seed=seed
            )
        self._sampler = sam
        return sam

    @property
    def iterations(self):
        if isinstance(self.n_iter, int):
            return range(self.n_iter)
        else:
            return range(self.n_iter[1])

    @overload
    @abstractmethod
    def forward(
        self,
        xi: pt.Tensor,
        xi_labels: Optional[pt.Tensor] = None,
        zeta: None = None,
        zeta_labels: None = None,
        reset_sampler: bool = True
    ) -> pt.Tensor:
        ...

    @overload
    @abstractmethod
    def forward(
        self,
        xi: pt.Tensor,
        xi_labels: Optional[pt.Tensor],
        zeta: pt.Tensor,
        zeta_labels: Optional[pt.Tensor] = None,
        reset_sampler: bool = True
    ) -> pt.Tensor:
        ...

    @abstractmethod
    def forward(
        self,
        xi: pt.Tensor,
        xi_labels: Optional[pt.Tensor] = None,
        zeta: Optional[pt.Tensor] = None,
        zeta_labels: Optional[pt.Tensor] = None,
        reset_sampler: bool = True
    ) -> Optional[pt.Tensor]:
        raise NotImplementedError()

    def freeze(self, rg: bool = False, include_hyper: bool = False):
        """ Freeze all the primal losse's parameters for some
        gradients operations.

        Parameters
        ----------
        rg : bool
            Set to ``True`` to unfreeze, and leave to ``False``
            to freeze.
        include_hyper: bool
            Set to ``True`` to freeze the rho and epsilon params
            as well.
        """
        frozen_params = self.parameters() if include_hyper else chain(
            self.primal_loss.parameters(), (self._lam,))
        for param in frozen_params:
            param.requires_grad = rg

    def eval(self):
        self.erm_mode = True
        return super().eval()

    def train(self, mode: bool = False):
        self.erm_mode = mode
        return super().train(mode)


class _OptimizeableDual(_DualLossBase):
    @property
    def optimizer(self) -> pt.optim.Optimizer:
        """ optimizer set up by the user

        Returns
        -------
        optimizer : pt.optim.Optimizer

        """
        if self._opti is None:
            raise AttributeError(' '.join([
                "Optimizer for:\n" + self.__str__(),
                "\nis ill defined (None), please set it beforehand."
            ]))
        return self._opti

    @optimizer.setter
    def optimizer(self, o: pt.optim.Optimizer):
        """ setter for the optimizer

        Parameters
        ----------
        o : pt.optim.Optimizer
            the target new optimizer instance
        """
        self._opti = o


class _SampledDualLoss(_OptimizeableDual):
    def generate_zetas(
        self,
        n_samples: Optional[int] = None
    ) -> Tuple[pt.Tensor, Optional[pt.Tensor]]:
        """ Generate zeta samples from the loss-sampler.
        Either generates the default number of samples, set at onstruction of the
        object, or the set number of samples specified as a parameter.

        Parameters
        ----------
        n_samples : Optional[int]
            number of samples to sample

        Returns
        -------
        zetas : pt.Tensor
        zeta_labels : Optional[pt.Tensor]

        Shapes
        ------
        zeta : (n_samples, m, d)
        zeta_labels : (n_samples, m, d')
        """
        if self.primal_loss.sampler is None:
            raise ValueError(" ".join([
                'Please set a sampler on your dual loss for the',
                'reference distribution of the regularization of the'
                'Wasserstein neighborhood before your forward pass.'
            ]))
        elif n_samples is None or n_samples <= 0:
            # Default:
            return self.primal_loss.sampler.sample(self.n_samples)
        else:
            return self.primal_loss.sampler.sample(n_samples)

    def default_sampler(
        self,
        xi: pt.Tensor,
        xi_labels: Optional[pt.Tensor],
        epsilon: pt.Tensor,
        seed: Optional[int]
    ) -> Optional[BaseSampler]:
        r""" Wraper for the original loss sampler

        Parameters
        ----------
        xi : pt.Tensor
            current batch of :math:`\xi` data samples
        xi_labels : pt.Tensor
            current batch of :math:`\xi` label samples
        epsilon : pt.Tensor
            scalar variance of the default sampler

        Returns
        -------
        sampler : BaseSampler
            default sampler set up by the original loss

        Shapes
        ------
        xi : (m, d)
        xi_labels : (m, d')
        """
        return self.primal_loss.default_sampler(xi, xi_labels, epsilon, seed)

    @property
    @abstractmethod
    def presample(self) -> bool:
        """ ``True`` for :class:`~DualPreSampledLoss`,
        ``False`` for :class:`~DualPostSampledLoss`.

        Returns
        -------
        bool
        """
        raise NotImplementedError()

    @property
    def sampler(self) -> BaseSampler:
        return self.primal_loss.sampler

    @sampler.deleter
    def sampler(self):
        del self.primal_loss.sampler

    @sampler.setter
    def sampler(self, s):
        self.primal_loss.sampler = s
