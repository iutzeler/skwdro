from typing import Any, Tuple, Optional
from abc import ABC, abstractmethod, abstractproperty

import torch as pt
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as ptag

from skwdro.base.costs import Cost
from skwdro.base.losses_torch import Loss
from skwdro.base.samplers.torch.base_samplers import BaseSampler




class _DualLoss(nn.Module, ABC):
    r""" Base class to register parameters for a dual loss:
    .. math::
        \begin{align}
            \rho\lambda\\
            + \epsilon\mathbb{E}_{\xi}\ln{\left(\mathbb{E}_{\zeta\sim\pi_0}e^{\frac{L_\theta(\zeta)-\lambda c(\zeta,\xi)}{\epsilon}}\right)}

    Parameters
    ----------
    loss : Loss
        regular loss :math:`L_\theta`, with a forward pass available with ``__call__``
    cost : Cost
        transport (ground) cost :math:`c(\xi, \zeta)`
    n_samples : int
        number of :math:`zeta` samples to draw from :math:`\pi_0`
    epsilon_0 : pt.Tensor
        first guess for a good regularization value :math:`\epsilon` for Sinkhorn
    rho_0 : pt.Tensor
        first guess for a good maximal distance between xi distribution and adversarial distribution
    n_iter : int
        number of gradient descent updates
    gradient_hypertuning : bool
        [WIP] set to ``True`` to tune rho and epsilon.

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

    Shapes
    ------
    rho_0: (1,)
    epsilon_0: (1,)
    """
    def __init__(self,
                 loss: Loss,
                 cost: Cost,
                 n_samples: int,
                 epsilon_0: pt.Tensor,
                 rho_0: pt.Tensor,
                 n_iter: int,
                 gradient_hypertuning: bool=False
                 ) -> None:
        super(_DualLoss, self).__init__()
        self.primal_loss = loss
        # TODO: implement __call__ for torch costs
        self.cost = cost.value

        # epsilon and rho are parameters so that they can be printed if needed.
        # But they are not included in the autograd graph (requires_grad=False).
        self.epsilon = nn.Parameter(epsilon_0, requires_grad=gradient_hypertuning)
        self.rho = nn.Parameter(rho_0, requires_grad=gradient_hypertuning)

        # Lambda is tuned during training, and it requires a proxy in its parameter form.
        # _lam is the tuned variable, and softplus(_lam) is the "proxy" that is accessed via
        # self.lam in the code (see the parameter decorated method).
        self._lam = nn.Parameter(1e-2 / rho_0) if rho_0 > 0. else pt.tensor(0.)

        # Private sampler points to the loss l_theta
        self._sampler = loss._sampler

        # number of zeta samples are checked at __init__, but can be dynamically changed
        self.n_samples = n_samples
        self.n_iter = n_iter
        self._opti = None

    @property
    def iterations(self):
        return range(self.n_iter)

    @abstractmethod
    def forward(self, *args):
        raise NotImplementedError()

    def compute_dual(self,
                     xi: pt.Tensor,
                     xi_labels: Optional[pt.Tensor],
                     zeta: pt.Tensor,
                     zeta_labels: Optional[pt.Tensor]
                     ) -> pt.Tensor:
        r""" Computes the forward pass for the dual loss value

        Parameters
        ----------
        xi : pt.Tensor
            original data samples
        xi_labels : Optional[pt.Tensor]
            original label samples
        zeta : pt.Tensor
            data samples generated from :math:`\pi_0`
        zeta_labels : Optional[pt.Tensor]
            labels samples generated from :math:`\pi_0`

        Returns
        -------
        dl: pt.Tensor
            dual loss, contracted as a scalar tensor

        Shapes
        ------
        xi : (m, d)
        xi_labels : (m, d')
        zeta : (n_samples, m, d)
        zeta_labels : (n_samples, m, d')
        dl : (1,)
        """
        if self.rho > 0.:
            first_term = self.lam * self.rho # (1,)

            # NOTE: Beware of the shape of the loss, we need a trailing dim
            l = self.primal_loss.value(zeta, zeta_labels) # -> (n_samples, m, 1)
            c = self.cost(
                    xi.unsqueeze(0), # (1, m, d)
                    zeta, # (n_samples, m, d)
                    xi_labels.unsqueeze(0) if xi_labels is not None else None, # (1, m, d') or None
                    zeta_labels # (n_samples, m, d') or None
                ) # -> (n_samples, m, 1)
            integrand = l - self.lam * c # -> (n_samples, m, 1)
            integrand /= self.epsilon # -> (n_samples, m, 1)

            # Expectation on the zeta samples (collapse 1st dim)
            second_term = pt.logsumexp(integrand, 0).mean(dim=0) # -> (m, 1)
            second_term -= pt.log(pt.tensor(zeta.size(0))) # -> (m, 1)
            return first_term + self.epsilon*second_term.mean() # (1,)
        elif self.rho == 0.:
            return self.rho * self.lam + self.primal_loss(
                    xi.unsqueeze(0), # (1, m, d)
                    xi_labels.unsqueeze(0) if xi_labels is not None else None # (1, m, d') or None
                ).mean() # (1,)
        elif self.rho.isnan().any():
            return pt.tensor(pt.nan, requires_grad=True)
        else:
            raise ValueError("Rho < 0 detected: -> " + str(self.rho.item()) + ", please provide a positive rho value")

    def generate_zetas(self,
                       n_samples: Optional[int]=None
                       ) -> Tuple[pt.Tensor, Optional[pt.Tensor]]:
        """ Generate zeta samples from the loss-sampler

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
        if n_samples is None or n_samples <= 0:
            # Default:
            return self.primal_loss.sampler.sample(self.n_samples)
        else:
            return self.primal_loss.sampler.sample(n_samples)

    def default_sampler(self, xi: pt.Tensor, xi_labels: Optional[pt.Tensor], epsilon: pt.Tensor, seed: int) -> BaseSampler:
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

    @abstractproperty
    def presample(self) -> bool:
        """ ``True`` for :class:`~DualPreSampledLoss`, ``False`` for :class:`~DualPostSampledLoss`.

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

    @property
    def theta(self):
        """ Any inner parameters that are not considered an intercept or a lagrangian parameter.
        """
        return self.primal_loss.theta

    @property
    def intercept(self):
        """ Any inner parameters from the primal loss that could be interpreted as an "intercept" or "bias".
        """
        return self.primal_loss.intercept

    @property
    def lam(self) -> pt.Tensor:
        r"""
        Proxy for the lambda parameter.
        ..math::
            \lambda := \mbox{soft}^+(\tilde{\lambda}})
        """
        return F.softplus(self._lam)

    @property
    def optimizer(self) -> pt.optim.Optimizer:
        """ optimizer set up by the user

        Returns
        -------
        optimizer : pt.optim.Optimizer

        """
        if self._opti is None:
            raise AttributeError("Optimizer for:\n"+self.__str__()+"\nis ill defined (None), please set it beforehand")
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

class DualPostSampledLoss(_DualLoss):
    r""" Dual loss implementing a sampling of the :math:`\zeta` vectors at each forward pass.

    Parameters
    ----------
    loss : Loss
        the loss of interest :math:`L_\theta`
    cost : Cost
        ground-distance function
    n_samples : int
        number of :math:`\zeta` samples to draw at each forward pass
    """
    def __init__(self,
                 loss: Loss,
                 cost: Cost,
                 n_samples: int,
                 epsilon_0: pt.Tensor,
                 rho_0: pt.Tensor,
                 n_iter: int=10000,
                 gradient_hypertuning: bool=False
                 ) -> None:
        super(DualPostSampledLoss, self).__init__(loss, cost, n_samples, epsilon_0, rho_0, n_iter, gradient_hypertuning)

        self._opti = pt.optim.AdamW(
                self.parameters(),
                lr=5e-2,
                betas=(.99, .999),
                weight_decay=0.,
                amsgrad=True,
                foreach=True)

    def reset_sampler_mean(self, xi: pt.Tensor, xi_labels: Optional[pt.Tensor]=None):
        """ Prepare the sampler for a new batch of :math:`xi` data.

        Parameters
        ----------
        xi : pt.Tensor
            new data batch
        xi_labels : Optional[pt.Tensor]
            new labels batch
        """
        self.primal_loss.sampler.reset_mean(xi, xi_labels)

    def forward(self, xi: pt.Tensor, xi_labels: Optional[pt.Tensor]=None, reset_sampler: bool=False) -> pt.Tensor:
        """ Forward pass for the dual loss, with the sampling of the adversarial samples

        Parameters
        ----------
        xi : pt.Tensor
            data batch
        xi_labels : Optional[pt.Tensor]
            labels batch
        reset_sampler : bool
            defaults to ``False``, if set resets the batch saved in the sampler

        Returns
        -------
        dl : pt.Tensor

        Shapes
        ------
        xi : (m, d)
        xi_labels : (m, d')
        dl : (1,)
        """
        if reset_sampler: self.reset_sampler_mean(xi, xi_labels)
        if self.rho < 0.:
            raise ValueError("Rho < 0 detected: -> " + str(self.rho.item()) + ", please provide a positive rho value")
        elif self.rho == 0.:
            return self.rho * self.lam + self.primal_loss(
                    xi.unsqueeze(0), # (1, m, d)
                    xi_labels.unsqueeze(0) if xi_labels is not None else None # (1, m, d') or None
                ).mean() # (1,)
        else:
            zeta, zeta_labels = self.generate_zetas(self.n_samples)
            return self.compute_dual(xi, xi_labels, zeta, zeta_labels)

    def __str__(self):
        return "Dual loss (sample IN for loop)\n" + 10*"-" + "\n".join(map(str, self.parameters()))

    @property
    def presample(self):
        return False

class DualPreSampledLoss(_DualLoss):
    r""" Dual loss implementing a forward pass without resampling the :math:`\zeta` vectors.

    Parameters
    ----------
    loss : Loss
        the loss of interest :math:`L_\theta`
    cost : Cost
        ground-distance function
    n_samples : int
        number of :math:`\zeta` samples to draw before the gradient descent begins (can be changed if needed between inferences)
    """
    zeta:        Optional[pt.Tensor]
    zeta_labels: Optional[pt.Tensor]
    def __init__(self,
                 loss: Loss,
                 cost: Cost,
                 n_samples: int,
                 epsilon_0: pt.Tensor,
                 rho_0: pt.Tensor,
                 n_iter: int=50,
                 gradient_hypertuning: bool=False
                 ) -> None:
        super(DualPreSampledLoss, self).__init__(loss, cost, n_samples, epsilon_0, rho_0, n_iter, gradient_hypertuning)

        self._opti = pt.optim.LBFGS(
                self.parameters(),
                lr=1.,
                max_iter=1,
                max_eval=10,
                tolerance_grad=1e-4,
                tolerance_change=1e-6,
                history_size=30)

        self.zeta        = None
        self.zeta_labels = None

    def forward(self, xi: pt.Tensor, xi_labels: Optional[pt.Tensor]=None, zeta: Optional[pt.Tensor]=None, zeta_labels: Optional[pt.Tensor]=None):
        r""" Forward pass for the dual loss, wrt the already sampled :math:`\zeta` values

        Parameters
        ----------
        xi : pt.Tensor
            data batch
        xi_labels : Optional[pt.Tensor]
            labels batch
        zeta : Optional[pt.Tensor]
            data batch
        zeta_labels : Optional[pt.Tensor]
            labels batch

        Returns
        -------
        dl : pt.Tensor

        Shapes
        ------
        xi : (m, d)
        xi_labels : (m, d')
        dl : (1,)
        """
        if zeta is None:
            if self.zeta is None:
                # No previously registered samples, fail
                raise ValueError("Please provide a zeta value for the forward pass of DualPreSampledLoss, else switch to an instance of DualPostSampledLoss.")
            else:
                # Reuse the same samples as last forward pass
                return self.compute_dual(xi, xi_labels, self.zeta, self.zeta_labels)
        else:
            self.zeta        = zeta
            self.zeta_labels = zeta_labels
            return self.compute_dual(xi, xi_labels, zeta, zeta_labels)

    def __str__(self):
        return "Dual loss (sample BEFORE for loop)\n" + 10*"-" + "\n".join(map(str, self.parameters()))

    @property
    def presample(self):
        return True

    @property
    def current_samples(self) -> Tuple[Optional[pt.Tensor], Optional[pt.Tensor]]:
        return self.zeta, self.zeta_labels

"""
DualLoss is an alias for the "post sampled loss" (resample at every forward pass)
"""
DualLoss = DualPostSampledLoss













### [WIP] DO NOT TOUCH, HIGH RISK OF EXPLOSION ###
def entropic_loss_oracle(
        lam,
        zeta,
        zeta_labels,
        xi,
        xi_labels,
        rho,
        epsilon,
        loss,
        cost
        ):
    result, _, _ = EntropicLossOracle.apply(
            lam, zeta, zeta_labels, xi, xi_labels, rho, epsilon, loss, cost
            )
    return result

class EntropicLossOracle(ptag.Function):

    @staticmethod
    def forward(
            lam,
            zeta,
            zeta_labels,
            xi,
            xi_labels,
            rho,
            epsilon,
            loss,
            cost
            ):
        first_term = lam * rho

        l = loss.value(zeta, zeta_labels)
        c = cost(xi.unsqueeze(-1), zeta, xi_labels, zeta_labels)
        integrand = l - lam * c
        integrand /= epsilon

        # Expectation on the zeta samples
        second_term = pt.logsumexp(integrand, 0).mean(dim=0)
        second_term -= pt.log(pt.tensor(zeta.size(0)))
        # second_term *= epsilon
        print(c.shape, l.shape)
        return first_term + epsilon*second_term.mean(), c, l

    @staticmethod
    def setup_context(ctx: Any, inputs: Tuple, output: Any):
        lam, _, _, xi, xi_labels, rho, epsilon, _, _ = inputs
        _, c, l = output
        ctx.save_for_backward(lam, xi, xi_labels, rho, epsilon, c, l)

    @staticmethod
    def backward(ctx, grad_result, grad_c, grad_l):
        if grad_result is None:
            return 9*(None,)
        grad_theta = grad_lam = None
        grad_xi = grad_xi_labels = None
        grad_rho = grad_epsilon = None

        lam, xi, xi_labels, rho, epsilon, c, l = ctx.saved_tensors

        print("# gradl #####")
        print(grad_l)
        print("# gradc #####")
        print(grad_c)
        grad_l_theta, grad_l_zeta, grad_l_zeta_labels = grad_l
        grad_c_xi, grad_c_zeta, grad_c_xi_labels, grad_c_zeta_labels = grad_c
        if ctx.needs_input_grad[0]:
            grad_theta = EntropicLossOracle.grad_theta(
                    lam,
                    epsilon,
                    c,
                    l,
                    grad_l_theta
                )
        if ctx.needs_input_grad[1]:
            grad_lam = EntropicLossOracle.grad_lam(
                    lam,
                    rho,
                    epsilon,
                    c,
                    l
                )


        return grad_theta, grad_lam, None, None, grad_xi, grad_xi_labels, grad_rho, grad_epsilon, None, None

    @staticmethod
    def grad_lam(lam, rho, epsilon, c, l):
        integrand = l - lam * c
        integrand /= epsilon
        return rho - (c * F.softmax(integrand, dim=0)).sum(dim=0).mean()

    @staticmethod
    def grad_theta(lam, epsilon, c, l, grad_l_theta):
        integrand = l - lam * c
        integrand /= epsilon
        return (grad_l_theta * F.softmax(integrand, dim=0)).sum(dim=0).mean()
