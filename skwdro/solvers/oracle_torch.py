from typing import Tuple, Optional
from itertools import chain

import torch as pt

from prodigyopt import Prodigy
from mechanic_pytorch import mechanize

from skwdro.base.costs_torch import Cost
from skwdro.base.losses_torch import Loss
from skwdro.solvers._dual_interfaces import _DualLoss
from skwdro.solvers.utils import Steps, interpret_steps_struct

IMP_SAMP = True

class CompositeOptimizer(pt.optim.Optimizer):
    def __init__(self, params, lbd, n_iter, optimizer):
        self.lbd = lbd
        if optimizer == 'mechanic':
            make_optim = lambda params : mechanize(pt.optim.Adam)(params, lr=1.0, weight_decay=0.)
            self.opts = {
                    'params': make_optim(params),
                    'lbd': make_optim([lbd])
                    }
            self.schedulers = {}
        elif optimizer == 'prodigy':
            make_optim = lambda params : Prodigy(params, lr=1.0, weight_decay=0, safeguard_warmup=True, use_bias_correction=True)
            self.opts = {
                    'params': make_optim(params),
                    'lbd': make_optim([lbd])
                    }
            pretrain_iters, train_iters = interpret_steps_struct(n_iter)
            T = {'params': pretrain_iters + train_iters, 'lbd': train_iters}
            self.schedulers = {k:pt.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T[k]) for (k, opt) in self.opts.items()}

        self.init_state_lbd = self.opts['lbd'].state_dict()
        super(CompositeOptimizer, self).__init__(chain(params, [lbd]), {})

    def __getstate__(self) -> object:
        s = {key:val.__getstate__() for key, val in self.opts.items()}
        s['init_state_lbd'] = self.init_state_lbd
        s["defaults"] = {}
        return s

    def step(self):
        for opt in self.opts.values():
            opt.step()
        for scheduler in self.schedulers.values():
            scheduler.step()
        with pt.no_grad():
            self.lbd.clamp_(0., None)

    def zero_grad(self):
        for opt in self.opts.values():
            opt.zero_grad()

    def state_dict(self):
        return {k:opt.state_dict() for (k, opt) in self.opts.items()}

    def load_state_dict(self, d):
        for (k, opt) in self.opts.items():
            opt.load_state_dict(d[k])

    def reset_lbd_state(self):
        self.opts['lbd'].load_state_dict(self.init_state_lbd)


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
                 n_iter: Steps=10000,
                 gradient_hypertuning: bool=False,
                 *,
                 imp_samp: bool=IMP_SAMP,
                 adapt="prodigy",
                 ) -> None:
        super(DualPostSampledLoss, self).__init__(loss, cost, n_samples, epsilon_0, rho_0, n_iter, gradient_hypertuning, imp_samp=imp_samp)
        if adapt:
            assert adapt in ("mechanic", "prodigy")
            self._opti = CompositeOptimizer(self.primal_loss.parameters(), self.lam, n_iter, adapt)

        else:
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
                 n_iter: Steps=50,
                 gradient_hypertuning: bool=False,
                 *,
                 imp_samp: bool=IMP_SAMP,
                 adapt="prodigy",
                 ) -> None:
        super(DualPreSampledLoss, self).__init__(loss, cost, n_samples, epsilon_0, rho_0, n_iter, gradient_hypertuning, imp_samp=imp_samp)

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













# ### [WIP] DO NOT TOUCH, HIGH RISK OF EXPLOSION ###
# def entropic_loss_oracle(
#         lam,
#         zeta,
#         zeta_labels,
#         xi,
#         xi_labels,
#         rho,
#         epsilon,
#         loss,
#         cost
#         ):
#     result, _, _ = EntropicLossOracle.apply(
#             lam, zeta, zeta_labels, xi, xi_labels, rho, epsilon, loss, cost
#             )
#     return result
#
# class EntropicLossOracle(ptag.Function):
#
#     @staticmethod
#     def forward(
#             lam,
#             zeta,
#             zeta_labels,
#             xi,
#             xi_labels,
#             rho,
#             epsilon,
#             loss,
#             cost
#             ):
#         first_term = lam * rho
#
#         l = loss.value(zeta, zeta_labels)
#         c = cost(xi.unsqueeze(-1), zeta, xi_labels, zeta_labels)
#         integrand = l - lam * c
#         integrand /= epsilon
#
#         # Expectation on the zeta samples
#         second_term = pt.logsumexp(integrand, 0).mean(dim=0)
#         second_term -= pt.log(pt.tensor(zeta.size(0)))
#         # second_term *= epsilon
#         print(c.shape, l.shape)
#         return first_term + epsilon*second_term.mean(), c, l
#
#     @staticmethod
#     def setup_context(ctx: Any, inputs: Tuple, output: Any):
#         lam, _, _, xi, xi_labels, rho, epsilon, _, _ = inputs
#         _, c, l = output
#         ctx.save_for_backward(lam, xi, xi_labels, rho, epsilon, c, l)
#
#     @staticmethod
#     def backward(ctx, grad_result, grad_c, grad_l):
#         if grad_result is None:
#             return 9*(None,)
#         grad_theta = grad_lam = None
#         grad_xi = grad_xi_labels = None
#         grad_rho = grad_epsilon = None
#
#         lam, xi, xi_labels, rho, epsilon, c, l = ctx.saved_tensors
#
#         print("# gradl #####")
#         print(grad_l)
#         print("# gradc #####")
#         print(grad_c)
#         grad_l_theta, grad_l_zeta, grad_l_zeta_labels = grad_l
#         grad_c_xi, grad_c_zeta, grad_c_xi_labels, grad_c_zeta_labels = grad_c
#         if ctx.needs_input_grad[0]:
#             grad_theta = EntropicLossOracle.grad_theta(
#                     lam,
#                     epsilon,
#                     c,
#                     l,
#                     grad_l_theta
#                 )
#         if ctx.needs_input_grad[1]:
#             grad_lam = EntropicLossOracle.grad_lam(
#                     lam,
#                     rho,
#                     epsilon,
#                     c,
#                     l
#                 )
#
#
#         return grad_theta, grad_lam, None, None, grad_xi, grad_xi_labels, grad_rho, grad_epsilon, None, None
#
#     @staticmethod
#     def grad_lam(lam, rho, epsilon, c, l):
#         integrand = l - lam * c
#         integrand /= epsilon
#         return rho - (c * F.softmax(integrand, dim=0)).sum(dim=0).mean()
#
#     @staticmethod
#     def grad_theta(lam, epsilon, c, l, grad_l_theta):
#         integrand = l - lam * c
#         integrand /= epsilon
#         return (grad_l_theta * F.softmax(integrand, dim=0)).sum(dim=0).mean()
