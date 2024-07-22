from typing import Callable, Dict, Tuple, Optional, overload
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

        def make_optim(params):
            if optimizer == 'mechanic':
                return mechanize(
                    pt.optim.Adam
                )(params, lr=1.0, weight_decay=0.)
            elif optimizer == 'prodigy':
                return Prodigy(
                    params,
                    lr=1.0,
                    weight_decay=0,
                    safeguard_warmup=True,
                    use_bias_correction=True
                )
            else:
                raise NotImplementedError(
                    "No composite optimizer by that name"
                )

        self.opts = {
            'params': make_optim(params),
            'lbd': make_optim([lbd])
        }
        if optimizer == 'prodigy':
            pretrain_iters, train_iters = interpret_steps_struct(n_iter)
            T = {'params': pretrain_iters + train_iters, 'lbd': train_iters}
            self.schedulers = {
                k: pt.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=T[k]
                ) for (k, opt) in self.opts.items()
            }
        else:
            self.schedulers = {}

        self.init_state_lbd = self.opts['lbd'].state_dict()
        super(CompositeOptimizer, self).__init__(chain(params, [lbd]), {})

    def __getstate__(self) -> Dict[str, object]:
        s = {key: val.__getstate__() for key, val in self.opts.items()}
        s['init_state_lbd'] = self.init_state_lbd
        s["defaults"] = {}
        return s

    @overload
    def step(self, closure: None = None) -> None:
        ...

    @overload
    def step(self, closure: Callable) -> float:
        raise NotImplementedError(
            "Please provide a null callable to the step f°"
        )

    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        del closure
        for opt in self.opts.values():
            opt.step()
        for scheduler in self.schedulers.values():
            scheduler.step()
        with pt.no_grad():
            self.lbd.clamp_(0., None)
        return None

    def zero_grad(self, *args, **kwargs):
        del args
        del kwargs
        for opt in self.opts.values():
            opt.zero_grad()

    def state_dict(self):
        return {k: opt.state_dict() for (k, opt) in self.opts.items()}

    def load_state_dict(self, state_dict):
        for (k, opt) in self.opts.items():
            opt.load_state_dict(state_dict[k])

    def reset_lbd_state(self):
        self.opts['lbd'].load_state_dict(self.init_state_lbd)


class DualPostSampledLoss(_DualLoss):
    r"""
    Dual loss implementing a sampling of the :math:`\zeta` vectors at
    each forward pass.

    Parameters
    ----------
    loss : Loss
        the loss of interest :math:`L_\theta`
    cost : Cost
        ground-distance function
    n_samples : int
        number of :math:`\zeta` samples to draw at each forward pass
    """

    def __init__(
        self,
        loss: Loss,
        cost: Cost,
        n_samples: int,
        epsilon_0: pt.Tensor,
        rho_0: pt.Tensor,
        n_iter: Steps = 10000,
        gradient_hypertuning: bool = False,
        *,
        imp_samp: bool = IMP_SAMP,
        adapt: Optional[str] = "prodigy",
    ) -> None:
        super(DualPostSampledLoss, self).__init__(
            loss,
            cost,
            n_samples,
            epsilon_0,
            rho_0,
            n_iter,
            gradient_hypertuning,
            imp_samp=imp_samp
        )
        if adapt:
            assert adapt in ("mechanic", "prodigy")
            self._opti = CompositeOptimizer(
                self.primal_loss.parameters(), self.lam, n_iter, adapt)

        else:
            self._opti = pt.optim.AdamW(
                self.parameters(),
                lr=5e-2,
                betas=(.99, .999),
                weight_decay=0.,
                amsgrad=True,
                foreach=True
            )

    def reset_sampler_mean(
        self,
        xi: pt.Tensor,
        xi_labels: Optional[pt.Tensor] = None
    ):
        """ Prepare the sampler for a new batch of :math:`xi` data.

        Parameters
        ----------
        xi : pt.Tensor
            new data batch
        xi_labels : Optional[pt.Tensor]
            new labels batch
        """
        self.primal_loss.sampler.reset_mean(xi, xi_labels)

    @overload
    def forward(
        self,
        xi: pt.Tensor,
        xi_labels: Optional[pt.Tensor] = None,
        zeta: None = None,
        zeta_labels: None = None,
        reset_sampler: bool = False
    ) -> pt.Tensor:
        pass

    @overload
    def forward(
        self,
        xi: pt.Tensor,
        xi_labels: Optional[pt.Tensor],
        zeta: pt.Tensor,
        zeta_labels: Optional[pt.Tensor] = None,
        reset_sampler: bool = False
    ) -> pt.Tensor:
        raise ValueError(
            "This class does not support forwarding pre-sampled zetas"
        )

    def forward(
        self,
        xi: pt.Tensor,
        xi_labels: Optional[pt.Tensor] = None,
        zeta: Optional[pt.Tensor] = None,
        zeta_labels: Optional[pt.Tensor] = None,
        reset_sampler: bool = False
    ) -> Optional[pt.Tensor]:
        """
        Forward pass for the dual loss, with the sampling of the
        adversarial samples

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
        del zeta, zeta_labels
        if reset_sampler:
            self.reset_sampler_mean(xi, xi_labels)
        if self.rho < 0.:
            raise ValueError(' '.join([
                "Rho < 0 detected: ->",
                str(self.rho.item()),
                ", please provide a positive rho value"
            ]))
        elif self.rho == 0.:
            first_term = self.rho * self.lam
            _pl: pt.Tensor = self.primal_loss(
                xi.unsqueeze(0),  # (1, m, d)
                # (1, m, d') or None
                xi_labels.unsqueeze(0) if xi_labels is not None else None
            ).mean()  # (1,)
            return first_term + _pl
        else:
            zeta_, zeta_labels_ = self.generate_zetas(self.n_samples)
            return self.compute_dual(xi, xi_labels, zeta_, zeta_labels_)

    def __str__(self):
        return "Dual loss (sample IN for loop)\n" + 10 * "-" + "\n".join(
            map(str, self.parameters())
        )

    @property
    def presample(self):
        return False


class DualPreSampledLoss(_DualLoss):
    r""" Dual loss implementing a forward pass without resampling the
    :math:`\zeta` vectors.

    Parameters
    ----------
    loss : Loss
        the loss of interest :math:`L_\theta`
    cost : Cost
        ground-distance function
    n_samples : int
        number of :math:`\zeta` samples to draw before the gradient
        descent begins (can be changed if needed between inferences).
    """
    zeta: Optional[pt.Tensor]
    zeta_labels: Optional[pt.Tensor]

    def __init__(
        self,
        loss: Loss,
        cost: Cost,
        n_samples: int,
        epsilon_0: pt.Tensor,
        rho_0: pt.Tensor,
        n_iter: Steps = 50,
        gradient_hypertuning: bool = False,
        *,
        imp_samp: bool = IMP_SAMP,
        adapt: Optional[str] = "prodigy",
    ) -> None:
        del adapt
        super(DualPreSampledLoss, self).__init__(
            loss,
            cost,
            n_samples,
            epsilon_0,
            rho_0,
            n_iter,
            gradient_hypertuning,
            imp_samp=imp_samp
        )

        self._opti = pt.optim.LBFGS(
            self.parameters(),
            lr=1.,
            max_iter=1,
            max_eval=10,
            tolerance_grad=1e-4,
            tolerance_change=1e-6,
            history_size=30
        )

        self.zeta = None
        self.zeta_labels = None

    @overload
    def forward(
        self,
        xi: pt.Tensor,
        xi_labels: Optional[pt.Tensor] = None,
        zeta: None = None,
        zeta_labels: None = None,
        reset_sampler: bool = False
    ) -> pt.Tensor:
        raise NotImplementedError(
            "This class must forward pre-sampled zeta values"
        )

    @overload
    def forward(
        self,
        xi: pt.Tensor,
        xi_labels: Optional[pt.Tensor],
        zeta: pt.Tensor,
        zeta_labels: Optional[pt.Tensor] = None,
        reset_sampler: bool = False
    ):
        del xi, xi_labels, zeta, zeta_labels, reset_sampler

    def forward(
        self,
        xi: pt.Tensor,
        xi_labels: Optional[pt.Tensor] = None,
        zeta: Optional[pt.Tensor] = None,
        zeta_labels: Optional[pt.Tensor] = None,
        reset_sampler: bool = False
    ) -> pt.Tensor:
        r""" Forward pass for the dual loss, wrt the already sampled
        :math:`\zeta` values

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
        del reset_sampler
        if zeta is None:
            if self.zeta is None:
                # No previously registered samples, fail
                raise ValueError(' '.join([
                    "Please provide a zeta value for the forward pass of",
                    "DualPreSampledLoss, else switch to",
                    "an instance of DualPostSampledLoss."
                ]))
            else:
                # Reuse the same samples as last forward pass
                return self.compute_dual(
                    xi,
                    xi_labels,
                    self.zeta,
                    self.zeta_labels
                )
        else:
            self.zeta = zeta
            self.zeta_labels = zeta_labels
            return self.compute_dual(xi, xi_labels, zeta, zeta_labels)

    def __str__(self):
        return "Dual loss (sample BEFORE for loop)\n" + 10 * "-" + "\n".join(
            map(str, self.parameters())
        )

    @property
    def presample(self):
        return True

    @property
    def current_samples(
        self
    ) -> Tuple[
        Optional[pt.Tensor],
        Optional[pt.Tensor]
    ]:
        return self.zeta, self.zeta_labels


"""
DualLoss is an alias for the "post sampled loss"
(resample at every forward pass)
"""
DualLoss = DualPostSampledLoss
