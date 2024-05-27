from typing import Optional

import torch as pt

from ._impsamp_interface import _SampleDisplacer
from skwdro.solvers.utils import maybe_unsqueeze
from skwdro.base.costs_torch import NormCost


class _DualFormulation(_SampleDisplacer):
    def compute_dual(
            self,
            xi: pt.Tensor,
            xi_labels: Optional[pt.Tensor],
            zeta: pt.Tensor,
            zeta_labels: Optional[pt.Tensor]
    ) -> pt.Tensor:
        r""" Computes the forward pass for the dual loss value.

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
        if self.rho == 0. or self.erm_mode:
            _pl: pt.Tensor = self.primal_loss(
                xi.unsqueeze(0),  # (1, m, d)
                maybe_unsqueeze(xi_labels, dim=0),  # (1, m, d') or None
            ).mean()  # (1,)
            first_term: pt.Tensor = self.rho * self.lam
            return first_term + _pl
        elif self.rho > 0.:
            p = pt.tensor(self.cost.power)
            first_term = self.lam * self.rho.pow(p)  # (1,)
            loss_estimate: pt.Tensor
            cost_estimate: pt.Tensor
            integrand: pt.Tensor

            if self.imp_samp:
                # For importance sampling, we displace all the samples.
                # They are now sampled around xi*, the displaced xi values.
                (
                    xi_star,
                    xi_labels_star,
                    zeta,
                    zeta_labels
                ) = self.displace_samples(
                    xi, xi_labels, zeta, zeta_labels
                )

                # The dual loss contains the sampled terms (in the exponential)
                # and the importance sampling part
                # #############################################################

                # Main terms:
                # -----------
                # L(zeta) - lambda*V(zeta|xi)
                # NOTE: Beware of the shape of the loss, we need a trailing dim
                loss_estimate = self.primal_loss.value(
                    zeta, zeta_labels
                )  # -> (n_samples, m, 1)
                cost_estimate = self.cost(
                    # (1, m, d)
                    xi.unsqueeze(0),
                    # (n_samples, m, d)
                    zeta,
                    # (1, m, d') or None
                    maybe_unsqueeze(xi_labels, dim=0),
                    # (n_samples, m, d') or None
                    zeta_labels
                )  # -> (n_samples, m, 1)
                # -> (n_samples, m, 1)
                integrand = loss_estimate - self.lam * cost_estimate
                integrand /= self.epsilon  # -> (n_samples, m, 1)

                # Importance sampling terms:
                # --------------------------
                # - [ V(zeta|xi)
                #   - V(zeta|xi*) ]
                # = + [ lprob(zeta|xi)
                #     - lprob(zeta|xi*)
                correction = self.sampler.log_prob(
                    zeta,
                    zeta_labels
                ) - self.sampler.log_prob_recentered(
                    xi_star,
                    # xi.unsqueeze(0),
                    xi_labels_star,
                    # maybe_unsqueeze(xi_labels, dim=0),
                    zeta,
                    zeta_labels
                )
                integrand += correction  # (n_samples, m, 1)
            else:
                loss_estimate = self.primal_loss.value(
                    zeta, zeta_labels
                )  # -> (n_samples, m, 1)
                cost_estimate = self.cost(
                    xi.unsqueeze(0),  # (1, m, d)
                    zeta,  # (n_samples, m, d)
                    maybe_unsqueeze(xi_labels, dim=0),  # (1, m, d') or None
                    zeta_labels  # (n_samples, m, d') or None
                )  # -> (n_samples, m, 1)
                # -> (n_samples, m, 1)
                integrand = loss_estimate - self.lam * cost_estimate
                integrand /= self.epsilon  # -> (n_samples, m, 1)

            # Expectation on the zeta samples (collapse 1st dim)
            second_term = pt.logsumexp(integrand, 0).mean(dim=0)  # -> (m, 1)
            second_term -= pt.log(pt.tensor(zeta.size(0)))  # -> (m, 1)
            return first_term + self.epsilon * second_term.mean()  # (1,)
        elif self.rho.isnan().any():
            return pt.tensor(pt.nan, requires_grad=True)
        else:
            raise ValueError(' '.join([
                "Rho < 0 detected: -> ",
                str(self.rho.item()),
                ", please provide a positive rho value"
            ]))

    def get_initial_guess_at_dual(
        self,
        xi: pt.Tensor,
        xi_labels: Optional[pt.Tensor]
    ):
        if self.rho > 0.:
            c = self.cost
            if issubclass(type(c), NormCost):
                p = pt.tensor(c.power)
                rho_N = xi.size(0) * self.rho.pow(p)  # (1,)
                q: pt.Tensor = p / (p - 1.) if p != 1. else pt.tensor(pt.inf)

                grads, grads_labels = self.get_displacement_direction(
                    xi,
                    xi_labels
                )  # (1, m, d), (1, m, d')
                with pt.no_grad():
                    grads_norms = c.value(
                        grads,
                        pt.zeros_like(grads),
                        grads_labels,
                        pt.zeros_like(
                            grads_labels) if grads_labels is not None else None
                    )  # (1, m, 1)

                    lam0 = ((grads_norms * rho_N)
                            .pow(q - 1)
                            .sum()
                            .pow(1. / q)
                            ) / (p * rho_N)
                    self._lam.data.mul_(0.).add_(lam0)
                return lam0
            else:
                raise
        else:
            self._lam.data.mul_(0.)


class _DualLoss(_DualFormulation):

    @property
    def theta(self):
        """
        Any inner parameters that are not considered an intercept
        or a lagrangian parameter.
        """
        return self.primal_loss.theta

    @property
    def intercept(self):
        """
        Any inner parameters from the primal loss that could be
        interpreted as an "intercept" or "bias".
        """
        return self.primal_loss.intercept

    @property
    def lam(self) -> pt.Tensor:
        r"""
        Proxy for the lambda parameter.
        ..math::
            \lambda := \mbox{soft}^+(\tilde{\lambda}})
        """
        # return F.softplus(self._lam)
        return self._lam
