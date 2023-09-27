from typing import Optional

import torch as pt
import torch.nn.functional as F

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
        if self.rho > 0.:
            first_term = self.lam * self.rho # (1,)

            if self.imp_samp:
                # For importance sampling, we displace all the samples.
                # They are now sampled around xi*, the displaced xi values.
                xi_star, xi_labels_star, zeta, zeta_labels = self.displace_samples(xi, xi_labels, zeta, zeta_labels)

                # The dual loss contains the sampled terms (in the exponential) and the importance sampling part
                # ##############################################################################################

                # Main terms:
                # -----------
                # L(zeta) - lambda*V(zeta|xi)
                # NOTE: Beware of the shape of the loss, we need a trailing dim
                l = self.primal_loss.value(zeta, zeta_labels) # -> (n_samples, m, 1)
                c = self.cost(
                        xi.unsqueeze(0), # (1, m, d)
                        zeta, # (n_samples, m, d)
                        maybe_unsqueeze(xi_labels, dim=0), # (1, m, d') or None
                        zeta_labels # (n_samples, m, d') or None
                    ) # -> (n_samples, m, 1)
                integrand = l - self.lam * c # -> (n_samples, m, 1)
                integrand /= self.epsilon # -> (n_samples, m, 1)

                # Importance sampling terms:
                # --------------------------
                # - [ V(zeta|xi)
                #   - V(zeta|xi*) ]
                correction = self.sampler.log_prob(
                        xi.unsqueeze(0),
                        maybe_unsqueeze(xi_labels, dim=0),
                        zeta,
                        zeta_labels
                    ) - self.sampler.log_prob(
                        xi_star,
                        xi_labels_star,
                        zeta,
                        zeta_labels
                        )
                #print("Corr mean: ", correction.mean().item())
                #print("Integ mean: ", integrand.mean().item())
                integrand -= correction # (n_samples, m, 1)
                #print("Integ-corr mean: ", integrand.mean().item())
            else:
                l = self.primal_loss.value(zeta, zeta_labels) # -> (n_samples, m, 1)
                c = self.cost(
                        xi.unsqueeze(0), # (1, m, d)
                        zeta, # (n_samples, m, d)
                        maybe_unsqueeze(xi_labels, dim=0), # (1, m, d') or None
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
                    maybe_unsqueeze(xi_labels, dim=0), # (1, m, d') or None
                ).mean() # (1,)
        elif self.rho.isnan().any():
            return pt.tensor(pt.nan, requires_grad=True)
        else:
            raise ValueError("Rho < 0 detected: -> " + str(self.rho.item()) + ", please provide a positive rho value")

    def get_initial_guess_at_dual(self, xi: pt.Tensor, xi_labels: Optional[pt.Tensor]):
        c = self.cost
        rho_N = xi.size(0) * self.rho # (1,)
        if issubclass(type(c), NormCost):
            p = pt.tensor(c.power)
            q: pt.Tensor = p / (p - 1)

            grads, grads_labels = self.get_optimal_displacement(xi, xi_labels) # (1, m, d), (1, m, d')
            with pt.no_grad():
                grads_norms = \
                        c.value(
                                grads,
                                pt.zeros_like(grads),
                                grads_labels,
                                pt.zeros_like(grads_labels) if grads_labels is not None else None
                                )\
                        * self._lam # (1, m, 1)

                lam0 = ((grads_norms * rho_N)
                        .pow(q - 1)
                        .sum()
                        .pow(1. / q)
                       ) / (p * rho_N)
                self._lam.data.mul(0.).add(lam0)
            return lam0
        else: raise


class _DualLoss(_DualFormulation):

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
        #return self._lam
