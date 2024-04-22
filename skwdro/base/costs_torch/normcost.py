from typing import Optional, Tuple
from types import NoneType

import torch as pt

from .base_cost import TorchCost
import skwdro.distributions as dst


class NormCost(TorchCost):
    """ p-norm to some power, with torch arguments
    """

    def __init__(self, p: float = 1., power: float = 1.,
                 name: Optional[str] = None):
        r"""
        Norm to represent the ground cost of type :math:`p`.
        It represents a distance depending on :math:`p`:
            * for :math:`p=1`: Manhattan
            * for :math:`p=2`: Euclidean distance
            * for :math:`p=\infty`: Sup-norm
        """
        super().__init__(name="Norm" if name is None else name, engine="pt")
        self.p = p
        self.power = power

    def value(
        self,
        xi: pt.Tensor,
        zeta: pt.Tensor,
        xi_labels: NoneType = None,
        zeta_labels: NoneType = None
    ):
        r"""
        Cost to displace :math:`\xi` to :math:`\zeta` in :math:`mathbb{R}^n`.

        Parameters
        ----------
        xi : Tensor
            Data point to be displaced
        zeta : Tensor
            Data point towards which ``xi`` is displaced
        """
        del xi_labels, zeta_labels
        diff = xi - zeta
        return pt.norm(diff, p=self.p, dim=-1, keepdim=True)**self.power

    def _sampler_data(self, xi, epsilon) -> pt.distributions.Distribution:
        if epsilon is None:
            epsilon = pt.tensor(1e-3)
        assert isinstance(epsilon, pt.Tensor)
        if self.power == 1:
            if self.p == 1:
                return dst.Laplace(
                    loc=xi,
                    scale=epsilon.to(xi)
                )
            elif self.p == 2:
                return dst.Normal(
                    loc=xi,
                    scale=epsilon.to(xi)
                )
            elif self.p == pt.inf:
                Warning("For sup norm, we use a gaussian sampler by default.")
                return dst.Normal(
                    loc=xi,
                    scale=epsilon.to(xi)
                )
            else:
                raise NotImplementedError()
        elif self.power == 2:
            if self.p == 2:
                return dst.Normal(
                    loc=xi,
                    scale=epsilon.to(xi)
                )
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

    def _sampler_labels(
        self,
        xi_labels,
        epsilon
    ) -> Optional[pt.distributions.Distribution]:
        if xi_labels is None:
            return None
        else:
            return dst.Dirac(xi_labels, 1, True)

    def solve_max_series_exp(
            self,
            xi: pt.Tensor,
            xi_labels: Optional[pt.Tensor],
            rhs: pt.Tensor,
            rhs_labels: Optional[pt.Tensor]
    ) -> Tuple[pt.Tensor, Optional[pt.Tensor]]:
        if xi_labels is not None and rhs_labels is not None:
            if self.p == 2 == self.power:
                return xi + .5 * rhs, xi_labels  # NO adding + .5 * rhs_labels
            else:
                raise NotImplementedError()
        else:
            if self.p == 2 == self.power:
                return xi + .5 * rhs, xi_labels
            else:
                raise NotImplementedError()
