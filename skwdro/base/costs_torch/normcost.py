from typing import Optional, Tuple
from types import NoneType

import torch as pt

from .base_cost import TorchCost
import skwdro.distributions as dst

class NormCost(TorchCost):
    """ p-norm to some power, with torch arguments
    """
    def __init__(self, p: float=1., power: float=1., name: Optional[str]=None):
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

    def value(self, xi: pt.Tensor, zeta: pt.Tensor, xi_labels: NoneType=None, zeta_labels: NoneType=None):
        r"""
        Cost to displace :math:`\xi` to :math:`\zeta` in :math:`mathbb{R}^n`.

        Parameters
        ----------
        xi : Tensor
            Data point to be displaced
        zeta : Tensor
            Data point towards which ``xi`` is displaced
        """
        diff = xi - zeta
        return pt.norm(diff, p=self.p, dim=-1, keepdim=True)**self.power

    def _sampler_data(self, xi, epsilon):
        if self.power == 1:
            if self.p == 1:
                return dst.Laplace(
                            loc=xi,
                            scale=epsilon
                        )
        elif self.power == 2:
            if self.p == 2:
                return dst.MultivariateNormal(
                        loc=xi,
                        scale_tril=epsilon*pt.eye(xi.size(-1))
                    )
            else: raise NotImplementedError()
        else: raise NotImplementedError()

    def _sampler_labels(self, xi_labels, epsilon):
        if xi_labels is None:
            return None
        else:
            return dst.Dirac(xi_labels, 1, True)

    def solve_max_series_exp(
            self,
            xi: pt.Tensor,
            xi_labels: NoneType,
            rhs: pt.Tensor,
            rhs_labels: NoneType
            ) -> Tuple[pt.Tensor, NoneType]:
        if self.p == 2 == self.power:
            return xi + .5 * rhs, None
        else:
            raise NotImplementedError()
