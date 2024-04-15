from typing import Optional, Tuple

import torch as pt

import skwdro.distributions as dst
from .normcost import NormCost


class Constant(dst.Distribution):
    def __init__(self, cst: pt.Tensor):
        super().__init__(validate_args=False)
        self.cst = cst

    def sample(self, sample_shape=pt.Size()):
        return self.cst.expand(*(sample_shape + self.cst.shape))

    def rsample(self, sample_shape=pt.Size()):
        return self.sample(sample_shape=sample_shape)


class NormLabelCost(NormCost):
    """ p-norm of the ground metric to change data + label
    """

    def __init__(self, p: float = 2., power: float = 1.,
                 kappa: float = 1e4, name: Optional[str] = None):
        r"""
        Norm used to add cost to switching labels:

        .. math::
            d_\kappa\left(\left[\begin{array}{c}\bm{X}\\y\end{array}\right],
                \left[\begin{array}{c}\bm{X'}\\y'\end{array}\right]\right) :=
            \|\bm{X}-\bm{X'}\|+\kappa |y-y'|
        """
        super().__init__(power=power, p=p, name="Kappa-norm" if name is None else name)
        self.name = name  # Overwrite the name
        self.kappa = kappa
        assert kappa >= 0, f"Input kappa={kappa}<0 is illicit since it 'encourages' flipping labels in the database, and thus makes no sense wrt the database in terms of 'trust' to the labels."

    @classmethod
    def _label_penalty(cls, y: pt.Tensor, y_prime: pt.Tensor, p: float):
        return pt.norm(y - y_prime, p=p, dim=-1, keepdim=True)

    @classmethod
    def _data_penalty(cls, x: pt.Tensor, x_prime: pt.Tensor, p: float):
        diff = x - x_prime
        return pt.norm(diff, p=p, dim=-1, keepdim=True)

    def value(
        self,
        xi: pt.Tensor,
        zeta: pt.Tensor,
        xi_labels: pt.Tensor,
        zeta_labels: pt.Tensor
    ):
        r"""
        Cost to displace
        :math:`\xi:=\left[\begin{array}{c}\bm{X}\\y\end{array}\right]`
        to :math:`\zeta:=\left[\begin{array}{c}\bm{X'}\\y'\end{array}\right]`
        in :math:`mathbb{R}^n`.

        Parameters
        ----------
        xi : Tensor, shape (n_samples, n_features)
            Data point to be displaced (without the label)
        zeta : Tensor, shape (n_samples, n_features)
            Data point towards which ``x`` is displaced
        xi_labels : Tensor, shape (n_samples, n_features_y)
            Label or target for the problem/loss
        zeta_labels : Tensor, shape (n_samples, n_features_y)
            Label or target in the dataset
        """
        assert zeta_labels is not None and xi_labels is not None
        if float(self.kappa) is float("inf"):
            # Writing convention: if kappa=+oo we put all cost on switching labels
            #  so the cost is reported on y.
            # To provide a tractable computation, we yield the y-penalty alone.
            return self._label_penalty(
                xi_labels, zeta_labels, self.p)**self.power
        elif self.kappa == 0.:
            # Writing convention: if kappa is null we put all cost on moving the data itself, so the worst-case distribution is free to switch the labels.
            # Warning : this usecase should not make sense anyway.
            return self._data_penalty(xi, zeta, self.p)**self.power
        else:
            distance = self._data_penalty(xi, zeta, self.p) \
                + self.kappa * \
                self._label_penalty(xi_labels, zeta_labels, self.p)
            distance /= 1. + self.kappa
            return distance**self.power

    def _sampler_labels(
        self,
        xi_labels,
        epsilon
    ) -> pt.distributions.Distribution:
        d = xi_labels.size(-1)
        if epsilon is None:
            epsilon = 1e-3
        if self.kappa == float('inf'):
            return dst.Dirac(xi_labels)
        assert isinstance(epsilon, pt.Tensor)
        if self.power == 1:
            if self.p == 1:
                return dst.Laplace(
                    loc=xi_labels,
                    scale=epsilon.to(xi_labels) / self.kappa
                )
            elif self.p == pt.inf:
                Warning("For sup norm, we use a gaussian sampler by default.")
                return dst.Normal(
                    loc=xi_labels,
                    scale=epsilon.to(xi_labels) / self.kappa
                )
            else:
                raise NotImplementedError()
        elif self.power == 2:
            if self.p == 2:
                return dst.MultivariateNormal(
                    loc=xi_labels,
                    scale_tril=epsilon * pt.eye(d) / self.kappa
                )
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

    def solve_max_series_exp(
        self,
        xi: pt.Tensor,
        xi_labels: Optional[pt.Tensor],
        rhs: pt.Tensor,
        rhs_labels: Optional[pt.Tensor]
    ) -> Tuple[pt.Tensor, Optional[pt.Tensor]]:
        if xi_labels is not None and rhs_labels is not None:
            if self.p == 2 == self.power:
                return xi + .5 * rhs, xi_labels + .5 * rhs_labels / self.kappa
            else:
                raise NotImplementedError()
        else:
            return super().solve_max_series_exp(xi, xi_labels, rhs, rhs_labels)
