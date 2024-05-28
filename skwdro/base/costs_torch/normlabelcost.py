from typing import Optional, Tuple, overload

import torch as pt

import skwdro.distributions as dst
from .normcost import NormCost


class NormLabelCost(NormCost):
    r""" p-norm of the ground metric to change data + label

    Norm used to add cost to switching labels:

    .. math::
        d_\kappa\left(\left[\begin{array}{c}\bm{X}\\y\end{array}\right],
            \left[\begin{array}{c}\bm{X'}\\y'\end{array}\right]\right) :=
        \|\bm{X}-\bm{X'}\|+\kappa |y-y'|
    """
    def __init__(
        self,
        p: float = 2.,
        power: float = 1.,
        kappa: float = 1e4,
        name: Optional[str] = None
    ):
        r""" Constructor
        """
        super().__init__(
            power=power,
            p=p,
            name="Kappa-norm" if name is None else name
        )
        self.kappa = kappa
        assert kappa >= 0, ' '.join([
            f"Input kappa={kappa}<0",
            "is illicit since it 'encourages'",
            "flipping labels in the database,",
            "and thus makes no sense wrt the database",
            "in terms of 'trust' to the labels."
        ])

    @classmethod
    def _label_penalty(cls, y: pt.Tensor, y_prime: pt.Tensor, p: float):
        return pt.norm(y - y_prime, p=p, dim=-1, keepdim=True)

    @classmethod
    def _data_penalty(cls, x: pt.Tensor, x_prime: pt.Tensor, p: float):
        diff = x - x_prime
        return pt.norm(diff, p=p, dim=-1, keepdim=True)

    @overload
    def value(
        self,
        xi: pt.Tensor,
        zeta: pt.Tensor,
        xi_labels: pt.Tensor,
        zeta_labels: pt.Tensor
    ) -> pt.Tensor:
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
        pass

    @overload
    def value(
        self,
        xi: pt.Tensor,
        zeta: pt.Tensor,
        xi_labels: None = None,
        zeta_labels: None = None
    ) -> pt.Tensor:
        pass

    @overload
    def value(
        self,
        xi: pt.Tensor,
        zeta: pt.Tensor,
        xi_labels: Optional[pt.Tensor] = None,
        zeta_labels: Optional[pt.Tensor] = None
    ) -> pt.Tensor:
        raise AssertionError()

    def value(
        self,
        xi: pt.Tensor,
        zeta: pt.Tensor,
        xi_labels: Optional[pt.Tensor] = None,
        zeta_labels: Optional[pt.Tensor] = None
    ) -> pt.Tensor:
        assert xi_labels is not None and zeta_labels is not None
        _c: pt.Tensor
        if float(self.kappa) is float("inf"):
            # Writing convention: if kappa=+oo we put all cost on switching
            # labels so the cost is reported on y.
            # To provide a tractable computation, we yield the y-penalty alone.
            _c = self._label_penalty(
                xi_labels, zeta_labels, self.p
            )**self.power
        elif self.kappa == 0.:
            # Writing convention: if kappa is null we put all cost on moving
            # the data itself, so the worst-case distribution is free to switch
            # the labels.
            # Warning : this usecase should not make sense anyway.
            _c = self._data_penalty(xi, zeta, self.p)**self.power
        else:
            distance = self._data_penalty(xi, zeta, self.p) \
                + self.kappa * \
                self._label_penalty(xi_labels, zeta_labels, self.p)
            distance /= 1. + self.kappa
            _c = distance**self.power
            del distance

        return _c

    @overload
    def _sampler_labels(
        self,
        xi_labels: pt.Tensor,
        epsilon: pt.Tensor
    ) -> dst.Distribution:
        pass

    @overload
    def _sampler_labels(
        self,
        xi_labels: None,
        epsilon: pt.Tensor
    ) -> None:
        raise ValueError()

    def _sampler_labels(
        self,
        xi_labels,
        epsilon
    ) -> Optional[pt.distributions.Distribution]:
        # d = xi_labels.size(-1)
        if epsilon is None:
            epsilon = pt.tensor(1e-3)
        elif not isinstance(epsilon, pt.Tensor):
            epsilon = pt.tensor(epsilon)
        if self.kappa == float('inf'):
            return dst.Dirac(xi_labels)
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
                return dst.Normal(
                    loc=xi_labels,
                    scale=epsilon.to(xi_labels)
                )
                # return dst.MultivariateNormal(
                #     loc=xi_labels,
                #     scale_tril=epsilon * pt.eye(d) / self.kappa
                # )
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

    @overload
    def solve_max_series_exp(
        self,
        xi: pt.Tensor,
        xi_labels: pt.Tensor,
        rhs: pt.Tensor,
        rhs_labels: pt.Tensor
    ) -> Tuple[pt.Tensor, pt.Tensor]:
        pass

    @overload
    def solve_max_series_exp(
        self,
        xi: pt.Tensor,
        xi_labels: Optional[pt.Tensor],
        rhs: pt.Tensor,
        rhs_labels: Optional[pt.Tensor]
    ) -> Tuple[pt.Tensor, Optional[pt.Tensor]]:
        pass

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
