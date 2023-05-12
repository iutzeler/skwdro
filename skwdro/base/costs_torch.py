from typing import Optional
from .costs import Cost

import torch as pt

class NormCost(Cost):
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

    def value(self, xi: pt.Tensor, zeta: pt.Tensor):
        r"""
        Cost to displace :math:`\xi` to :math:`\zeta` in :math:`mathbb{R}^n`.

        Parameters
        ----------
        xi : Tensor
            Data point to be displaced
        zeta : Tensor
            Data point towards which ``xi`` is displaced
        """
        diff = (xi - zeta).reshape(-1)
        return pt.norm(diff, p=self.p)**self.power


class NormLabelCost(NormCost):
    """ p-norm of the ground metric to change data + label
    """

    def __init__(self, p: float=2., power: float=1., kappa: float=1e4, name: Optional[str]=None):
        r"""
        Norm used to add cost to switching labels:

        .. math::
            d_\kappa\left(\left[\begin{array}{c}\bm{X}\\y\end{array}\right],
                \left[\begin{array}{c}\bm{X'}\\y'\end{array}\right]\right) :=
            \|\bm{X}-\bm{X'}\|+\kappa |y-y'|
        """
        super().__init__(power=power, p=p, name="Kappa-norm" if name is None else name)
        self.name = name # Overwrite the name
        self.kappa = kappa
        assert kappa >= 0, f"Input kappa={kappa}<0 is illicit since it 'encourages' flipping labels in the database, and thus makes no sense wrt the database in terms of 'trust' to the labels."

    @classmethod
    def _label_penalty(cls, y: float, y_prime: float):
        return abs(y - y_prime)

    @classmethod
    def _data_penalty(cls, x: pt.Tensor, x_prime: pt.Tensor, p: float):
        diff = (x - x_prime).reshape(-1)
        return float(pt.norm(diff, p=p))

    def value(self, x: pt.Tensor, x_prime: pt.Tensor, y: float, y_prime: float):
        r"""
        Cost to displace :math:`\xi:=\left[\begin{array}{c}\bm{X}\\y\end{array}\right]`
        to :math:`\zeta:=\left[\begin{array}{c}\bm{X'}\\y'\end{array}\right]`
        in :math:`mathbb{R}^n`.

        Parameters
        ----------
        x : Tensor, shape (n_samples, n_features)
            Data point to be displaced (without the label)
        x_prime : Tensor, shape (n_samples, n_features)
            Data point towards which ``x`` is displaced
        y : float
            Label or target for the problem/loss
        y_prime : float
            Label or target in the dataset
        """
        if self.kappa is float("inf"):
            # Writing convention: if kappa=+oo we put all cost on switching labels
            #  so the cost is reported on y.
            # To provide a tractable computation, we yield the y-penalty alone.
            return self._label_penalty(y, y_prime)**self.power
        elif self.kappa == 0.:
            # Writing convention: if kappa is null we put all cost on moving the data itself, so the worst-case distribution is free to switch the labels.
            # Warning : this usecase should not make sense anyway.
            return self._data_penalty(x, x_prime, self.p)**self.power
        else:
            distance = self._data_penalty(x, x_prime, self.p) \
                + self.kappa * self._label_penalty(y, y_prime)
            return distance**self.power
