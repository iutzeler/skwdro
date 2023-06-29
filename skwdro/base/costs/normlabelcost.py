from typing import Optional

import numpy as np

from .normcost import NormCost

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
    def _label_penalty(cls, y: np.ndarray, y_prime: np.ndarray):
        if isinstance(y, int) or isinstance(y, float) or len(y.shape) < 2:
            # Old code for scalar y (non-parallelized)
            return abs(y-y_prime)
        elif y.shape[-1] == 1:
            # d = 1
            return np.abs(y - y_prime)
        else:
            # TODO
            raise NotImplementedError("Multi-dim y not implemented")


    @classmethod
    def _data_penalty(cls, x: np.ndarray, x_prime: np.ndarray, p: float):
        if len(x.shape) > 2:
            diff = x - x_prime
            return np.linalg.norm(diff, ord=p, axis=2, keepdims=True)
        else: return np.linalg.norm(x-x_prime)

    def value(self, x: np.ndarray, x_prime: np.ndarray, y: np.ndarray, y_prime: np.ndarray):
        r"""
        Cost to displace :math:`\xi:=\left[\begin{array}{c}\bm{X}\\y\end{array}\right]`
        to :math:`\zeta:=\left[\begin{array}{c}\bm{X'}\\y'\end{array}\right]`
        in :math:`mathbb{R}^n`.

        Parameters
        ----------
        x : Array, shape (n_samples, m, d)
            Data point to be displaced (without the label)
        x_prime : Array, shape (n_samples, m, d)
            Data point towards which ``x`` is displaced
        y : Array, shape (n_samples, m, 1)
            Label or target for the problem/loss
        y_prime : Array, shape (n_samples, m, 1)

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

            # Rescale to avoid overflows
            distance /= (1. + self.kappa)

            return distance**self.power
