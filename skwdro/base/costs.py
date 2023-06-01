import numpy as np
from typing import Optional

ENGINES_NAMES = {
    "pt": "PyTorch tensors",
    "np": "Numpy arrays",
    "jx": "Jax arrays"
}


ENGINES_NAMES = {
    "pt": "PyTorch tensors",
    "np": "Numpy arrays",
    "jx": "Jax arrays"
}

class Cost:
    """ Base class for transport functions """

    def __init__(self, name: str="", engine: str=""):
        self.name = name
        self.engine = engine

    def value(self, x, y):
        raise NotImplementedError("Please Implement this method")

    def __str__(self) -> str:
        return "Cost named " + self.name + " using as data: " + ENGINES_NAMES[self.engine]

    def sampler(self, xi, xi_labels, epsilon):
        return self._sampler_data(xi, epsilon), self._sampler_labels(xi_labels, epsilon)

    def _sampler_data(self, xi, epsilon):
        raise NotImplementedError()

    def _sampler_labels(self, xi_labels, epsilon):
        raise NotImplementedError()


class NormCost(Cost):
    """ p-norm to some power """

    def __init__(self, p=1.0, power=1.0, name=None):
        super().__init__(name="Norm" if name is None else name, engine="np")
        self.p = p
        self.power = power

    def value(self,x,y):
        r"""
        Cost to displace :math:`\xi` to :math:`\zeta` in :math:`mathbb{R}^n`.

        Parameters
        ----------
        x : Array, shape (n_samples, n_features)
            Data point to be displaced
        y : Array, shape (n_samples, n_features)
            Data point towards which ``xi`` is displaced
        """
        diff = np.array(x-y).flatten()
        return np.linalg.norm(diff,ord=self.p)**self.power



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
        return np.abs(y - y_prime)

    @classmethod
    def _data_penalty(cls, x: np.ndarray, x_prime: np.ndarray, p: float):
        diff = (x - x_prime).flatten()
        return float(np.linalg.norm(diff, ord=p))

    def value(self, x: np.ndarray, x_prime: np.ndarray, y: float, y_prime: float):
        r"""
        Cost to displace :math:`\xi:=\left[\begin{array}{c}\bm{X}\\y\end{array}\right]`
        to :math:`\zeta:=\left[\begin{array}{c}\bm{X'}\\y'\end{array}\right]`
        in :math:`mathbb{R}^n`.

        Parameters
        ----------
        x : Array, shape (n_samples, n_features)
            Data point to be displaced (without the label)
        x_prime : Array, shape (n_samples, n_features)
            Data point towards which ``x`` is displaced
        y : Array, shape (n_samples,)
            Label or target for the problem/loss
        y_prime : Array, shape (n_samples,)
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
