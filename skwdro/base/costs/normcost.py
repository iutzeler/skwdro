import numpy as np

from .base_cost import Cost

class NormCost(Cost):
    """ p-norm to some power """

    def __init__(self, p=1.0, power=1.0, name=None):
        super().__init__(name="Norm" if name is None else name, engine="np")
        self.p = p
        self.power = power

    def value(self,x,y, *_):
        r"""
        Cost to displace :math:`\xi` to :math:`\zeta` in :math:`mathbb{R}^n`.

        Parameters
        ----------
        x : Array, shape (n_samples, n_features, d)
            Data point to be displaced
        y : Array, shape (n_samples, n_features, d)
            Data point towards which ``xi`` is displaced
        """
        if len(x.shape) > 2:
            return self._parallel_value(x, y)
        diff = np.array(x-y).flatten()
        return np.linalg.norm(diff,ord=self.p)**self.power

    def _parallel_value(self, x, y):
        return np.linalg.norm(x - y, axis=2, ord=self.p, keepdims=True) ** self.power
