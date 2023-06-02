from types import NoneType
from typing import Optional
import torch as pt
import torch.distributions as dst

from skwdro.base.samplers.torch.base_samplers import NoLabelsSampler, LabeledSampler
from skwdro.base.samplers.torch.newsvendor_sampler import NewsVendorNormalSampler
from skwdro.base.samplers.torch.classif_sampler import ClassificationNormalNormalSampler
from skwdro.base.costs import Cost

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
        diff = (xi - zeta).reshape(-1)
        return pt.norm(diff, p=self.p, dim=-1)**self.power

    def _sampler_data(self, xi, epsilon):
        if self.power == 1:
            if self.p == 1:
                return dst.Laplace(
                            loc=xi,
                            scale=epsilon
                        )
            elif self.p == 2:
                return dst.MultivariateNormal(
                        loc=xi,
                        scale_tril=epsilon*pt.eye(xi.size(-1))
                    )
            else: raise NotImplementedError()
        else: raise NotImplementedError()

    def _sampler_labels(self, xi_labels, epsilon):
        return None

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
    def _label_penalty(cls, y: pt.Tensor, y_prime: pt.Tensor, p: float):
        return pt.norm(y - y_prime, p=p, dim=-1)

    @classmethod
    def _data_penalty(cls, x: pt.Tensor, x_prime: pt.Tensor, p: float):
        diff = (x - x_prime).reshape(-1)
        return pt.norm(diff, p=p, dim=-1)

    def value(self, xi: pt.Tensor, zeta: pt.Tensor, xi_labels: pt.Tensor, zeta_labels: pt.Tensor):
        r"""
        Cost to displace :math:`\xi:=\left[\begin{array}{c}\bm{X}\\y\end{array}\right]`
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
        if float(self.kappa) is float("inf"):
            # Writing convention: if kappa=+oo we put all cost on switching labels
            #  so the cost is reported on y.
            # To provide a tractable computation, we yield the y-penalty alone.
            return self._label_penalty(xi_labels, zeta_labels, self.p)**self.power
        elif self.kappa == 0.:
            # Writing convention: if kappa is null we put all cost on moving the data itself, so the worst-case distribution is free to switch the labels.
            # Warning : this usecase should not make sense anyway.
            return self._data_penalty(xi, zeta, self.p)**self.power
        else:
            distance = self._data_penalty(xi, zeta, self.p) \
                + self.kappa * self._label_penalty(xi_labels, zeta_labels, self.p)
            return distance**self.power

    def _sampler_labels(self, xi_labels, epsilon):
        if self.power == 1:
            if self.p == 1:
                return dst.Laplace(
                            loc=xi_labels,
                            scale=epsilon/self.kappa
                        )
            elif self.p == 2:
                return dst.MultivariateNormal(
                        loc=xi_labels,
                        scale_tril=epsilon*pt.eye(xi_labels.size(-1))/self.kappa
                    )
            else: raise NotImplementedError()
        else: raise NotImplementedError()
