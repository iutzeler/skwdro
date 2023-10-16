from typing import List, Optional, Union
import numpy as np

from skwdro.base.costs import Cost
from skwdro.base.costs_torch import Cost as TorchCost
from skwdro.base.losses import Loss as LossNumpy
from skwdro.solvers.oracle_torch import _DualLoss as LossTorch
from skwdro.solvers.optim_cond import OptCondTorch as OptCond

Bounds = Optional[List[float]]
LossType = Union[LossNumpy, LossTorch]

class Distribution:
    empirical: bool
    with_labels: bool
    def __init__(self, m: int, name: str) -> None:
        self.m = m
        self.name = name
        self._samples   = None
        self._samples_x = None
        self._samples_y = None

    @property
    def samples(self):
        if self.with_labels:
            raise AttributeError()
        else:
            return self._samples

    @property
    def samples_y(self):
        if self.with_labels:
            return self._samples_y
        else:
            raise AttributeError()

    @property
    def samples_x(self):
        if self.with_labels:
            return self._samples_x
        else:
            raise AttributeError()

    @samples.setter
    def samples(self, data):
        if isinstance(data, np.ndarray):
            self._samples = data
        else:
            raise TypeError()

    @samples_x.setter
    def samples_x(self, data):
        if isinstance(data, np.ndarray):
            self._samples_x = data
        else:
            raise TypeError()

    @samples_y.setter
    def samples_y(self, labels):
        if isinstance(labels, np.ndarray):
            self._samples_y = labels
        else:
            raise TypeError()


class WDROProblem:
    """ Base class for WDRO problem """

    def __init__(
            self,
            cost: Cost|TorchCost,
            loss: LossType,
            p_hat: Distribution,
            n: int=0,
            d: int=0,
            d_labels: int=0,
            theta_bounds: Bounds=None,
            xi_bounds: Bounds=None,
            xi_labels_bounds: Bounds=None,
            rho: float=0.,
            *,
            order_stop: int=2,
            tol_theta_stop: float=1e-6,
            tol_lambda_stop: float=1e-10,
            monitoring_stop: str="t&l",
            mode_stop: str="rel",
            metric_stop: str="param",
            name="WDRO Problem"):

        ## Optimization variable
        self.n = n # size of Theta
        self.theta_bounds = theta_bounds

        ## Uncertain variable
        self.d = d # size of Xi
        self.xi_bounds = xi_bounds

        ## Uncertain labels
        self.d_label = d_labels # size of Xi
        self.xi_labels_bounds = xi_labels_bounds

        ## Problem loss
        self.loss = loss

        ## Radius
        self.rho = rho

        ## Transport cost
        self.c = cost.value

        ## Base distribution
        self.p_hat = p_hat

        ## Problem name
        self.name = name

        ## Optimality conditions
        self.opt_cond = OptCond(
                order_stop,
                tol_theta_stop,
                tol_lambda_stop,
                monitoring=monitoring_stop,
                mode=mode_stop,
                metric=metric_stop
            )


class EmpiricalDistributionWithoutLabels(Distribution):
    """ Empirical Probability distribution """

    empirical = True
    with_labels = False

    def __init__(self, m: int, samples: np.ndarray, name="Empirical distribution"):
        super(EmpiricalDistributionWithoutLabels, self).__init__(m, name)
        self._samples = samples


class EmpiricalDistributionWithLabels(Distribution):
    """ Empirical Probability distribution with Labels """

    empirical = True
    with_labels = True

    def __init__(self, m: int, samples_x: np.ndarray, samples_y: np.ndarray, name="Empirical distribution"):
        super(EmpiricalDistributionWithLabels, self).__init__(m, name)
        self._samples_x = samples_x.copy('K')
        self._samples_y = samples_y.copy('K')
