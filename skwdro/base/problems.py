from typing import List, Optional, Union
import numpy as np

from skwdro.base.costs import Cost
from skwdro.base.losses import Loss as LossNumpy
from skwdro.solvers.oracle_torch import _DualLoss as LossTorch

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
            cost: Cost,
            loss: LossType,
            P: Distribution,
            n: int=0,
            Theta_bounds: Bounds=None,
            d: int=0,
            Xi_bounds: Bounds=None,
            dLabel: int=0,
            XiLabel_bounds: Bounds=None,
            rho: float=0.,
            name="WDRO Problem"):

        ## Optimization variable
        self.n = n # size of Theta
        self.Theta_bounds = Theta_bounds

        ## Uncertain variable
        self.d = d # size of Xi
        self.Xi_bounds = Xi_bounds

        ## Uncertain labels
        self.dLabel = dLabel # size of Xi
        self.XiLabel_bounds = XiLabel_bounds

        ## Problem loss
        self.loss = loss

        ## Radius
        self.rho = rho

        ## Transport cost
        self.c = cost.value

        ## Base distribution
        self.P = P

        ## Problem name
        self.name = name



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
        self._samples_x = samples_x
        self._samples_y = samples_y
