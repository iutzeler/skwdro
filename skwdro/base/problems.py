from typing import List, Optional, Union
import numpy as np

from skwdro.base.costs import Cost
from skwdro.base.losses import Loss as LossNumpy
from skwdro.solvers.oracle_torch import _DualLoss as LossTorch

Bounds = Optional[List[float]]
LossType = Union[LossNumpy, LossTorch]

class Distribution:
    empirical: bool
    def __init__(self, m: int, name: str) -> None:
        self.m = m
        self.name = name

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



class EmpiricalDistribution(Distribution):
    """ Empirical Probability distribution """

    empirical = True

    def __init__(self, m: int, samples: np.ndarray, name="Empirical distribution"):
        super(EmpiricalDistribution, self).__init__(m, name)
        self.samples = samples


class EmpiricalDistributionWithLabels(Distribution):
    """ Empirical Probability distribution with Labels """

    empirical = True

    def __init__(self, m, samplesX: np.ndarray, samplesY: np.ndarray, name="Empirical distribution"):
        super(EmpiricalDistributionWithLabels, self).__init__(m, name)
        self.samplesX = samplesX
        self.samplesY = samplesY


