import numpy as np

from skwdro.base.costs import Cost

class WDROProblem:
    """ Base class for WDRO problem """

    def __init__(self, cost: Cost, n=0, Theta_bounds=None , d=0, Xi_bounds=None , dLabel=0, XiLabel_bounds=None , costLabel=None, loss=None, rho = 0., P = None,  name="WDRO Problem"):

        ## Optimization variable
        self.n = n # size of Theta
        self.Theta_bounds = Theta_bounds

        ## Uncertain variable
        self.d = d # size of Xi
        self.Xi_bounds = Xi_bounds

        ## Uncertain labels
        self.dLabel = dLabel # size of Xi
        self.XiLabel_bounds = XiLabel_bounds
        self.costLabel = costLabel

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




class EmpiricalDistribution:
    """ Empirical Probability distribution """

    empirical = True

    def __init__(self, m , samples , name="Empirical distribution"):
        self.m = m
        self.samples = samples


class EmpiricalDistributionWithLabels:
    """ Empirical Probability distribution with Labels """

    empirical = True

    def __init__(self, m , samplesX, samplesY , name="Empirical distribution"):
        self.m = m
        self.samplesX = samplesX
        self.samplesY = samplesY


