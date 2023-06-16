import torch as pt
import torch.nn as nn 
import torch.distributions as dst
import numpy as np
from typing import Optional

from skwdro.base.losses_torch import Loss
from skwdro.base.costs_torch import NormCost
from skwdro.base.samplers.torch.base_samplers import NoLabelsSampler
from skwdro.base.samplers.torch.cost_samplers import NoLabelsCostSampler
from skwdro.base.samplers.torch.portfolio_sampler import PortfolioNormalSampler

class RiskPortfolioLoss_torch(Loss):

    def __init__(self, cost: Optional[NormCost]=None, *, xi, epsilon, sampler: Optional[NoLabelsSampler]=None, m, \
                reparam="softmax", name="Portfolio Torch Module Risk loss"):
        super(RiskPortfolioLoss_torch, self).__init__(sampler)
        self._theta_tilde = nn.Parameter(pt.tensor([[0.2 for _ in range(m)]])) #Useful for reparametrization
        self._theta = pt.tensor(0.1)
        self.reparam = reparam
        self.name = name
        self.sampler = NoLabelsCostSampler(cost,xi.float(),epsilon)

    def value(self, xi, xi_labels=None):
        if isinstance(xi, (np.ndarray,np.generic)):
            xi = pt.from_numpy(xi)
        if self.reparam == "softmax":
            softmax = nn.Softmax(dim=1)
            self._theta = softmax(self._theta_tilde)
        elif self.reparam == "softplus":
            softplus = nn.Softplus()
            softplus_theta_tilde = nn.functional.softplus(self._theta_tilde)
            self._theta = (softplus(self._theta_tilde)/pt.sum(softplus_theta_tilde))
        elif self.reparam == "none":
            self._theta = self._theta_tilde
        else:
            raise ValueError("Reparametrization function not recognized")
        return -nn.functional.linear(input=xi.type(pt.FloatTensor), weight=self._theta, bias=None)
    
    @property
    def theta(self):
        return self._theta
    
    @property
    def theta_tilde(self):
        return self._theta_tilde
    
    @property
    def intercept(self):
        return None

    @classmethod
    def default_sampler(cls, xi, xi_labels, epsilon):
        return PortfolioNormalSampler(xi, sigma=epsilon)
    
class MeanRisk_torch(Loss):
    def __init__(self, sampler: Optional[NoLabelsSampler]=None, \
                *, loss: Loss, eta:pt.Tensor, alpha:pt.Tensor, \
                name = "Mean-Risk Portfolio Torch Module General loss"):
        super(MeanRisk_torch, self).__init__(sampler)
        self.loss = loss
        self.eta = nn.Parameter(eta, requires_grad=False)
        self.alpha = nn.Parameter(alpha, requires_grad=False)
        self._tau = nn.Parameter(pt.tensor(0.0))
        self.name = name
        self.sampler = loss.sampler

    def value(self, xi, xi_labels=None):
        if isinstance(xi, (np.ndarray,np.generic)):
            xi = pt.from_numpy(xi)
        f_theta = self.loss.value(xi)
        relu = nn.ReLU()
        positive_part = relu(f_theta - self._tau)
        return f_theta + self.eta*self._tau + (self.eta/self.alpha)*positive_part
    
    @classmethod
    def default_sampler(cls, xi, xi_labels, epsilon):
        #return PortfolioLaplaceSampler(xi, sigma=epsilon)
        return PortfolioNormalSampler(xi, sigma=epsilon)
    
    @property
    def theta(self):
        return self.loss.theta
    
    @property
    def tau(self):
        return self._tau

    @tau.setter
    def tau(self, val):
        self._tau = val
    
    @property
    def theta_tau(self):
        return self._theta_tau
    
    @property
    def intercept(self):
        return None


    
        
