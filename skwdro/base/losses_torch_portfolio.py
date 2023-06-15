import torch as pt
import torch.nn as nn 
import numpy as np

from skwdro.base.losses_torch import Loss
from skwdro.base.samplers.torch.portfolio_sampler import PortfolioNormalSampler

class RiskPortfolioLoss_torch(Loss):

    def __init__(self, m, reparam="softmax", name="Portfolio Torch Module Risk loss"):
        super(RiskPortfolioLoss_torch, self).__init__()
        self._theta_tilde = nn.Parameter(pt.tensor([[0.2 for _ in range(m)]]))
        self._theta = pt.tensor(0.0) #Default value for the initialization
        self.reparam = reparam
        self.name = name

    def value(self, X):
        if self.reparam == "softmax":
            softmax = nn.Softmax(dim=1)
            self._theta = softmax(self._theta_tilde)
        elif self.reparam == "softplus":
            softplus = nn.Softplus()
            softplus_theta_tilde = nn.functional.softplus(self._theta_tilde)
            self._theta = (softplus(self._theta_tilde)/pt.sum(softplus_theta_tilde))
        elif self.reparam == "none":
            print("No reparametrization")
        else:
            raise ValueError("Reparametrization function not recognized")
        return -nn.functional.linear(input=X.type(pt.FloatTensor), weight=self._theta, bias=None)
    
    @property
    def theta(self):
        return self._theta
    
    @property
    def intercept(self):
        return None
    
class MeanRisk_torch(Loss):
    def __init__(self, loss: Loss, eta:pt.Tensor, alpha:pt.Tensor, \
                name = "Mean-Risk Portfolio Torch Module General loss"):
        super(MeanRisk_torch, self).__init__()
        self.loss = loss
        self.eta = nn.Parameter(eta, requires_grad=False)
        self.alpha = nn.Parameter(alpha, requires_grad=False)
        self._tau = nn.Parameter(pt.tensor(0.0))
        self.name = name

    def value(self, X, X_labels=None):
        f_theta = self.loss.value(X)
        relu = nn.ReLU()
        positive_part = relu(f_theta - self._tau)
        return f_theta + self.eta*self._tau + (self.eta/self.alpha)*positive_part
    
    @classmethod
    def default_sampler(cls, xi, xi_labels, epsilon):
        return PortfolioNormalSampler(xi, sigma=epsilon)
    
    @property
    def theta(self):
        return self.loss.theta
    
    @property
    def tau(self):
        return self._tau
    
    @property
    def intercept(self):
        return None


    
        
