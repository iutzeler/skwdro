import torch
import torch.nn as nn

from abc import ABC, abstractmethod

class ApproxDRO(ABC):
    def __init__(self, f, rho, n, eps, sigma, loss=nn.MSELoss(reduction='none'), p=2, normalize=1):
        super(ApproxDRO, self).__init__()
        assert p in (1, 2)
        self.n = n
        self.model = f
        self.d_in = f.in_features
        self.d = self.d_in + 1
        self.rho = rho
        self.loss_func = loss
        self.eps = eps
        self.distr = torch.distributions.Normal(0, sigma) if p == 2 else torch.distributions.Laplace(0, sigma)
        self.sigma = sigma
        self.normalize = normalize
        self.p = p
    
    def cost(self, diff_samples, dim=None):
        return (1/2**(self.p-1)) * torch.linalg.vector_norm(diff_samples, dim=dim)**self.p
    
    def f(self, x, y):
        b = y.size()
        assert x.size() == b + (self.d_in,)

        loss = self.model(x, y)
        assert loss.size() == b

        # pred = self.model(x).squeeze(-1)
        # assert pred.size() == y.size(), (pred.size(), y.size())

        # return self.normalize*self.loss_func(pred, y)
        return loss

    @abstractmethod
    def forward(self, indices, x, y):
        pass

    @property
    def upper_bound_integral(self):
        """ Return an upper-bound on
            E[c(x, Y)] when Y ~ Pi_0(.|x) 
        """
        return 1/2**(self.p - 1)*self.sigma**self.p

    def is_feasible(self):
        """ If returns true, then the transport plan Pi_0 is strictly feasible """
        return self.upper_bound_integral < self.rho
