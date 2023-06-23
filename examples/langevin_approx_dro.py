import torch
import torch.nn as nn
import math
import logging

from functools import partial

from biased_approx_dro import BiasedApproxDRO

def stoch_langevin_dynamic(log_pi, sigma, lr, T, m, init, bound_grad=100):
    b, d = init.size()
    x = torch.clone(init)
    samples = torch.empty((b, m, d))

    noise = torch.distributions.Normal(0, sigma)

    for i in range(T):
        x.requires_grad_(True)
        grad, = torch.autograd.grad(log_pi(x), x, grad_outputs=torch.ones(x.size(0)))
        assert grad.size() == x.size()

        with torch.no_grad():
            bounded_grad = bound_grad / max(bound_grad, torch.linalg.norm(grad)) * grad
            x = x + lr * bounded_grad + math.sqrt(2*lr) * noise.sample(x.size())
            if i >= T - m:
                samples[:,i - (T - m),:] = x
    assert not samples.requires_grad
    return samples

class LangevinApproxDRO(BiasedApproxDRO, nn.Module):
    def __init__(self, f, rho, n, eps, sigma, T=1000, m_train=100, m_eval=1000, sigma_langevin=None, lr=0.1, **kwargs):
        super(LangevinApproxDRO, self).__init__(f, rho, n, eps, sigma, m_eval, **kwargs)
        self.lbd = nn.Parameter(torch.tensor([10.])) #nn.Parameter(10*abs(torch.randn(1)))
        self.T = T
        self.sigma_langevin = sigma if sigma_langevin is None else sigma_langevin
        self.lr = lr
        self.m_train = m_train
    
    def _langevin_log_pi(self, xi, adv_xi):
        assert not self.lbd.requires_grad

        b, _ = xi.size()
        assert     xi.size() == (b, self.d)
        assert adv_xi.size() == (b, self.d)

        cost = self.cost(adv_xi - xi, dim=-1)
        assert cost.size() == (b,)

        adv_loss = self.f(adv_xi[...,:-1], adv_xi[...,-1])
        assert adv_loss.size() == (b,), adv_loss.size()

        res = (adv_loss - (self.lbd + self.eps/self.sigma**2)*cost)/self.eps

        return res
    
    def sample_adv(self, x, y, reset=True):
        if not self.training:
            return super(LangevinApproxDRO, self).sample_adv(x, y)

        b, _ = x.size()
        assert x.size() == (b, self.d_in)
        assert y.size() == (b,)

        assert self.training
        for param in self.parameters():
            param.requires_grad = False
        
        xi = torch.cat((x, y.unsqueeze(-1)), dim=-1)
        log_pi = partial(self._langevin_log_pi, xi)

        if reset:
            self.adv_xi = stoch_langevin_dynamic(log_pi, self.sigma_langevin, self.lr, self.T, self.m_train, xi)
            assert self.adv_xi.size() == (b, self.m_train, self.d)

        assert hasattr(self, "adv_xi")
        adv_xi = self.adv_xi

        for param in self.parameters():
            param.requires_grad = True

        cost = self.cost(adv_xi - xi[:,None,:], dim=-1)
        assert cost.size() == (b, self.m_train)

        adv_loss = self.f(adv_xi[...,:-1], adv_xi[...,-1])
        assert adv_loss.size() == (b, self.m_train), adv_loss.size()

        return adv_loss, cost

    def forward_det(self, x, y, sampled):
        adv_loss, cost = sampled

        b, = y.size()
        assert x.size() == (b, self.d_in)
        assert y.size() == (b,)

        arg_exp = adv_loss - self.lbd*cost

        res = self.lbd * self.rho \
                + torch.mean(arg_exp)

        return res

    
    def forward(self, x, y, reset=True):
        if not self.training:
            assert False
            # return super(LangevinApproxDRO, self).forward(indices, x, y)

        #b, = indices.size()
        b, = y.size()
        assert x.size() == (b, self.d_in)
        assert y.size() == (b,)

        sampled = self.sample_adv(x, y, reset=reset)
        
        res = self.forward_det(x, y, sampled)

        logging.debug(f"{res.item()=} {self.lbd.item()=}")

        return res
