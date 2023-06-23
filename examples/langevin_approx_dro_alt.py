import torch
import torch.nn as nn
import math
import logging

from functools import partial

from biased_approx_dro import BiasedApproxDRO

def resample(log_pi, samples, logprobs, n_resample):
    b = samples.size(0)
    d = samples.size(-1)
    assert logprobs.size() == samples.size()[:-1]

    resized_samples = samples.view(b, -1, d)
    resized_logprobs = logprobs.view(b, -1)

    _, adv_b = resized_logprobs.size()
    assert resized_samples.size() == (b, adv_b, d)
    assert resized_logprobs.size() == (b, adv_b)

    new_logprobs = log_pi(resized_samples) - resized_logprobs
    assert new_logprobs.size() == (b, adv_b)

    distr = torch.distributions.Categorical(logits=new_logprobs, validate_args=True)
    assert distr.batch_shape == (b,)
    assert distr.event_shape == torch.Size([])

    new_samples_indices = distr.sample(sample_shape=(n_resample,))
    assert new_samples_indices.size() == (n_resample, b)

    new_samples_indices = torch.permute(new_samples_indices, (1, 0))
    assert new_samples_indices.size() == (b, n_resample)

    expanded_new_samples_indices = new_samples_indices.unsqueeze(-1).expand(b, n_resample, d)
    assert expanded_new_samples_indices.size() == (b, n_resample, d)

    new_samples = torch.gather(resized_samples, 1, expanded_new_samples_indices)
    assert new_samples.size() == (b, n_resample, d)

    return new_samples

def stoch_langevin_dynamic(log_pi, sigma, lr, T, m, init, bound_grad=100):
    b = init.size()[:-1]
    d = init.size(-1)
    x = torch.clone(init)
    samples = torch.empty(b + (m, d))

    noise = torch.distributions.Normal(0, sigma)

    for i in range(T):
        x.requires_grad_(True)
        grad, = torch.autograd.grad(log_pi(x), x, grad_outputs=torch.ones(b))
        assert grad.size() == x.size()

        with torch.no_grad():
            bounded_grad = bound_grad / max(bound_grad, torch.linalg.norm(grad)) * grad
            x = x + lr * bounded_grad + math.sqrt(2*lr) * noise.sample(x.size())
            if i >= T - m:
                samples[...,i - (T - m),:] = x
    assert not samples.requires_grad
    return samples

class LangevinApproxDRO(BiasedApproxDRO, nn.Module):
    def __init__(self, f, rho, n, eps, sigma, T=10, m_train=10, m_eval=1000, sigma_langevin=None, lr=0.1, m_parallel=10, **kwargs):
        super(LangevinApproxDRO, self).__init__(f, rho, n, eps, sigma, m_eval, **kwargs)
        self.lbd = nn.Parameter(torch.tensor([10.])) #nn.Parameter(10*abs(torch.randn(1)))
        self.T = T
        self.sigma_langevin = sigma if sigma_langevin is None else sigma_langevin
        self.lr = lr
        self.m_train = m_train
        self.m_parallel = m_parallel
        self.prev_adv_xi = None
        self.prev_adv_xi_logprobs = None
    
    def _langevin_log_pi(self, xi, adv_xi):
        for param in self.parameters():
            assert not param.requires_grad

        b, _ = xi.size()
        assert     xi.size() == (b, self.d)
        adv_b = adv_xi.size()[1:-1]
        assert adv_xi.size() == (b,) + adv_b + (self.d,), (adv_xi.size(), (b, adv_b, self.d))

        expanded_xi = xi.view(b, *[1 for _ in range(len(adv_b))], self.d)

        diff_xi = adv_xi - expanded_xi
        assert diff_xi.size() == (b,) + adv_b + (self.d,)

        cost = self.cost(adv_xi - expanded_xi, dim=-1)
        assert cost.size() == (b,) + adv_b

        adv_loss = self.f(adv_xi[...,:-1], adv_xi[...,-1])
        assert adv_loss.size() == (b,) + adv_b

        res = (adv_loss - (self.lbd + self.eps/self.sigma**2)*cost)/self.eps

        return res
    
    def sample_adv(self, x, y):
        assert self.training

        b, _ = x.size()
        assert x.size() == (b, self.d_in)
        assert y.size() == (b,)

        for param in self.parameters():
            param.requires_grad = False
        
        xi = torch.cat((x, y.unsqueeze(-1)), dim=-1)
        assert xi.size() == (b, self.d)

        log_pi = partial(self._langevin_log_pi, xi)

        if self.prev_adv_xi is None:
            init = xi.unsqueeze(1).expand((b, self.m_parallel, self.d))
        else:
            assert self.prev_adv_xi.size() == (b, self.m_parallel, self.m_train, self.d)
            assert self.prev_adv_xi_logprobs.size() == (b, self.m_parallel, self.m_train)
            init = resample(log_pi, self.prev_adv_xi, self.prev_adv_xi_logprobs, self.m_parallel)

        assert init.size() == (b, self.m_parallel, self.d)

        adv_xi = stoch_langevin_dynamic(log_pi, self.sigma_langevin, self.lr, self.T, self.m_train, init)
        assert adv_xi.size() == (b, self.m_parallel, self.m_train, self.d)

        self.prev_adv_xi = adv_xi
        self.prev_adv_xi_logprobs = log_pi(self.prev_adv_xi)
        assert self.prev_adv_xi.size() == (b, self.m_parallel, self.m_train, self.d)
        assert self.prev_adv_xi_logprobs.size() == (b, self.m_parallel, self.m_train)

        for param in self.parameters():
            param.requires_grad = True

        cost = self.cost(adv_xi - xi[:,None,None,:], dim=-1)
        assert cost.size() == (b, self.m_parallel, self.m_train)

        adv_loss = self.f(adv_xi[...,:-1], adv_xi[...,-1])
        assert adv_loss.size() == (b, self.m_parallel, self.m_train), adv_loss.size()

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
    
    def forward(self, x, y):
        assert self.training

        b, = y.size()
        assert x.size() == (b, self.d_in)
        assert y.size() == (b,)

        sampled = self.sample_adv(x, y)
        
        res = self.forward_det(x, y, sampled)

        logging.debug(f"{res.item()=} {self.lbd.item()=}")

        return res
