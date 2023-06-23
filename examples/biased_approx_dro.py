import torch
import torch.nn as nn
import math
import logging

try:
	from .approx_dro import ApproxDRO
except ImportError:
	from approx_dro import ApproxDRO

class BiasedApproxDRO(ApproxDRO, nn.Module):
	def __init__(self, f, rho, n, eps, sigma, m, sample_y=True, **kwargs):
		super(BiasedApproxDRO, self).__init__(f, rho, n, eps, sigma, **kwargs)
		self.lbd = nn.Parameter(10*abs(torch.randn(1)))
		self.m = m
		self.sample_y = sample_y
	
	def sample_adv(self, x, y):
		b, = y.size()
		assert x.size() == (b, self.d_in)

		samples = self.distr.sample(sample_shape=(b, self.m, self.d))
		adv_x = x[:,None,:] + samples[:,:,:-1]
		adv_y = y[:,None  ] + float(self.sample_y)*samples[:,:, -1]
		cost = self.cost(samples, dim=-1)
		assert cost.size() == (b, self.m)

		adv_loss = self.f(adv_x, adv_y)
		assert adv_loss.size() == (b, self.m), adv_loss.size()

		return adv_loss, cost
	
	def forward(self, *args):
		x, y = args[-2:]
		b, _ = x.size()
		assert x.size() == (b, self.d_in)
		assert y.size() == (b,)

		adv_loss, cost = self.sample_adv(x, y)

		arg_exp = (adv_loss - (self.lbd)*cost)/self.eps

		log_expectation = torch.logsumexp(arg_exp, dim=-1) - math.log(self.m)

		res = (self.lbd) * self.rho \
				+ self.eps * (torch.mean(log_expectation))

		logging.debug(f"{res.item()=} {self.lbd.item()=}")

		return res

class RegBiasedApproxDRO(BiasedApproxDRO, nn.Module):
	def __init__(self, f, rho, n, eps, sigma, m, **kwargs):
		super(RegBiasedApproxDRO, self).__init__(f, rho, n, eps, sigma, m, **kwargs)
	
	def forward(self, *args):
		x, y = args[-2:]
		b, _ = x.size()
		assert x.size() == (b, self.d_in)
		assert y.size() == (b,)

		adv_loss, cost = self.sample_adv(x, y)

		arg_exp = (adv_loss/self.lbd - cost)/self.eps

		log_expectation = torch.logsumexp(arg_exp, dim=-1) - math.log(self.m)

		res = self.lbd * self.rho \
				+ self.eps * self.lbd * (torch.mean(log_expectation))

		logging.debug(f"{res.item()=} {self.lbd.item()=}")

		return res
