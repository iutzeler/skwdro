import torch
import torch.nn as nn
import pytest

import hybrid_sgd

class DummyModule(nn.Module):
	def __init__(self, a, b):
		super(DummyModule, self).__init__()
		self.a = nn.Parameter(a)
		self.b = nn.Parameter(b)
	
	def forward(self):
		return torch.linalg.norm(self.a)**2 + torch.linalg.norm(self.b)**2

@pytest.fixture
def model():
	return DummyModule(torch.rand(1), torch.rand(1))

@pytest.fixture
def lr():
	return 1.5*torch.rand(1).item()

lr_a = lr
lr_b = lr

@pytest.fixture
def optimizer(model, lr_a, lr_b):
	return hybrid_sgd.HybridSGD([
		{'params': [model.a], 'lr':lr_a, 'mwu':True},
		{'params': [model.b], 'lr':lr_b, 'non_neg':True}
		], lr=lr_a)

def test_hybrid_sgd(model, optimizer, lr_a, lr_b):
	optimizer.zero_grad()

	loss = model.forward()
	loss.backward()

	with torch.no_grad():
		exp_a = model.a * torch.exp(-lr_a*model.a.grad)
		exp_b = torch.clip(model.b - lr_b*model.b.grad, 0, None)
	
	optimizer.step()

	assert exp_a == pytest.approx(model.a.item())
	assert exp_b == pytest.approx(model.b.item())
	
	optimizer.zero_grad()

	assert model.a.grad.item() == 0.
	assert model.b.grad.item() == 0.





