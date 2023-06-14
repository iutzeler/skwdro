import torch
import torch.nn as nn
import pytest

import skwdro
import skwdro.solvers.hybrid_sgd as hybrid_sgd

class DummyModule(nn.Module):
    def __init__(self, a, b, c):
        super(DummyModule, self).__init__()
        self.a = nn.Parameter(a)
        self.b = nn.Parameter(b)
        self.c = nn.Parameter(c)
    
    def forward(self):
        return 0.5*torch.linalg.norm(self.a)**2 + 0.5*torch.linalg.norm(self.b)**2 + 0.5*torch.linalg.norm(self.c)**2

@pytest.fixture
def model():
    return DummyModule(torch.rand(1), torch.rand(1), torch.rand(2))

@pytest.fixture
def lr():
    return 1.5*torch.rand(1).item()

lr_a = lr
lr_b = lr
lr_c = lr

@pytest.fixture
def optimizer(model, lr_a, lr_b, lr_c):
    return hybrid_sgd.HybridSGD([
        {'params': [model.a], 'lr':lr_a, 'mwu':True},
        {'params': [model.b], 'lr':lr_b, 'non_neg':True},
        {'params': [model.c], 'lr':lr_c, 'mwu_simplex':True}
        ], lr=lr_a)

def test_hybrid_sgd(model, optimizer, lr_a, lr_b, lr_c):
    optimizer.zero_grad()

    loss = model.forward()
    loss.backward()

    assert torch.isclose(model.a.grad, model.a).all()
    assert torch.isclose(model.b.grad, model.b).all()
    assert torch.isclose(model.c.grad, model.c).all()

    with torch.no_grad():
        exp_a = model.a * torch.exp(-lr_a*model.a.grad)
        exp_b = torch.clip(model.b - lr_b*model.b.grad, 0, None)
        exp_c = model.c * torch.exp(-lr_c*model.c.grad)
        exp_c /= torch.sum(exp_c)
    
    optimizer.step()

    assert exp_a == pytest.approx(model.a.item())
    assert exp_b == pytest.approx(model.b.item())
    for i in range(exp_c.size(0)):
        assert exp_c[i] == pytest.approx(model.c[i].item())





