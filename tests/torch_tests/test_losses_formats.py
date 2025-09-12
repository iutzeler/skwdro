import pytest
from skwdro.torch import robustify
from skwdro.solvers import DualLoss
from skwdro.base.losses_torch import WrappedPrimalLoss
from skwdro.base.samplers.torch import ClassificationNormalIdSampler
from skwdro.base.costs_torch import NormCost

import torch as pt


def parametrize_(fn):
    b_p = pytest.mark.parametrize(
        "b", (True, False)
    )(fn)
    int_p = pytest.mark.parametrize(
        "interface_", (0, 1)
    )(b_p)
    return pytest.mark.parametrize(
        "red", ('mean', 'sum', 'none', None)
    )(int_p)

def fake_data(b: bool = False):
    if b:
        X = pt.randn(100, 3)
        y = pt.rand(100, 3)
    else:
        X = pt.randn(1, 3)
        y = pt.rand(1,3)

    return X, y

def build_model(loss, X, y, interface_: int, red: str|None):
    inference_model = pt.nn.Linear(3, 3, bias=False)
    if interface_ == 0:
        return robustify(
            loss,
            inference_model,
            pt.tensor(1.),
            X, y,
            cost_spec='t-NC-2-2',
            reduction=red
        )
    elif interface_ == 1:
        sampler = ClassificationNormalIdSampler(
            X, y,
            seed=666,
            sigma=.1,
        )
        cost = NormCost(2, 2)
        return DualLoss(
            WrappedPrimalLoss(loss, inference_model, sampler, True),
            cost,
            10,
            pt.tensor(1.),
            pt.tensor(1.),
            1000,
            reduction=red
        )
    else:
        raise ValueError("Wrong value for interface_: " + str(interface_))

@parametrize_
def test_functional(b: bool, interface_: int, red: str|None):
    X, y = fake_data(b)
    loss = pt.nn.functional.binary_cross_entropy_with_logits
    model = build_model(loss, X, y, interface_, red)
    l = model(X, y)
    if red == 'none':
        if b:
            assert l.shape == pt.Size([100])
        else:
            assert l.shape == pt.Size([1])
    else:
        assert l.shape == pt.Size([])

class MyLoss(pt.nn.Module):
    reduction = None
    def __init__(self, reduction, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.l = pt.nn.BCEWithLogitsLoss(reduction=reduction)
        self.reduction = reduction

    def forward(self, x, y):
        return self.l(x, y)

@parametrize_
def test_oop(b: bool, interface_: int, red: str|None):
    X, y = fake_data(b)
    loss = MyLoss(reduction='none')
    model = build_model(loss, X, y, interface_, red)
    l = model(X, y)
    if red == 'none':
        if b:
            assert l.shape == pt.Size([100])
        else:
            assert l.shape == pt.Size([1])
    else:
        assert l.shape == pt.Size([])

    loss = pt.nn.BCEWithLogitsLoss(reduction='sum')
    try:
        model = build_model(loss, X, y, interface_, red)
        l = model(X, y)
    except AssertionError as e:
        assert 'reduction' in e.args[0]
        loss = pt.nn.BCEWithLogitsLoss(reduction='mean')
        try:
            model = build_model(loss, X, y, interface_, red)
            l = model(X, y)
        except AssertionError as e2:
            assert 'reduction' in e2.args[0]
        else:
            raise
    else:
        raise
