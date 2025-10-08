import pytest
from skwdro.base.losses_torch.wrapper import WrappingError, WrappedPrimalLoss
from skwdro.torch import robustify
from skwdro.solvers import DualLoss
from skwdro.base.samplers.torch import ClassificationNormalIdSampler, NoLabelsCostSampler
from skwdro.base.costs_torch import NormCost

import torch as pt


def parametrize_(fn):
    b_p = pytest.mark.parametrize(
        "b", (True, False)
    )(fn)
    int_p = pytest.mark.parametrize(
        "interface_", (0, 1)
    )(b_p)
    r_p = pytest.mark.parametrize(
        "red", ('mean', 'sum', 'none', None)
    )(int_p)
    return pytest.mark.parametrize(
        "lb", (True, False)
    )(r_p)

def fake_data(b: bool = False):
    if b:
        X = pt.randn(100, 3)
        y = pt.rand(100, 3)
    else:
        X = pt.randn(1, 3)
        y = pt.rand(1,3)

    return X, y

def assert_no_sampler(dual_loss):
    with pytest.raises(WrappingError) as e_err:
        dual_loss.primal_loss.default_sampler(
            None, None, None, None
        )
    assert "No default" in str(e_err.value)

def assert_has_theta(dual_loss):
    assert isinstance(dual_loss.primal_loss.theta, pt.Tensor)

def build_model(loss, X, y, interface_: int, red: str|None, lb: bool):
    inference_model = pt.nn.Linear(3, 3, bias=False)
    if interface_ == 0:
        return robustify(
            loss,
            inference_model,
            pt.tensor(1.),
            X, y,
            cost_spec='t-NC-2-2',
            reduction=red,
            loss_reduces_spatial_dims=not lb
        )
    elif interface_ == 1:
        cost = NormCost(2, 2)
        if lb:
            sampler = ClassificationNormalIdSampler(
                X, y,
                seed=666,
                sigma=.1,
            )
        else:
            sampler = NoLabelsCostSampler(
                cost, X, .1
            )
        return DualLoss(
            WrappedPrimalLoss(loss, inference_model, sampler, lb, lb),
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
def test_functional(b: bool, interface_: int, red: str|None, lb: bool):
    X, y = fake_data(b)
    if lb:
        loss = pt.nn.functional.binary_cross_entropy_with_logits
    else:
        def _loss(input, reduction='sum', *args, **kwargs):
            del reduction, args, kwargs
            return pt.sum(input**2, dim=-1).unsqueeze(-1)
        loss = _loss  # trick for type inference (too invasive with the def kw)
        y = None
    model = build_model(loss, X, y, interface_, red, lb)
    assert_has_theta(model)
    assert_no_sampler(model)
    l = model(X, y)
    if red == 'none':
        if b:
            assert l.shape == pt.Size([100])
        else:
            assert l.shape == pt.Size([1])
        assert_weird_input_shapes(model.primal_loss, X, y, b, lb)
    else:
        assert l.shape == pt.Size([])

class MyClassifLoss(pt.nn.Module):
    reduction = None
    def __init__(self, reduction, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.l = pt.nn.BCEWithLogitsLoss(reduction=reduction)
        self.reduction = reduction

    def forward(self, x, y):
        return self.l(x, y)

class MyLikLoss(pt.nn.Module):
    reduction = None
    def __init__(self, reduction, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.reduction = reduction

    def forward(self, x):
        return pt.sum(x**2, dim=-1).unsqueeze(-1)


def assert_weird_input_shapes(model, X, y, b, lb: bool):
    # red always to none
    if b:
        # (1, 100, 3), ?(100, 3)
        print("(1, 100, 3), ?(100, 3)", X.unsqueeze(0).shape, y.shape if y is not None else ' - ')
        l = model(X.unsqueeze(0), y)
        # (1, 100, 1)
        assert l.shape == pt.Size([1, 100, 1])
    elif lb:
        assert y is not None
        # (3,), (3,)
        print("(3,), (3,)", X.squeeze(0).shape, y.squeeze(0).shape)
        l = model(X.squeeze(0), y.squeeze(0))
        # (,)
        assert l.shape == pt.Size([])
    else:
        # (3,)
        print("(3,)", X.squeeze(0).shape, ' - ')
        l = model(X.squeeze(0), None)
        # (,)
        assert l.shape == pt.Size([])

@parametrize_
def test_oop(b: bool, interface_: int, red: str|None, lb: bool):
    X, y = fake_data(b)
    if lb:
        loss = MyClassifLoss(reduction='none')
    else:
        loss = MyLikLoss(reduction='none')
        y = None
    model = build_model(loss, X, y, interface_, red, lb)
    assert_has_theta(model)
    assert_no_sampler(model)
    l = model(X, y)
    if red == 'none':
        if b:
            assert l.shape == pt.Size([100])
        else:
            assert l.shape == pt.Size([1])
        assert_weird_input_shapes(model.primal_loss, X, y, b, lb)
    else:
        assert l.shape == pt.Size([])

    loss = pt.nn.BCEWithLogitsLoss(reduction='sum')
    try:
        model = build_model(loss, X, y, interface_, red, lb)
        l = model(X, y)
    except AssertionError as e:
        assert 'reduction' in e.args[0]
        loss = pt.nn.BCEWithLogitsLoss(reduction='mean')
        try:
            model = build_model(loss, X, y, interface_, red, lb)
            l = model(X, y)
        except AssertionError as e2:
            assert 'reduction' in e2.args[0]
        else:
            raise
    else:
        raise
