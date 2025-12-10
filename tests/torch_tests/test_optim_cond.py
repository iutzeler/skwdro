import pytest
import torch as pt
from itertools import product
from skwdro.solvers.optim_cond import OptCondTorch, L_AND_T, JUST_L
from skwdro.solvers.utils import NoneGradError
from skwdro.torch import robustify


setups = list(product(
    (1, 2, "inf"),
    (1e-5, -1, 0.),
    (1e-5, -1, 0.),
    (100, None),
    (
        "both",
        "theta_and_lambda",
        "t&l",
        "lambda_and_theta",
        "l&t",
        "one",
        "theta_or_lambda",
        "tUl",
        "lambda_or_theta",
        "lUt",
        "theta", "t",
        "lambda", "l"
    ),
    ('rel', 'abs'),
    ('param', 'grad'),
))

@pytest.mark.parametrize(
    "p,tt,tl,it,mon,mod,mt,v",
    [
        (1, 1e-5, 1e-5, 100, 'wrong', 'rel', 'param', True),
        (1, 1e-5, 1e-5, 100, 'both', 'wrong', 'param', True),
        (1, 1e-5, 1e-5, 100, 't', 'rel', 'wrong', True),
        (1, 1e-5, 1e-5, 100, 'l', 'rel', 'wrong', True),
        (1, 1e-5, 1e-5, 100, 'both', 'rel', 'param', True),
        *((*s, False) for s in setups[1:])
    ]
)
def test_opt_cond(
    p: int|str,
    tt: float,
    tl: float,
    it: int|None,
    mon: str,
    mod: str,
    mt: str,
    v: bool
):
    x, y = pt.randn(3, 2), pt.tensor([1000.]*3).unsqueeze(-1)
    if 'wrong' in (mon, mod, mt):
        with pytest.raises(ValueError) as e_err:
            assert it is not None
            oc = OptCondTorch(
                p,
                tt,
                tl,
                it,
                monitoring=mon,
                mode=mod,
                metric=mt,
                verbose=v
            )
            model1 = make_model(x, y)
            model2 = make_model(x, y, True)
            model1.n_iter = it
            model2.n_iter = it
            make_checks(oc, model1, model2, tt, it)
        assert 'Please' in str(e_err.value)

    else:
        oc = OptCondTorch(
            p,
            tt,
            tl,
            it,
            monitoring=mon,
            mode=mod,
            metric=mt,
            verbose=v
        )
        model1 = make_model(x, y)
        model2 = make_model(x, y, True)
        if it is not None:
            model1.n_iter = it
            model2.n_iter = it
        if mt == 'grad':
            if tt > 0. and mon in L_AND_T and mon not in JUST_L:
                with pytest.raises(ValueError) as e_err:
                    make_checks(oc, model1, model2, tt, it)
                check_ng_error(e_err)
            model1(x, y).backward()
            model2(x, y).backward()
            make_checks(oc, model1, model2, tt, it)
        else:
            make_checks(oc, model1, model2, tt, it)

def check_ng_error(e):
    assert 'Please' in str(e.value) or isinstance(e.value, NoneGradError)

def make_model(x, y, far=False):
    model = robustify(
        pt.nn.MSELoss(reduction='none'),
        pt.nn.Linear(2, 1),
        pt.tensor(1e-3),
        x, y
    )
    if far:
        assert isinstance(model.primal_loss.transform, pt.nn.Linear)
        model.primal_loss.transform.weight = pt.nn.Parameter(
            pt.ones_like(model.primal_loss.transform.weight)*100.
        )
        model._lam = pt.nn.Parameter(pt.tensor(1000.))
    return model

def make_checks(oc: OptCondTorch, model1, model2, tt, it):
    cvg1 = oc(model1, 1)
    if it == 100:
        assert oc(model2, 1000)
        assert (not cvg1) or (tt <= 0.)
    else:
        assert (not cvg1) or (tt <= 0.)
    for i in range(3 if it is not None else 1):
        if i > 1:
            oc(model1, i)
