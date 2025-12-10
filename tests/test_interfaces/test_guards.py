import pytest
import warnings
import numpy as np
from sklearn.exceptions import DataConversionWarning
from skwdro.linear_models import LinearRegression, LogisticRegression
from skwdro.operations_research import Weber, NewsVendor, Portfolio

X = np.random.rand(3, 2)
Y = np.array([0, 1, 0])

all_constrs = pytest.mark.parametrize("constr,l", [
    (LinearRegression, True), (LogisticRegression, True),
    (Weber, True), (NewsVendor, False), (Portfolio, False)
])

@all_constrs
def test_init(constr, l: bool):
    del l
    model = constr(n_iter=10)
    assert hasattr(model, 'rho')
    assert hasattr(model, 'random_state')
    with pytest.raises(ValueError) as e_err:
        model = constr(rho=-1.)
    assert "non-negative" in str(e_err.value)

@all_constrs
def test_rho_typing(constr, l: bool):
    del l
    model = constr(n_iter=10)
    model.rho = None
    with pytest.raises(TypeError) as e_err:
        model.fit(X, Y)
    assert "should be numeric" in str(e_err.value)

@all_constrs
def test_y_shape_controls(constr, l: bool):
    model = constr()
    if l:
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            model.fit(X, Y[:, None])
            assert len(ws) == 1
            assert issubclass(ws[-1].category, DataConversionWarning)

@all_constrs
def test_deprecation_np(constr, l: bool):
    del l
    model = constr(n_iter=10, solver='entropic')
    with pytest.raises(DeprecationWarning) as e_err:
        model.fit(X, Y)
    assert "The entropic (numpy) solver is now deprecated" in str(e_err.value)

@all_constrs
def test_solver_typos(constr, l: bool):
    del l
    model = constr(n_iter=10, solver='typos')
    with pytest.raises(NotImplementedError) as e_err:
        model.fit(X, Y)
    assert "Designation for solver not recognized" in str(e_err.value)

@all_constrs
def test_no_oc(constr, l: bool):
    model = constr(n_iter=10, opt_cond = None, solver='torch')
    model.fit(X, Y if l else None)
    assert hasattr(model, "coef_")

@pytest.mark.parametrize('constr,s,l', [
    (LinearRegression, 'dedicated', True),
    (LinearRegression, 'torch', True),
    (LogisticRegression, 'dedicated', True),
    (LogisticRegression, 'torch', True),
    (Weber, 'torch', True),
    (Portfolio, 'torch', False),
    (NewsVendor, 'torch', False),
    (NewsVendor, 'dedicated', False),
])
def test_types_solves(constr, s:  str, l: bool):
    model = constr(n_iter=10, solver=s)
    model.fit(X, Y if l else None)
    assert hasattr(model, "coef_")
