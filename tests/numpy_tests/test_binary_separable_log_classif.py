import numpy as np

from skwdro.base.costs import NormLabelCost
from skwdro.linear_models import LogisticRegression
from skwdro.solvers.optim_cond import OptCond

ANGLE_TOL = 1e-1 * np.pi

def generate_points():
    opposites = np.array([[1., 1.], [-1., -1.]])
    labels = np.array([1., -1., 1., -1., 1., -1.])
    return np.concatenate((
        opposites,
        opposites+np.array([[-1, 1]]),
        opposites-np.array([[-1, 1]])
        ), axis=0), labels


def fit_estimator(fi=True):
    estimator = LogisticRegression(
            rho=1e-3,
            l2_reg=None,
            fit_intercept=fi,
            cost=NormLabelCost(2., 1., 10**np.random.randint(2, 4), "test"),
            solver="entropic",
            solver_cond=OptCond(np.inf, 1e-6, 1e-6, 10000)
        )
    X, y = generate_points()
    estimator.fit(X, y)
    return estimator

def angle_to_northeast(coefs):
    r"""
    Angle between theta and [1,1]^T.

    .. math::
        \cos\alpha = \frac{\|\vec{1}\vec{1}^T\theta\|}{\|\vec{1}\|^2\|\theta\|}\\
            = \frac{\sqrt{2}|\theta_1+\theta_2|}{2\|\theta\|}\\
            = \frac{|\theta_1+\theta_2|}{\sqrt{2}\|\theta\|}
    """
    return np.arccos(abs(coefs.sum())/np.linalg.norm(coefs)/np.sqrt(2))

def test_fit_enthropic_fi():
    estimator = fit_estimator()
    # Needs more tolerance for now since intercept difficult to estimate
    assert angle_to_northeast(estimator.coef_) < ANGLE_TOL

def test_fit_enthropic_lin():
    estimator = fit_estimator(False)
    assert angle_to_northeast(estimator.coef_) < ANGLE_TOL