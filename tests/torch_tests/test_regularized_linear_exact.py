import numpy as np

from skwdro.linear_models import LogisticRegression

import pytest

import os

L2_REG = 1e-5 #Don't change

def fit_estimator(my_rho_norm, reg, sigma, X, y):
    rho_norm = my_rho_norm# ** 2 # because no 0.5 in cost
    estimator = LogisticRegression(
            rho=np.sqrt(2) * rho_norm,
            # rho=2.*rho_norm,
            l2_reg=L2_REG,
            cost="t-NC-2-2",
            # cost="t-NLC-2-2-1",
            fit_intercept=False,
            solver="entropic_torch",
            solver_reg=reg,
            sampler_reg=sigma,
            n_zeta_samples=50,
        )
    estimator.fit(X, y)
    return estimator.coef_, estimator.robust_loss_

direlpath = "data/log_reg_reg"
dirpath = os.path.join(os.path.split(__file__)[0], direlpath)

def decode(filename):
    root, _ = os.path.splitext(filename)
    splitted = root.split('_')
    d, n = int(splitted[3]), int(splitted[4])
    rho, eps, sigma = [float(v) for v in splitted[6:]]
    res = np.load(os.path.join(dirpath, filename))
    X, y, theta, robust_loss = res['X'], res['y'], res['theta'], res['robust_loss']
    res.close()
    return (n, d, rho, eps, sigma, X, y, theta, robust_loss)

files = os.listdir(dirpath)

ATOL = RTOL = 5e-2
LATOL = LRTOL = 5e-2
#@pytest.mark.xfail()
@pytest.mark.parametrize("n, d, rho, eps, sigma, X, y, theta, robust_loss", [decode(filename) for filename in files])
def test_log_reg_reg(n, d, rho, eps, sigma, X, y, theta, robust_loss):
    est_theta, est_robust_loss = fit_estimator(rho, eps, sigma, X, y)
    print(np.linalg.norm(est_theta - theta))
    nrm = lambda x : x / np.linalg.norm(x)
    assert np.isclose(nrm(est_theta), nrm(theta), atol=ATOL, rtol=RTOL).all()
    assert np.isclose(est_robust_loss, robust_loss, atol=LATOL, rtol=LRTOL).all()
