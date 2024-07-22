import numpy as np

from skwdro.linear_models import LogisticRegression

from test_binary_separable_log_classif import generate_points, angle_to_northeast

ANGLE_TOL = 1e-1 * np.pi

def fit_estimator(fi=True):
    estimator = LogisticRegression(
            rho=1e-2,
            l2_reg=0.1,
            cost="t-NC-1-1",
            fit_intercept=fi,
            solver="dedicated"
        )
    X, y = generate_points()
    estimator.fit(X, y)
    return estimator

def test_fit_enthropic_fi(): # [WIP]
    estimator = fit_estimator()
    # Needs more tolerance for now since intercept difficult to estimate
    assert angle_to_northeast(estimator.coef_) < 10*ANGLE_TOL

def test_fit_enthropic_lin():# [WIP]
    estimator = fit_estimator(False)
    assert angle_to_northeast(estimator.coef_) < ANGLE_TOL
