import numpy as np

from skwdro.linear_models import LogisticRegression

ANGLE_TOL = 1e-1 * np.pi

def generate_points():
    opposites = np.array([[1., 1.], [-1., -1.]])
    data = np.concatenate([
        opposites[0, :] + np.random.normal(0, 0.1, size=(100, 1)) * np.array([[-1, 1]]),
        opposites[1, :] + np.random.normal(0, 0.1, size=(100, 1)) * np.array([[-1, 1]])
    ], axis=0)
    labels = np.array([1.] * 100 + [-1.] * 100)
    return data, labels


def fit_estimator(fi=True):
    estimator = LogisticRegression(
            rho=1e-3,
            l2_reg=0.,
            cost="t-NLC-2-2",
            fit_intercept=fi,
            solver="entropic_torch"
        )
    X, y = generate_points()
    estimator.fit(X, y)
    return estimator

def angle_to_northeast(coefs):
    return np.abs(1. - abs(coefs.sum())/np.linalg.norm(coefs)/np.sqrt(2))

def test_fit_enthropic_fi():
    estimator = fit_estimator()
    # Needs more tolerance for now since intercept difficult to estimate
    assert angle_to_northeast(estimator.coef_) < ANGLE_TOL

def test_fit_enthropic_lin():
    estimator = fit_estimator(False)
    assert angle_to_northeast(estimator.coef_) < ANGLE_TOL
