import numpy as np

from sklearn.metrics import accuracy_score
from skwdro.linear_models import LogisticRegression

ANGLE_TOL = 2e-1 * np.pi


# ##################################
# ##### BINARY CLASSIFICATION ######
# ##################################

def generate_points():
    n = 20
    opposites = np.array([[1., 1.], [-1., -1.]])
    data = np.concatenate([
        opposites[0, :] + np.random.normal(0, 2, size=(n, 1)) * np.array([[-1, 1]]),
        opposites[1, :] + np.random.normal(0, 2, size=(n, 1)) * np.array([[-1, 1]])
    ], axis=0)
    labels = np.array([1.] * n + [-1.] * n)
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
    assert accuracy_score(y, estimator.predict(X)) > 0.8
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


# #####################################
# ##### MULTICLASS CLASSIFICATION #####
# #####################################


def generate_multiclass_points():
    """
    Generate synthetic data for 3-class classification.
    """
    n = 20
    centers = np.array([
        [2, 2],  # Class 0
        [-2, -2],  # Class 1
        [2, -2],  # Class 2
    ])
    data = np.concatenate([
        centers[0, :] + np.random.normal(0, 1, size=(n, 2)),
        centers[1, :] + np.random.normal(0, 1, size=(n, 2)),
        centers[2, :] + np.random.normal(0, 1, size=(n, 2)),
    ])
    labels = np.array([0] * n + [1] * n + [2] * n)
    return data, labels


def fit_multiclass_estimator(fi=True):
    estimator = LogisticRegression(
        rho=1e-3,
        l2_reg=0.0,
        cost="t-NLC-2-2",
        fit_intercept=fi,
        solver="entropic_torch",
    )
    X, y = generate_multiclass_points()
    estimator.fit(X, y)
    assert accuracy_score(y, estimator.predict(X)) > 0.8
    return estimator


def angle_to_centroids(coefs, centers):
    """
    Check if coefficient vectors align with the directions to class centroids.
    """
    centroid_norms = np.linalg.norm(centers, axis=1, keepdims=True)
    normed_centers = centers / centroid_norms
    coef_norms = np.linalg.norm(coefs, axis=0, keepdims=True)
    normed_coefs = coefs / coef_norms
    return np.abs(1.0 - np.abs(np.sum(normed_centers * normed_coefs, axis=0)))


def test_fit_multiclass_enthropic_fi():
    estimator = fit_multiclass_estimator()
    X, y = generate_multiclass_points()
    # Calculate centroids of classes for validation
    centers = np.array([X[y == i].mean(axis=0) for i in np.unique(y)])
    assert np.all(angle_to_centroids(estimator.coef_, centers) < ANGLE_TOL)


def test_fit_multiclass_enthropic_lin():
    estimator = fit_multiclass_estimator(False)
    X, y = generate_multiclass_points()
    # Calculate centroids of classes for validation
    centers = np.array([X[y == i].mean(axis=0) for i in np.unique(y)])
    assert np.all(angle_to_centroids(estimator.coef_, centers) < ANGLE_TOL)
