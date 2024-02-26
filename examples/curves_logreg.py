
"""
===================
Logistic regression
===================

An example of logistic regression for binary classification.
"""

from sklearn.datasets import make_blobs
from numpy.random import RandomState
import numpy as np

from skwdro.linear_models import LogisticRegression


RHO = 1e-4

# Generate the random data ##################################################
rng = RandomState(seed=666)
n = 100
d = 2
X, y, centers = make_blobs(n, d, centers=2, shuffle=True, random_state=rng, return_centers=True) # type: ignore
y = 2 * y - 1 # type: ignore

# Center data to avoid learning intercept
#X -= centers.mean(axis=0, keepdims=True) # type : ignore
# ###########################################################################

# ENTROPIC SOLVER
print("Sinkhorn solver #####")
for epsilon in [1e-7, 1e-2, 1e1]:
    estimator_ent = LogisticRegression(
            rho=RHO,
            l2_reg=0.,
            fit_intercept=True,
            cost="t-NC-2-2",
            n_zeta_samples=100,
            solver_reg=epsilon,
            solver="entropic_torch"
            )
    estimator_ent.fit(X, y)
