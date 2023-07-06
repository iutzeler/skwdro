"""
===================
Logistic regression
===================

An example of logistic regression for binary classification.
"""

from time import time

from sklearn.datasets import make_blobs
from numpy.random import RandomState
import matplotlib.pyplot as plt
import numpy as np
import torch as pt

from skwdro.linear_models import LogisticRegression
from skwdro.base.costs import NormCost
from skwdro.base.costs_torch import NormLabelCost


# Generate the random data ##################################################
rng = RandomState(seed=666)
n = 100
d = 2
X, y, centers = make_blobs(n, d, centers=2, shuffle=True, random_state=rng, return_centers=True) # type: ignore
y = 2 * y - 1 # type: ignore

# Center data to avoid learning intercept
#X -= centers.mean(axis=0, keepdims=True) # type : ignore
# ###########################################################################
cost=NormLabelCost(2., 1., 100., "Kappa-cost (w/ kappa=100)")
#cost=NormCost(2., 1., "=")


# SPECIFIC SOLVER
print("Specific solver w/ LP fast solve #####")
t = time()
print(".", end='')
estimator = LogisticRegression(
        rho=1e-2,
        l2_reg=0.,
        fit_intercept=True,
        cost="n-NC-2-2",
        solver="dedicated"
        )
print(".", end='')
estimator.fit(X, y)
print(".")

print("Lambda: ", estimator.dual_var_)
print("Theta: ", estimator.coef_, estimator.intercept_)
print("Elapsed time: ", time()-t)

print("#######")

# ENTROPIC SOLVER
print("Sinkhorn solver #####")
t = time()
print(".", end='')
estimator_ent = LogisticRegression(
        rho=1e-2,
        l2_reg=0.,
        fit_intercept=True,
        cost="t-NLC-2-2",
        n_zeta_samples=20,
        solver="entropic_torch"
        )


print(".", end='')
estimator_ent.fit(X, y)
print(".")

print("Lambda: ", estimator_ent.dual_var_)
print("Theta: ", estimator_ent.coef_, estimator_ent.intercept_)
print("Elapsed time: ", time()-t)

# ENTROPIC SOLVER
print("Sinkhorn pre-sampled solver #####")
t = time()
print(".", end='')
estimator_pre = LogisticRegression(
        rho=1e-2,
        l2_reg=0.,
        fit_intercept=True,
        cost="t-NLC-2-2",
        n_zeta_samples=20,
        solver="entropic_torch_post"
        )


print(".", end='')
estimator_pre.fit(X, y)
print(".")

print("Lambda: ", estimator_pre.dual_var_)
print("Theta: ", estimator_pre.coef_, estimator_pre.intercept_)
print("Elapsed time: ", time()-t)

print("ERM (rho=0) solver #####")
t = time()
print(".", end='')
estimator_ent = LogisticRegression(
        rho=0.,
        l2_reg=0.,
        fit_intercept=True,
        cost="t-NLC-2-2",
        n_zeta_samples=0,
        solver="entropic_torch"
        )


print(".", end='')
estimator_ent.fit(X, y)
print(".")

print("Lambda: ", estimator_ent.dual_var_)
print("Theta: ", estimator_ent.coef_, estimator_ent.intercept_)
print("Elapsed time: ", time()-t)
def plot_line(est, x):
    c0, c1 = est.coef_
    return -(x*c0 + est.intercept_) / c1

plt.scatter(X[y==-1, 0], X[y==-1, 1], color="r")
plt.scatter(X[y==1, 0], X[y==1, 1], color="b")
line_plot = [X[:, 0].min(), X[:, 0].max()]
plt.plot(line_plot, [plot_line(estimator, line_plot[0]), plot_line(estimator, line_plot[1])], 'k--', label=f"cvx: {estimator.score(X, y)}")
plt.plot(line_plot, [plot_line(estimator_ent, line_plot[0]), plot_line(estimator_ent, line_plot[1])], 'k:', label=f"Adam: {estimator_ent.score(X, y)}")
plt.plot(line_plot, [plot_line(estimator_pre, line_plot[0]), plot_line(estimator_pre, line_plot[1])], 'k-', label=f"BFGS: {estimator_pre.score(X, y)}")

plt.legend()
plt.show()

print("#######")
