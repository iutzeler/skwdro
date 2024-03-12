"""
===================
Linear regression
===================

An example of logistic regression for binary classification.
"""

from time import time

from sklearn.model_selection import train_test_split
from numpy.random import RandomState
import matplotlib.pyplot as plt
import numpy as np
import torch as pt

from skwdro.linear_models import LinearRegression
from skwdro.base.costs_torch import NormLabelCost


RHO = 1e-4
EPSILON = 1e-3

# Generate the random data ##################################################
rng = RandomState(seed=666)
n = 100

X = np.random.uniform(-1., 1., (n, 1))
alpha = rng.randn()
beta = rng.rand()
y = alpha * X.squeeze() + beta + 2e-1 * rng.randn(n)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=rng)

# Center data to avoid learning intercept
# ###########################################################################
cost=NormLabelCost(2., 1., 100., "Kappa-cost (w/ kappa=100)")
#cost=NormCost(2., 1., "=")


# SPECIFIC SOLVER
print("Specific solver w/ LP fast solve #####")
t = time()
print(".", end='')
estimator = LinearRegression(
        rho=RHO,
        l2_reg=0.,
        solver_reg=EPSILON,
        fit_intercept=True,
        cost="n-NC-2-2",
        solver="dedicated"
        )
print(".", end='')
estimator.fit(X_train, y_train)
print(".")

print("Lambda: ", estimator.dual_var_)
print("Theta: ", estimator.coef_, estimator.intercept_)
print("Elapsed time: ", time()-t)

print("#######")

# ENTROPIC SOLVER
print("Sinkhorn solver #####")
t = time()
print(".", end='')
estimator_ent = LinearRegression(
        rho=RHO,
        solver_reg=EPSILON,
        l2_reg=0.,
        fit_intercept=True,
        cost="t-NLC-2-2",
        n_zeta_samples=20,
        solver="entropic_torch"
        )


print(".", end='')
estimator_ent.fit(X_train, y_train)
print(".")

print("Lambda: ", estimator_ent.dual_var_)
print("Theta: ", estimator_ent.coef_, estimator_ent.intercept_)
print("Elapsed time: ", time()-t)

# ENTROPIC SOLVER
print("Sinkhorn pre-sampled solver #####")
t = time()
print(".", end='')
estimator_pre = LinearRegression(
        rho=RHO,
        solver_reg=EPSILON,
        l2_reg=0.,
        fit_intercept=True,
        cost="t-NLC-2-2",
        n_zeta_samples=20,
        solver="entropic_torch_pre"
        )


print(".", end='')
estimator_pre.fit(X_train, y_train)
print(".")

print("Lambda: ", estimator_pre.dual_var_)
print("Theta: ", estimator_pre.coef_, estimator_pre.intercept_)
print("Elapsed time: ", time()-t)

print("ERM (rho=0) solver #####")
t = time()
print(".", end='')
estimator_erm = LinearRegression(
        rho=0.,
        l2_reg=0.,
        fit_intercept=True,
        cost="t-NLC-2-2",
        n_zeta_samples=0,
        solver="entropic_torch_pre"
        )


print(".", end='')
estimator_erm.fit(X_train, y_train)
print(".")

print("Lambda: ", estimator_erm.dual_var_)
print("Theta: ", estimator_erm.coef_, estimator_erm.intercept_)
print("Elapsed time: ", time()-t)

def plot_line(est, x):
    c0, c1 = est.coef_, est.intercept_
    return c0 * x + c1

line_plot = [X.min(), X.max()] # type: ignore
fig, axes = plt.subplots(2, 2)
for ax, name, est in zip(
        axes.flatten(),
        ("cvx", "BFGS", "Adam", "Adam-ERM"),
        (estimator, estimator_pre, estimator_ent, estimator_erm)
         ):
    ax.scatter(X_train.flatten(), y_train, color="r") # type: ignore
    ax.scatter(X_test.flatten(), y_test, color="b") # type: ignore
    ax.plot(line_plot, [plot_line(estimator, line_plot[0]), plot_line(estimator, line_plot[1])], 'k:', label=f"cvx (baseline)")
    ax.plot(line_plot, [plot_line(est, line_plot[0]), plot_line(est, line_plot[1])], 'g-.', label=name+f": {est.score(X_test, y_test):.5f}/{est.score(X_train, y_train):.5f}")
    ax.legend()

plt.show()

print("#######")
