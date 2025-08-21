r"""
Effect of the epsilon Sinkhorn regularization parameter
=======================================================

This example illustrates the use of the :class:`skwdro.linear_models.LogisticRegression` class on datasets that are shifted at test time.
It uses this setting to study the (small) impact of the regularization hyperparameter on the accuracy of the classification.

"""
import numpy as np

from sklearn.datasets import make_blobs

from skwdro.linear_models import LogisticRegression

from skwdro.solvers.optim_cond import OptCondTorch
from utils.classifier_comparison_utils import plot_classifier_comparison

# %%
# Setup
# ~~~~~

n = 50 # Total number of samples
n_train = (3 * n) // 4 # Number of training samples
n_test = n - n_train # Number of test samples

sdevs = [(2.5, 5), (1, 5)]

# Fix centers for blobs dataset
pos = 4
centers = [np.array([-pos,-pos]), np.array([pos,pos])]

# Create datasets with variance that is shifted at test time
datasets = []
for (sdev_1, sdev_2) in sdevs:
    train_dataset = make_blobs(n_samples=n_train, centers=centers, cluster_std=(sdev_1, sdev_2)) # type: ignore
    test_dataset = make_blobs(n_samples=n_test, centers=centers, cluster_std=(sdev_2, sdev_1)) # type: ignore
    datasets.append((train_dataset, test_dataset))

# %%
# WDRO classifiers
# ~~~~~~~~~~~~~~~~
# We build various ``SkWDRO`` estimators for :math:`\varepsilon` varying.

# Rho chosen analytically
rho = 1e-0  # 2*4**2

# Enthropic regularization: test various ones
e0, e1 = -3, 1
regs = np.logspace(e0, e1, base=10, num=5)

# Kappa: weight of label shift
kappa = 100000

# Cost:
# t: torch backend
# NLC: norm cost that takes labels into account
# 2 2 : squared 2-norm
# kappa: weight of label shift
cost = f"t-NLC-2-2-{kappa}"

# WDRO classifier
classifiers = [
    LogisticRegression(rho=0.),
    *(LogisticRegression(
        rho=rho,
        cost=cost,
        solver_reg=eps,
        n_zeta_samples=100,
        opt_cond=OptCondTorch(order='inf', tol_theta=1e-9, mode='abs')
    ) for eps in regs)
]

# %%
# Make plot
# ~~~~~~~~~
# Observe that the accuracy changes with the :math:`\varepsilon`.

names = [
    "Logistic Regression",
    *(f"$\\rho=32$, $\\varepsilon=10^{{{eps}}}$" for eps in range(e0, e1+1))
]
levels = [0., 0.1, 0.25, 0.45, 0.5, 0.55, 0.75, 0.9, 1.]
plot_classifier_comparison(names, classifiers, datasets, levels=levels) # type: ignore
