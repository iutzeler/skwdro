r"""
Spatial perturbations and logistic regression
=====================
This example illustrates the use of the :class:`skwdro.linear_models.LogisticRegression` class on datasets that are shifted at test time.

"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from skwdro.linear_models import LogisticRegression

from utils.classifier_comparison import plot_classifier_comparison

# %%
# Setup
# ~~~~~

n = 500 # Total number of samples
n_train = (3 * n) // 4 # Number of training samples
n_test = n - n_train # Number of test samples

sdevs = [(2.5, 5), (1, 5)]

# Fix centers for blobs dataset
pos = 4
centers = [np.array([-pos,-pos]), np.array([pos,pos])]

# Create datasets with variance that is shifted at test time
datasets = []
for (sdev_1, sdev_2) in sdevs:
    train_dataset = make_blobs(n_samples=n_train, centers=centers, cluster_std=(sdev_1, sdev_2))
    test_dataset = make_blobs(n_samples=n_test, centers=centers, cluster_std=(sdev_2, sdev_1))
    datasets.append((train_dataset, test_dataset))

# %%
# WDRO classifiers
# ~~~~~~~~~~~~~~~~

# Rho chosen analytically
rhos = [0, 2*4**2]

# Kappa: weight of label shift
kappa = 1000

# Cost:
# t: torch backend
# NLC: norm cost that takes labels into account
# 2 2 : squared 2-norm
# kappa: weight of label shift
cost = f"t-NLC-2-2-{kappa}"

# WDRO classifier
classifiers = []
for rho in rhos:
    classifiers.append(LogisticRegression(rho=rho, cost=cost))

# %%
# Make plot
# ~~~~~~~~~

names = ["Logistic Regression", "WDRO Logistic Regression"]
levels = [0., 0.25, 0.45, 0.5, 0.55, 0.75, 1.]
plot_classifier_comparison(names, classifiers, datasets, levels=levels)
