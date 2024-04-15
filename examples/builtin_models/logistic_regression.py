r"""
Logistic regression
=====================
This example illustrates the use of the :class:`skwdro.linear_models.LogisticRegression` class and the influence of the radius.

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

n = 200 # Total number of samples
n_train = n // 2 # Number of training samples
n_test = n - n_train # Number of test samples

datasets = []

# Moons dataset
dataset = make_moons(n_samples=n, noise=0.1)
Xtrain, Xtest, ytrain, ytest = train_test_split(*dataset, train_size=n_train, test_size=n_test)
train_dataset, test_dataset = (Xtrain, ytrain), (Xtest, ytest)
datasets.append((train_dataset, test_dataset))

# Fix centers for blobs dataset
_, _, centers = make_blobs(centers=2, return_centers=True)

# Blobs dataset with std=2
dataset = make_blobs(n_samples=n, centers=centers, cluster_std=2)
Xtrain, Xtest, ytrain, ytest = train_test_split(*dataset, train_size=n_train, test_size=n_test)
train_dataset, test_dataset = (Xtrain, ytrain), (Xtest, ytest)
datasets.append((train_dataset, test_dataset))

# Blobs dataset with std=4
dataset = make_blobs(n_samples=n, centers=centers, cluster_std=4)
Xtrain, Xtest, ytrain, ytest = train_test_split(*dataset, train_size=n_train, test_size=n_test)
train_dataset, test_dataset = (Xtrain, ytrain), (Xtest, ytest)
datasets.append((train_dataset, test_dataset))

# %%
# WDRO classifiers
# ~~~~~~~~~~~~~~~~

rhos = [0.001, 0.01, 0.1]
classifiers = []
for rho in rhos:
    classifiers.append(LogisticRegression(rho=rho))

# %%
# Make plot
# ~~~~~~~~~

names = [f"$\\rho={rho}$" for rho in rhos]
plot_classifier_comparison(names, classifiers, datasets)

