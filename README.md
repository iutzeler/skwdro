[![CI tests](https://github.com/iutzeler/skwdro/actions/workflows/test.yml/badge.svg?branch=dev)](https://github.com/iutzeler/skwdro/actions/workflows/test.yml)
[![style tests](https://github.com/iutzeler/skwdro/actions/workflows/style.yml/badge.svg)](https://github.com/iutzeler/skwdro/actions/workflows/style.yml)
[![doc tests](https://github.com/iutzeler/skwdro/actions/workflows/doc.yml/badge.svg)](https://github.com/iutzeler/skwdro/actions/workflows/doc.yml)


<div align="center">
  <h1>SkWDRO - Wasserstein Distributionaly Robust Optimization</h1>
  <h4>Model robustification with thin interface</h4>
  <h6><q cite="https://adversarial-ml-tutorial.org/introduction">You can make pigs fly</q>, <a href="https://adversarial-ml-tutorial.org/introduction">[Kolter&Madry, 2018]</a></h6>
</div>

[![Python](https://img.shields.io/badge/Python-blue?logo=python&logoColor=yellow&style=for-the-badge)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-purple?logo=PyTorch&style=for-the-badge)](https://pytorch.org/)
[![Scikit Learn](https://img.shields.io/badge/ScikitLearn-red?logo=scikit-learn&style=for-the-badge)](https://scikit-learn.org)
![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg?style=for-the-badge)



``skwdro`` is a Python package that offers **WDRO versions** for a large range of estimators, either by extending **``scikit-learn`` estimator** or by providing a wrapper for **``pytorch`` modules**.

Have a look at ``skwdro`` [documentation](https://skwdro.readthedocs.io/en/latest/)!

<!-- 
# Why WDRO & ``skwdro``?
 -->


# Getting started with ``skwdro``

## Installation

### Development mode with ``hatch``

First install ``hatch`` and clone the archive. In the root folder, ``make shell`` gives you an interactive shell in the correct environment and ``make test`` runs the tests (it can be launched from both an interactive shell and a normal shell).
``make reset_env`` removes installed environments (useful in case of troubles).

### With ``pip``

``skwdro`` will be available on PyPi *soon*, for now only the *development mode* is available.

<!--  Run the following command to get the latest version of the package

```shell
pip install -U skwdro
```

It is also available on conda-forge and can be installed using, for instance:

```shell
conda install -c conda-forge skwdro
``` -->

## First steps with ``skwdro``

### ``scikit-learn`` interface

Robust estimators from ``skwdro`` can be used as drop-in replacements for ``scikit-learn`` estimators (they actually inherit from ``scikit-learn`` estimators and classifier classes.). ``skwdro`` provides robust estimators for standard problems such as linear regression or logistic regression. ``LinearRegression`` from ``skwdro.linear_model`` is a robust version of ``LinearRegression`` from ``scikit-learn`` and be used in the same way. The only difference is that now an uncertainty radius ``rho`` is required.

We assume that we are given ``X_train`` of shape ``(n_train, n_features)`` and ``y_train`` of shape ``(n_train,)`` as training data and ``X_test`` of shape ``(n_test, n_features)`` as test data.

```python
from skwdro.linear_model import LinearRegression

# Uncertainty radius
rho = 0.1

# Fit the model
robust_model = LinearRegression(rho=rho)
robust_model.fit(X_train, y_train)

# Predict the target values
y_pred = robust_model.predict(X_test)
```
You can refer to the documentation to explore the list of ``skwdro``'s already-made estimators.


### ``pytorch`` interface

Didn't find a estimator that suits you? You can compose your own using the ``pytorch`` interface: it allows more flexibility, custom models and optimizers.

Assume now that the data is given as a dataloader `train_loader`.

```python
import torch
import torch.nn as nn
import torch.optim as optim

from skwdro.torch import robustify

# Uncertainty radius
rho = 0.1

# Define the model
model = nn.Linear(n_features, 1)

# Define the loss function
loss_fn = nn.MSELoss()

# Define a sample batch for initialization
sample_batch_x, sample_batch_y = next(iter(train_loader))

# Robust loss
robust_loss = robustify(loss_fn, model, rho, sample_batch_x, sample_batch_y)

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        loss = robust_loss(model(batch_x), batch_y)
        loss.backward()
        optimizer.step()
```

You will find detailed description on how to `robustify` modules.


<!-- 
# Cite

``skwdro`` is the result of perseverant research. It is licensed under [BSD 3-Clause](https://github.com/scikit-learn-contrib/skwdro/blob/main/LICENSE). You are free to use it and if you do so, please cite

```bibtex
@inproceedings{skwdro,
    title     = {},
    author    = {},
    booktitle = {},
    year      = {},
}
``` -->


# Useful links

- link to documentation: https://contrib.scikit-learn.org/skwdro/
- link to ``skwdro`` arXiv article: https://arxiv.org/pdf/2204.07826.pdf


The goal of this toolbox is to provide robust versions of classical machine learning models using the framework of Wasserstein Distributionally Robust Optimization.

Here is a draft of the doc: [https://skwdro.readthedocs.io/en/latest/](https://skwdro.readthedocs.io/en/latest/)

### Warning: The project is currently UNDER DEVELOPMENT 

skwdro can undergo significant changes in the interface or inner code. 

#### Development mode

To develop, first install `hatch`. `make shell` gives you an interactive shell in the correct environment and `make test` runs the tests (it can be launched from both an interactive shell and a normal shell).
`make reset_env` removes installed environments, useful in case of troubles.

#### Example

To train a robust Linear Regression model, you can mimick the syntax of scikit-learn:
```
from sklearn.linear_model import LinearRegression
from skwdro.linear_models import LinearRegression as RobustLinearRegression
```
