<table>
    <tr>
        <td rowspan=3>
            <b> CI </b>
        </td>
        <td>
            Test
        </td>
        <td>
            <a href="https://github.com/iutzeler/skwdro/actions/workflows/doc.yml" alt="Doc tests"><img alt="Workflow Test" src="https://img.shields.io/github/actions/workflow/status/iutzeler/skwdro/test.yml?style=for-the-badge&label=Tests"></a>
        </td>
    </tr>
    <tr>
        <td>
            Style
        </td>
        <td>
            <a href="https://github.com/iutzeler/skwdro/actions/workflows/doc.yml" alt="Doc tests"><img alt="Workflow Style" src="https://img.shields.io/github/actions/workflow/status/iutzeler/skwdro/style.yml?style=for-the-badge&label=Style"></a>
        </td>
    </tr>
    <tr>
        <td>
            Doc
        </td>
        <td>
            <a href="https://github.com/iutzeler/skwdro/actions/workflows/doc.yml" alt="Doc tests"><img alt="Workflow Doc" src="https://img.shields.io/github/actions/workflow/status/iutzeler/skwdro/doc.yml?style=for-the-badge&label=Doc build"></a>
        </td>
    </tr>
    <tr>
        <td>
            <b> Doc </b>
        </td>
        <td>
            Readthedocs
        </td>
        <td>
            <a href="https://skwdro.readthedocs.io/en/latest/" alt="Read the Docs"><img src="https://img.shields.io/badge/ReadTheDocs-blue?style=for-the-badge&logo=sphinx"></a>
        </td>
    </tr>
    <tr>
        <td rowspan=3>
            <b> Checks </b>
        </td>
        <td>
            Code style
        </td>
        <td>
            <a href="https://github.com/astral-sh/ruff" alt="Ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json&style=for-the-badge"></a>
        </td>
    </tr>
    <tr>
        <td>
            Types
        </td>
        <td>
            <a href="https://github.com/python/mypy" alt="MyPY"><img src="https://img.shields.io/badge/mypy-checked-blue?style=for-the-badge&logo=python"></a>
        </td>
    </tr>
    <tr>
        <td>
            Build
        </td>
        <td>
            <a href="https://github.com/prefix-dev/rattler-build" alt="Rattlebuild-badge"><img src="https://img.shields.io/badge/Built_by-rattle--build-yellow?logo=anaconda&style=for-the-badge&logoColor=black"></a>
        </td>
    </tr>
    <tr>
        <td rowspan=3>
            <b> Install </b>
        </td>
        <td>
            Pip
        </td>
        <td>
            <a href="https://pypi.org/project/skwdro/"><img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/skwdro?style=for-the-badge"></a>
        </td>
    </tr>
    <tr>
        <td>
            Conda
        </td>
        <td>
            <a href="https://anaconda.org/flvincen/skwdro"> <img src="https://anaconda.org/flvincen/skwdro/badges/version.svg" /> </a>
        </td>
    </tr>
    <tr>
        <td>
            Github
        </td>
        <td>
            <a href="https://github.com/iutzeler/skwdro"><img src="https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white"></a>
        </td>
    </tr>
</table>

<div align="center">
  <h1>SkWDRO - Wasserstein Distributionally Robust Optimization for Machine Learning</h1>
  <h4>Model robustification with thin Pytorch and Scikit-learn interface</h4>
  <h6><q cite="https://adversarial-ml-tutorial.org/introduction">You can make pigs fly</q>, <a href="https://adversarial-ml-tutorial.org/introduction">[Kolter&Madry, 2018]</a></h6>
</div>

<a href="https://www.python.org" alt="Python"><img src="https://img.shields.io/badge/Python-blue?logo=python&logoColor=yellow&style=for-the-badge"></a>
<a href="https://pytorch.org/" alt="PyTorch"><img src="https://img.shields.io/badge/PyTorch-purple?logo=PyTorch&style=for-the-badge"></a>
<a href="https://scikit-learn.org" alt="Scikit Learn"><img src="https://img.shields.io/badge/ScikitLearn-red?logo=scikit-learn&style=for-the-badge"></a>
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

``skwdro`` is now available on PyPi.

Run the following command to get the latest version of the package

```shell
pip install -U skwdro
```

It is also available on [anaconda.org](anaconda.org) and can be installed using, for instance:

```shell
conda install -c conda-forge -c flvincen -c pytorch skwdro
```

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

Didn't find an estimator that suits you? You can compose your own using the ``pytorch`` interface: it allows more flexibility, custom models and optimizers.

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
        loss = robust_loss(batch_x, batch_y)
        loss.backward()
        optimizer.step()
```

You will find detailed description on how to `robustify` modules in the documentation.


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
