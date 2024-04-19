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
