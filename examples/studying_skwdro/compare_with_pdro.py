r"""
######################################
Comparison with the python-dro package
######################################

.. admonition:: TLDR

   A new toolbok appeared for general DRO. Their support for Wasserstein ambiguity sets is limited to certain specific models; ``SkWDRO`` is thus complementary to it.
   In the intersection of our two playgrounds, one can find (regularized) WDRO linear regressions. So we run a quick comparison notebook bellow. In short: both get similar accuracy performances, but ``SkWDRO`` often yields similar or better running times.

As us authors of ``SkWDRO`` were developping this library, an amazing new challenger has appeared in the scene of libraries for distributionally robust optimisation: `python-dro <https://python-dro.org>`_ emerged to tackle a **very** wide range of ambiguity sets (as we explain in the `wdro tutorial <wdro.html#distributional-robustness-divergences>`__), and they propose both the Wasserstein ambiguity set as well as its Sinkhorn regularized counterpart.
As of the version `0.3.3 <https://github.com/namkoong-lab/dro/releases/tag/v0.3.3>`__ of their repository, those are implemented for specific cases:

* for WDRO: only for linear models, and for specific neural networks under :math:`W_\infty` uncertainty (i.e. adversarial attacks, of the same flavor as so-called "*fast-gradient-sign attacks*"),
* for Sinkhorn-regularized-WDRO: only for linear models (even though to be fair, generalization of their interface to neural networks seems achievable at first glance).

This example notebook illustrates the use of the :class:`skwdro.linear_models.LinearRegression` side to side with the (KL-regularized or not) version of WDRO implemented by the python-dro library.
The aim is to compare fairly the two libraries with similar hyperparameters settings, on a similar task.
"""
import timeit
import tqdm.auto as tqdm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.datasets import make_regression
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.utils import check_X_y

from dro.linear_model.wasserstein_dro import WassersteinDRO
from dro.linear_model.sinkhorn_dro import SinkhornLinearDRO
from skwdro.linear_models import LinearRegression

# %%
# Problem setup
# =============

# Total number of samples: chosen to be a bit prohibitive for SVM-like kernel methods (would deserve a separate analysis)
n = 500
# "Low"-dimensional setting to avoid this notebook to run for unreasonable amounts of time.
d = 5
n_train = int(np.floor(0.8 * n)) # Number of training samples: 80% of dataset
n_test = n - n_train # Number of test samples

# Generate some data
X, y, *_ = make_regression(n_samples=n, n_features=d, noise=50, random_state=0)
assert isinstance(X, np.ndarray)

# Normalize the data
X = minmax_scale(X, feature_range=(-1, 1))
y = minmax_scale(y, feature_range=(-1, 1))

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=n_train, test_size=n_test, random_state=0)
assert isinstance(X_train, np.ndarray)
assert isinstance(X_test, np.ndarray)

# %%
# WDRO linear regression
# ======================
#
# Bellow we define multiple models for the robust optimisation of a
# linear regression problem (OLS):
#
# * the non-regularized original WDRO formulation, cast as a convex optimisation
#   problem for the 2-norm squared (i.e. :math:`W_2` measure transport cost,
#   for the Mahalanobis distance),
# * the regularized Sinkhorn-Wasserstein distance according to the two similar
#   but subtly divergent approaches of Gao et al. [#WGX23]_ and
#   Azizian et al. [#AIM23]_ with regard to the way the problem is dualized,
#   both in the same hyperparameter settings.
#
# Those cases are treated in a low dimensional setting for ease of reproduction
# with the two libraries at hand.
#
# In terms of reproducibility, we set a few of the hyperparameters: the distance
# used for the ground-cost is the norm (squared) with unit metric tensor (
# implying also an isotropic covariance matrix for the sampler), and no target
# switches allowed.
# The variance of the sampler is fixed, as well as the regularization parameter.
# Recall that the latter bears slightly different meanings in the two approaches,
# refer to the formula we present in
# `the Sinkhorn regularization tutorial <why_skwdro.html>`__ as compared to the
# work of Gao [#WGX23]_.
# We set a fixed number of SGD iterations for the two libraries (5000 here, which
# seems to be enough).

rhos = [1e-6, 1e-3, 1e-1]
SIGMA = 1e-2
EPSILON_REGULARISATION = 1e-1

def fit_wdro_from_dro(rho: float):
    estimator = WassersteinDRO(
        input_dim=d,
        solver='SCS',
        model_type='ols'
    )
    estimator.update({
        'cost_matrix': np.eye(d),
        'eps': rho,
        'p': 2,
        'kappa': 'inf'
    })
    estimator.fit(*check_X_y(X_train, y_train))
    return estimator

def fit_skwdro_from_dro(rho: float):
    estimator = SinkhornLinearDRO(
        input_dim=d,
        fit_intercept=True,
        max_iter=5_000,
        reg_param=EPSILON_REGULARISATION,
        model_type='ols'
    )
    estimator.update({
        'cost_matrix': np.eye(d),
        'eps': rho,
        'p': 2,
        'kappa': 'inf'
    })
    estimator.fit(*check_X_y(X_train, y_train))
    return estimator

def fit_wdro_from_skwdro(rho: float):
    estimator = LinearRegression(
        solver='dedicated',
        rho=rho,
        fit_intercept=True
    )
    estimator.fit(X_train, y_train)
    return estimator

def fit_skwdro_from_skwdro(rho: float):
    estimator = LinearRegression(
        rho=rho,
        sampler_reg=SIGMA,
        learning_rate=1e-3,
        n_iter=5_000,
        solver_reg=EPSILON_REGULARISATION,
        fit_intercept=True
    )
    estimator.fit(X_train, y_train)
    return estimator

estimators_funcs = [
    fit_wdro_from_dro,
    fit_skwdro_from_dro,
    fit_wdro_from_skwdro,
    fit_skwdro_from_skwdro
]

# %%
# Evaluation
# ==========

all_train_errors = []
all_test_errors = []
all_timers = []
method_names = [
    'WDRO (p-dro)',
    'Sk-WDRO (p-dro)',
    'WDRO (skwdro)',
    'Sk-WDRO (skwdro)'
]

for rho in tqdm.tqdm(rhos, desc='Radii', position=0, leave=True):
    train_errors = []
    test_errors = []
    timers = []

    for fitter in tqdm.tqdm(estimators_funcs, desc='Method-score', leave=False, position=1):
        estimator = fitter(rho)
        train_errors.append(mean_squared_error(y_train, estimator.predict(X_train)))
        test_errors.append(mean_squared_error(y_test, estimator.predict(X_test)))

    all_train_errors.append(train_errors)
    all_test_errors.append(test_errors)

    for fn in [
        'fit_wdro_from_dro',
        'fit_skwdro_from_dro',
        'fit_wdro_from_skwdro',
        'fit_skwdro_from_skwdro'
    ]:
        timers.append(timeit.timeit(fn+'(rho)', globals=globals(), number=3))

    all_timers.append(timers)

# %%
# Plotting
# ========

# Build pandas DataFrames for seaborn
# Build pandas DataFrames for seaborn (rho treated as categorical)

def _rho_formatter(rho: float) -> str:
    return f"$10^{int(np.log10(rho))}$"

train_df = pd.DataFrame([
    {'rho': _rho_formatter(rhos[i]), 'method': method_names[j], 'train_error': all_train_errors[i][j]}
        for i in range(len(rhos)) for j in range(len(method_names))
])

test_df = pd.DataFrame([
    {'rho': _rho_formatter(rhos[i]), 'method': method_names[j], 'test_error': all_test_errors[i][j]}
        for i in range(len(rhos)) for j in range(len(method_names))
])

time_df = pd.DataFrame([
    {'rho': _rho_formatter(rhos[i]), 'method': method_names[j], 'time': all_timers[i][j]}
        for i in range(len(rhos)) for j in range(len(method_names))
])

# %%
# WDRO comparison plots
# ^^^^^^^^^^^^^^^^^^^^^
# We first propose a plot comparing disciplined WDRO implementations across
# libraries.
# It compares the test losses for the two methods, evaluated as the ERM (with
# mean-squared error), and the wall-clock running times.

wdro_methods = ['WDRO (p-dro)', 'WDRO (skwdro)']
wdro_test = test_df[test_df['method'].isin(wdro_methods)]
wdro_time = time_df[test_df['method'].isin(wdro_methods)]
assert isinstance(wdro_test, pd.DataFrame)
assert isinstance(wdro_time, pd.DataFrame)

# %%
# Test loss plot
plt.figure(figsize=(10,5))
ax = sns.barplot(data=wdro_test, x='rho', y='test_error', hue='method', palette='viridis')
for container in ax.containers:
    ax.bar_label(container)
plt.yscale('log')
plt.title('WDRO Test Errors Across Libraries')
plt.tight_layout()
# plt.savefig('/tmp/wdro_test_errors.png')
# plt.show()
# plt.close()

# %%
# We see that the two libraries are relatively similar, especially for small
# Wasserstein radii, which is to be expected considering the similarity between
# the implementations.

# %%
# Timing plot
plt.figure(figsize=(10,5))
ax = sns.barplot(data=wdro_time, x='rho', y='time', hue='method', palette='coolwarm')
for container in ax.containers:
    ax.bar_label(container)
plt.yscale('log')
plt.title('WDRO Timing Across Libraries')
plt.tight_layout()
# plt.savefig('/tmp/wdro_timing.png')
# plt.show()
# plt.close()

# %%
# Again to no surprise, the running times of the two libraries are usually
# comparable, even though it seems like the implementation in ``SkWDRO`` seems
# faster in some setting.
# For fair comparison, we may impute this difference to the prior factorization
# that python-dro performs on the Mahalanobis metric (here the identity matrix),
# which is not handled by ``SkWDRO``. We argue that whitening the data prior to
# the optimisation procedure might yield equivalent results in cases where this
# geometry is important, and on another hand that using the regularized
# formulation from our library would allow one to use the Mahalanobis distance
# as their transport cost (see the `costs tutorial <tutos/costs.html>`__ for
# more details on how to do that).

# %%
# SK-WDRO comparison plots
# ^^^^^^^^^^^^^^^^^^^^^^^^
# Here is another set of plots comparing Sinkhorn-regularized WDRO ("*sk-wdro*")
# implementations, both relying under the hood on ``PyTorch``.

# %%
# Let's compare Sinkhorn-based WDRO models from both libraries
sk_methods = ['Sk-WDRO (p-dro)', 'Sk-WDRO (skwdro)']
sk_test = test_df[test_df['method'].isin(sk_methods)]
sk_time = time_df[test_df['method'].isin(sk_methods)]
assert isinstance(sk_test, pd.DataFrame)
assert isinstance(sk_time, pd.DataFrame)


# %%
# Test loss plot
plt.figure(figsize=(10,5))
ax = sns.barplot(data=sk_test, x='rho', y='test_error', hue='method', palette='viridis')
for container in ax.containers:
    ax.bar_label(container)
plt.yscale('log')
plt.title('Sk-WDRO Test Errors Across Libraries')
plt.tight_layout()
# plt.savefig('/tmp/skwdro_test_errors.png')
# plt.show()
# plt.close()


# %%
# Timing plot
plt.figure(figsize=(10,5))
ax = sns.barplot(data=sk_time, x='rho', y='time', hue='method', palette='coolwarm')
for container in ax.containers:
    ax.bar_label(container)
plt.yscale('log')
plt.title('Sk-WDRO Timing Across Libraries')
plt.tight_layout()
# plt.savefig('/tmp/skwdro_timing.png')
# plt.show()
# plt.close()

# %%
# The speed of ``SkWDRO`` is substantially higher (:math:`\approx\times 100` faster)
# in this low dimensional setting with a medium-sized dataset.
# Other experiments could be run to show more balanced results with fewer samples
# (e.g. we obtained closer timings with :math:`n=100`), or more drastic difference
# of running time performance.
#
# For the test accuracy though, it seems to depend heavily on the chosen
# robustness radius: for smaller radii ``SkWDRO`` is more performant while
# the technique from [#WGX23]_ is more suited for higher radii.
# We have the intuition that this comes from the implementation in ``python-dro``
# which fixes the dual parameter :math:`\lambda`, removing the need to optimize
# it. In exchange, this forces the user to pick a good starting value for it.
# We invite curious readers to tune it by hand for their code in order to see
# better and more radius-agnostic convergence properties.
#
# Tackling non-linear models
# ==========================
#
# To mark the difference in goals between ``SkWDRO`` and ``python-dro``, we highlight
# that our library's approach focuses on large parameters space models like
# neural networks that are not implementable in other frameworks. This is
# illustrated in other tutorials in this documentation.
#
# Neural nets on simple examples
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# You can first try our library on simpler low-dimensional examples that are
# tractable enough for obtaining quick and easy visual cues.
# This is illustrated in more details in the
# `moons dataset example <examples/Custom/neural_net.html>`__, in a simple two
# dimensional setting with a non-linearly-separable dataset.
#
# Neural net on more difficult datasets
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# This approach can scale, thanks to pytorch, to higher-dimensional settings,
# at the expense of computation time.
# We showcase this on the `iWildsCam dataset <https://wilds.stanford.edu/>`__,
# in a
# `separate documentation page explaining the experiments <../../wilds.html>`__.
# This example also shows how the optimisation procedure behaves in more
# challenging setting.
#
# References
# ==========
# .. [#AIM23] Azizian, Iutzeler, and Malick: **Regularization for Wasserstein Distributionally Robust Optimization**, *COCV*, 2023
# .. [#WGX23] Wang, Gao, and Xie: **Sinkhorn Distributionally Robust Optimization**, *arXiv (2109.11926)*, 2023
