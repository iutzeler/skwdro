import numpy as np
    
import sklearn
import sklearn.datasets
import sklearn.model_selection
import sklearn.linear_model
from sklearn.neighbors import KernelDensity

import skwdro
import skwdro.linear_models

import pytest

d = 10
n_train = 100
n_test = 100
cluster_std = 0.7
n_xp = 100
eps = 1e-10

def generate_data(d, n_train, n_test, cluster_std):
    X, y = sklearn.datasets.make_blobs(n_train + n_test, n_features=d, centers=np.array([np.zeros(d), np.ones(d)]), cluster_std=cluster_std)
    y = 2 * y - 1
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=n_test, train_size=n_train)
    return X_train, X_test, y_train, y_test

@pytest.mark.parametrize("solver", ["entropic_torch_pre", "entropic_torch_post"])
@pytest.mark.parametrize("rho", [1e-3, 1e-2, 1e-1])
def test_entropic_solvers(solver, rho):
    model = skwdro.linear_models.LogisticRegression(rho=rho, solver_reg=eps, solver=solver, kappa=1e5, cost_power=1, n_zeta_samples=100)
    X_train, X_test, y_train, y_test = generate_data(d, n_train, n_test, cluster_std)
    model.fit(X_train, y_train)
    coef_entr, intercept_entr, robust_loss_entr = model.coef_, model.intercept_, model.robust_loss_

    model = skwdro.linear_models.LogisticRegression(rho=rho, solver_reg=eps, solver="dedicated", kappa=1e5, cost_power=1, n_zeta_samples=100)
    X_train, X_test, y_train, y_test = generate_data(d, n_train, n_test, cluster_std)
    model.fit(X_train, y_train)
    coef_dedi, intercept_dedi, robust_loss_dedi = model.coef_, model.intercept_, model.robust_loss_

    assert np.allclose(robust_loss_entr, robust_loss_dedi)
    # assert np.allclose(coef_entr, coef_dedi)
    # assert np.allclose(intercept_entr, intercept_dedi)




