"""
===================
Neural Network
===================

An example of Neural Network.
"""

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from skwdro.linear_models import ShallowNet as RobustShallowNet


if __name__ == '__main__':
    torch.use_deterministic_algorithms(True)
    seed = 4
    rng = np.random.default_rng(seed) # do not use np.random, see https://numpy.org/doc/stable/reference/random/generator.html#distributions

    d = 3
    m = 4
    nbneurone=7

    x0 = rng.standard_normal(d)
    X_train = rng.standard_normal((m, d))
    y_train = X_train.dot(x0) +  0.2*rng.standard_normal(m)

    X_test = rng.standard_normal((m, d))
    y_test = X_test.dot(x0) +  0.2*rng.standard_normal(m)

    #X_train, X_test, y_train, y_test = train_test_split(X,y)


    rob_lin = RobustShallowNet(rho=0.1,solver="entropic",fit_intercept=True, nbneurone=nbneurone)
    rob_lin.fit(X_train, y_train)
    y_pred_rob = rob_lin.predict(X_test)


    rob_lin2 = RobustShallowNet(rho=0.1,solver="entropic_torch",fit_intercept=True, nbneurone=nbneurone)
    rob_lin2.fit(X_train, y_train)
    y_pred_rob2 = rob_lin2.predict(X_test)

    rob_lin3 = RobustShallowNet(rho=0.1,solver="dedicated",fit_intercept=True, nbneurone=nbneurone)
    rob_lin3.fit(X_train, y_train)
    y_pred_rob3 = rob_lin3.predict(X_test)


    lin = LinearRegression()
    lin.fit(X_train, y_train)
    y_pred = lin.predict(X_test)




    print(f"sklearn error: {np.linalg.norm(y_pred-y_test)} " )
    print(f"skwdro error w/entropic: {np.linalg.norm(y_pred_rob-y_test)}")
    print(f"skwdro error w/torch: {np.linalg.norm(y_pred_rob2-y_test)}")
    print(f"skwdro error w/dedicated: {np.linalg.norm(y_pred_rob3-y_test)}")
