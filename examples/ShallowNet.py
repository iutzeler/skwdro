"""
===================
Neural Network
===================

An example of Neural Network.
"""

import numpy as np
import torch as pt
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from skwdro.linear_models import ShallowNet as RobustShallowNet



class AuxNet(nn.Module):
    def __init__(self, d, nbneurone, fit_intercept):
        super(AuxNet, self).__init__()
        self.linear1 = nn.Linear(d, nbneurone, bias=fit_intercept) # d -> nbneurone
        self.linear2 = nn.Linear(nbneurone, 1, bias=fit_intercept) # nbneurone -> 1

    def forward(self, x):
        return self.linear2(pt.relu(self.linear1(x)))

class ShallowNet:
    def __init__(self, d, nbneurone, fit_intercept):
        self.nn = AuxNet(d, nbneurone, fit_intercept)
        self.optimizer = pt.optim.AdamW(self.nn.parameters(), lr=1e-2, weight_decay=1e-2)
        self.loss_fn = nn.MSELoss()

    def fit(self, X, Y):
        X = pt.tensor(X, dtype=pt.float32, device="cpu")
        Y = pt.tensor(Y, dtype=pt.float32, device="cpu")
        for i in range(100):
            self.optimizer.zero_grad()
            loss = self.loss_fn(self.nn(X).flatten(), Y.flatten())
            loss.backward()
            self.optimizer.step()
            #print(loss.item())

    def predict(self, X):
        X = pt.tensor(X, dtype=pt.float32, device="cpu")
        return self.nn(X).cpu().detach().numpy().flatten()

if __name__ == '__main__':
    pt.use_deterministic_algorithms(True)
    seed = 5
    rng = np.random.default_rng(seed) # do not use np.random, see https://numpy.org/doc/stable/reference/random/generator.html#distributions

    d = 10
    m = 100
    nbneurone=100

    x0 = rng.standard_normal(d)
    X_train = rng.standard_normal((m, d))
    y_train = X_train.dot(x0) +  0.2*rng.standard_normal(m)

    X_test = rng.standard_normal((m*10, d))
    y_test = X_test.dot(x0) +  0.2*rng.standard_normal(m*10) # a lot of test data -> more accurate loss

    #X_train, X_test, y_train, y_test = train_test_split(X,y)

    def Loss(yhat, y):
        if yhat.shape != y.shape:
            # print(f"for info, yhat.shape != y.shape: {yhat.shape} != {y.shape}")
            pass
        return np.square(yhat.flatten()-y.flatten()).sum()/len(yhat)

    mdl = ShallowNet(d, nbneurone, fit_intercept=True)
    mdl.fit(X_train, y_train)
    fit_train, fit_test = Loss(mdl.predict(X_train), y_train), Loss(mdl.predict(X_test), y_test)
    print(f"relu Shallownet Pytorch: TRAIN {fit_train:.4f} - TEST {fit_test:.4f}")

    for solver in ["entropic", "entropic_torch", "dedicated"]:
        mdl = RobustShallowNet(rho=0.1,solver="entropic_torch",fit_intercept=True, nbneurone=nbneurone)
        mdl.fit(X_train, y_train)
        fit_train, fit_test = Loss(mdl.predict(X_train), y_train), Loss(mdl.predict(X_test), y_test)
        print(f"Skwdro[{solver}] TRAIN {fit_train:.4f} - TEST {fit_test:.4f}")

    mdl = LinearRegression()
    mdl.fit(X_train, y_train)
    fit_train, fit_test = Loss(mdl.predict(X_train), y_train), Loss(mdl.predict(X_test), y_test)
    print(f"sklearn linear: TRAIN {fit_train:.4f} - TEST {fit_test:.4f}")
