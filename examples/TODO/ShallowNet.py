"""
===================
Neural Network
===================

An example of Neural Network.
"""

import argparse
import random
from datetime import datetime

import numpy as np
import torch as pt
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from skwdro.neural_network import ShallowNet as RobustShallowNet

class AuxNet(nn.Module):
    def __init__(self, d, n_neurons, fit_intercept, ly1=None, ly2=None):
        super(AuxNet, self).__init__()
        self.linear1 = nn.Linear(d, n_neurons, bias=fit_intercept) # d -> n_neurons
        # linear1.weight: tensor(n_neurons, d)
        # linear1.bias: tensor(n_neurons, )
        # linear2.weight: tensor(1, n_neurons)
        self.linear2 = nn.Linear(n_neurons, 1, bias=False) # n_neurons -> 1

        dtype, device = pt.float32, "cpu"
        # Use the given neurons 
        if ly1 is not None and ly2 is not None:
            self.linear1.weight.data = pt.tensor(ly1[:, :-1], dtype=dtype, device=device, requires_grad=True)
            self.linear1.bias.data = pt.tensor(ly1[:, -1:].flatten(), dtype=dtype, device=device, requires_grad=True)
            self.linear2.weight.data = pt.tensor(ly2, dtype=dtype, device=device, requires_grad=True)

    def forward(self, x):
        return self.linear2(pt.relu(self.linear1(x)))

class ShallowNet:
    def __init__(self, d, n_neurons, fit_intercept, ly1=None, ly2=None):
        self.nn = AuxNet(d, n_neurons, fit_intercept, ly1=ly1, ly2=ly2)
        self.optimizer = pt.optim.AdamW(self.nn.parameters(), lr=1e-2, weight_decay=0)
        self.loss_fn = nn.MSELoss()

    def fit(self, X, Y):
        X = pt.tensor(X, dtype=pt.float32, device="cpu")
        Y = pt.tensor(Y, dtype=pt.float32, device="cpu")
        for i in range(1000):
            self.optimizer.zero_grad()
            loss = self.loss_fn(self.nn(X).flatten(), Y.flatten())
            loss.backward()
            self.optimizer.step()

    def predict(self, X):
        X = pt.tensor(X, dtype=pt.float32, device="cpu")
        return self.nn(X).cpu().detach().numpy().flatten()

    def params(self):
        ly1nb = self.nn.linear1.weight.data.cpu().detach().numpy()
        ly1b = self.nn.linear1.bias.data.cpu().detach().numpy()
        ly1 = np.hstack((ly1nb, ly1b[:,None]))
        ly2 = self.nn.linear2.weight.data.cpu().detach().numpy()
        return np.array(ly1), np.array(ly2) # otherwise it's just a pointer to memory...

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=None, type=int, help="if set, will use a fixed seed instead of a random one each run")
    args = parser.parse_args()

    if args.seed is None:
        seed = random.seed(datetime.now().timestamp()) # better than time.time()
        pt.use_deterministic_algorithms(False)
    else:
        seed = args.seed
        pt.use_deterministic_algorithms(True)

    rng = np.random.default_rng(seed)
    np.random.seed(seed) # should be used by sklearn methods by default

    dim_data, n_data, n_neurons = 5, 10, 10
    scaling = 1e-2*10

    ly1 = rng.uniform(-scaling, scaling, size=(n_neurons, dim_data+1))
    ly2 = rng.uniform(-scaling, scaling, size=(1, n_neurons))

    x0 = rng.standard_normal(dim_data)
    X_train = rng.standard_normal((n_data, dim_data))
    y_train = X_train.dot(x0) +  0.01*rng.standard_normal(n_data)

    X_test = rng.standard_normal((n_data*10, dim_data))
    y_test = X_test.dot(x0)# +  0.1*rng.standard_normal(n_data*10) # a lot of test data -> more accurate loss

    #X_train, X_test, y_train, y_test = train_test_split(X,y)

    def Loss(yhat, y):
        if yhat.shape != y.shape:
            # print(f"for info, yhat.shape != y.shape: {yhat.shape} != {y.shape}")
            pass
        return np.square(yhat.flatten()-y.flatten()).sum()/len(yhat)

    mdl = ShallowNet(dim_data, n_neurons, fit_intercept=True, ly1=ly1, ly2=ly2)
    mdl.fit(X_train, y_train)
    fit_train, fit_test = Loss(mdl.predict(X_train), y_train), Loss(mdl.predict(X_test), y_test)
    print(f"relu Shallownet Pytorch: TRAIN {fit_train:.4f} - TEST {fit_test:.4f}")
    ly1_gd, ly2_gd = mdl.params() # (n_neurons, data_dim+1), (1, n_neurons)

    #mdl = RobustShallowNet(rho=0.01,solver="entropic_torch",fit_intercept=True, n_neurons=n_neurons)
    mdl = RobustShallowNet(rho=0.01,solver="entropic_torch",fit_intercept=True, n_neurons=n_neurons, ly1=ly1, ly2=ly2)
    mdl.fit(X_train, y_train)
    fit_train, fit_test = Loss(mdl.predict(X_train), y_train), Loss(mdl.predict(X_test), y_test)
    print(f"Skwdro TRAIN {fit_train:.4f} - TEST {fit_test:.4f}")
    ly1_ro, ly2_ro = mdl.params() # (n_neurons, data_dim+1), (1, n_neurons)

    mdl = LinearRegression()
    mdl.fit(X_train, y_train)
    fit_train, fit_test = Loss(mdl.predict(X_train), y_train), Loss(mdl.predict(X_test), y_test)
    print(f"sklearn linear: TRAIN {fit_train:.4f} - TEST {fit_test:.4f}")
