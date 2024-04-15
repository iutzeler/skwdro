"""
===================
Plotting Neural Network with one dimensional data
===================

An example plot to show a case where robustness gives a much better test loss
in a regression setting.
"""

# stdlib
import argparse
import random
import time
from datetime import datetime
import sys

# graphics
import matplotlib.pyplot as plt
import matplotlib as mpl

# computations
import numpy as np
import torch as pt
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# skwdro
from skwdro.neural_network import ShallowNet as RobustShallowNet

# Standard shallow neural network
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
    def __init__(self, d, n_neurons, fit_intercept, ly1=None, ly2=None, lr=1e-2):
        self.nn = AuxNet(d, n_neurons, fit_intercept, ly1=ly1, ly2=ly2)
        self.optimizer = pt.optim.AdamW(self.nn.parameters(), lr=lr, weight_decay=0)
        #self.optimizer = pt.optim.SGD(self.nn.parameters(), lr=lr, weight_decay=0)
        self.loss_fn = nn.MSELoss()

    def fit(self, X, Y, iterlimit = 1000, timelimit=None, gradcheck=False):
        X = pt.tensor(X, dtype=pt.float32, device="cpu")
        Y = pt.tensor(Y, dtype=pt.float32, device="cpu")
        start = time.time()
        lastcheck = time.time()
        lastloss = 0
        i = 0
        while True:
            # stop decision
            if timelimit is None and i > iterlimit:
                break
            i += 1
            if timelimit is not None and time.time() - start > timelimit:
                break
            if gradcheck and (time.time() - lastcheck > 0.5):
                lastcheck = time.time()
                with pt.no_grad():
                    gradnorm = (pt.norm(self.nn.linear1.weight.grad) + pt.norm(self.nn.linear1.bias.grad) + pt.norm(self.nn.linear2.weight.grad)).item()
                    if gradnorm < 1e-7:
                        break
                    print(gradnorm, "at the end, loss=", lastloss, "ite", i)

            # standard optimisation loop
            self.optimizer.zero_grad()
            loss = self.loss_fn(self.nn(X).flatten(), Y.flatten())
            loss.backward()
            lastloss = loss.item()
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

    # seed stuff
    if args.seed is None:
        seed = random.seed(datetime.now().timestamp()) # better than time.time() (someone cares)
    else:
        seed = args.seed
    random.seed(seed)
    rng = np.random.default_rng(seed)
    np.random.seed(seed) # should be used by sklearn methods by default
    pt.manual_seed(seed) # skwdro currently use global generator
    pt.use_deterministic_algorithms(True)

    dim_data, n_data, n_neurons = 1, 10, 100
    x0 = rng.standard_normal(dim_data) # don't @ me

    s = 1.3
    f = lambda x : (np.sin(x/s)+0.1*np.sin(100*x/s)).flatten()

    X_train = rng.standard_normal((n_data, dim_data))
    X_train = np.array([np.pi/100*i + np.pi/200 for i in range(-n_data, n_data)])[:, None]*s
    y_train = f(X_train) #+  0.5*rng.standard_normal(n_data)

    X_test = rng.standard_normal((n_data*10, dim_data))
    X_test = np.array([np.pi*2/100 * i for i in range(-n_data//2, n_data//2)])[:, None]*s
    y_test = f(X_test)

    def Loss(yhat, y):
        if yhat.shape != y.shape:
            # print(f"for info, yhat.shape != y.shape: {yhat.shape} != {y.shape}")
            pass
        return np.square(yhat.flatten()-y.flatten()).sum()/len(yhat)

    # Initialiazing the two layers of the network, used for both optimizers
    scaling = 1e-2
    ly1 = rng.uniform(-scaling, scaling, size=(n_neurons, dim_data+1))
    ly2 = rng.uniform(-scaling, scaling, size=(1, n_neurons))

    # Standard pytorch shallow network
    print("Fitting the Standard pytorch shallow network")
    mdl = ShallowNet(dim_data, n_neurons, fit_intercept=True, ly1=ly1, ly2=ly2, lr=1e-2)
    mdl.fit(X_train, y_train, iterlimit=20000, timelimit=None, gradcheck=False) # deter
    #mdl.fit(X_train, y_train, iterlimit=1000, timelimit=20, gradcheck=True) # will differ
    gd_fit_train, gd_fit_test = Loss(mdl.predict(X_train), y_train), Loss(mdl.predict(X_test), y_test)
    print(f"relu Shallownet Pytorch: TRAIN {gd_fit_train:.6f} - TEST {gd_fit_test:.6f}")
    ly1_gd, ly2_gd = mdl.params() # (n_neurons, dim_data+1), (1, n_neurons)

    # Skwdro robust shallow network
    print("Fitting the Skwdro robust shallow network")
    mdl = RobustShallowNet(rho=0.1,solver="entropic_torch",fit_intercept=True, n_neurons=n_neurons, ly1=ly1, ly2=ly2)
    mdl.fit(X_train, y_train)
    ro_fit_train, ro_fit_test = Loss(mdl.predict(X_train), y_train), Loss(mdl.predict(X_test), y_test)
    print(f"Skwdro TRAIN {ro_fit_train:.6f} - TEST {ro_fit_test:.6f}")
    ly1_ro, ly2_ro = mdl.params() # (n_neurons, dim_data+1), (1, n_neurons)

    print("Preparing plot..")
    # Compute points to plot the output for GD and "robust GD"
    lact_gd = np.sort(np.array([-w2/w1 for w1, w2 in ly1_gd]))
    lact_gd = np.array([[x, 1] for x in lact_gd])
    points_gd = (np.maximum(lact_gd@ly1_gd.T, 0) @ ly2_gd.T).flatten()

    lact_ro = np.sort(np.array([-w2/w1 for w1, w2 in ly1_ro]))
    lact_ro = np.array([[x, 1] for x in lact_ro])
    points_ro = (np.maximum(lact_ro@ly1_ro.T, 0) @ ly2_ro.T).flatten()

    fig = plt.figure(figsize=(10,4))
    ax = fig.add_subplot()
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.7, linewidth=0.4)
    ax.axvline(x=0, color="black", linestyle="-", alpha=0.7, linewidth=0.4)
    ax.grid(True, alpha=0.2)
    # Data points
    ax.scatter(X_train.flatten(), y_train.flatten(), marker="o", color="red", s=80, alpha=1, label="train data")
    # We only need to plot points where the output change its slope.
    ax.plot(lact_gd[:, 0], points_gd, linestyle="-", marker="x", label="Adam optimizer")
    ax.plot(lact_ro[:, 0], points_ro, linestyle="-", marker="x", label="Wdro optimizer")
    ax.text(0.05, 0.9, f"Test loss\n{gd_fit_test:.5f} - Adam\n{ro_fit_test:.5f} - Wdro", transform=ax.transAxes)
    X_test_sort = np.sort(X_test.flatten()) # of course
    y_test_sort = f(X_test_sort)
    ax.plot(X_test_sort,y_test_sort, linestyle="-", marker="|", label="test data")
    ax.set_xlim(X_train[0].flatten(), X_train[-1].flatten())
    ax.set_ylim(-1, 1)
    plt.legend()
    plt.show()
