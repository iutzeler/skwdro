"""
===================
Neural Network
===================

An example of Neural Network.
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
    parser.add_argument("--plot", action="store_true", help="plot one dimensional data")
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
    scaling = 1e-2
    x0 = rng.standard_normal(dim_data)

    s = 1.3
    f = lambda x : x.dot(x0)
    f = lambda x : np.sin(x*x0*2).flatten()
    f = lambda x : (np.sin(x/s)+0.1*np.sin(100*x/s)).flatten()

    X_train = rng.standard_normal((n_data, dim_data))
    X_train = np.array([np.pi/100*i + np.pi/200 for i in range(-n_data, n_data)])[:, None]*s
    y_train = f(X_train) #+  0.5*rng.standard_normal(n_data)

    X_test = rng.standard_normal((n_data*10, dim_data))
    X_test = np.array([np.pi*2/100 * i for i in range(-n_data//2, n_data//2)])[:, None]*s
    y_test = f(X_test)

    #X_train, X_test, y_train, y_test = train_test_split(X,y)

    def Loss(yhat, y):
        if yhat.shape != y.shape:
            # print(f"for info, yhat.shape != y.shape: {yhat.shape} != {y.shape}")
            pass
        return np.square(yhat.flatten()-y.flatten()).sum()/len(yhat)


    ly1 = rng.uniform(-scaling, scaling, size=(n_neurons, dim_data+1))
    ly2 = rng.uniform(-scaling, scaling, size=(1, n_neurons))

    # Standard pytorch shallow network
    print("Fitting Standard pytorch shallow network")
    mdl = ShallowNet(dim_data, n_neurons, fit_intercept=True, ly1=ly1, ly2=ly2, lr=1e-2)
    mdl.fit(X_train, y_train, iterlimit=20000, timelimit=None, gradcheck=False) # deter
    #mdl.fit(X_train, y_train, iterlimit=1000, timelimit=20, gradcheck=True) # will differ
    gd_fit_train, gd_fit_test = Loss(mdl.predict(X_train), y_train), Loss(mdl.predict(X_test), y_test)
    print(f"relu Shallownet Pytorch: TRAIN {gd_fit_train:.6f} - TEST {gd_fit_test:.6f}")
    ly1_gd, ly2_gd = mdl.params() # (n_neurons, dim_data+1), (1, n_neurons)

    if False: #TODO remove
        lact_gd = np.sort(np.array([-w2/w1 for w1, w2 in ly1_gd]))
        lact_gd = np.array([[x, 1] for x in lact_gd])
        points_gd = (np.maximum(lact_gd@ly1_gd.T, 0) @ ly2_gd.T).flatten()
        fig = plt.figure(figsize=(10,4))
        ax = fig.add_subplot()
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.7, linewidth=0.4)
        ax.axvline(x=0, color="black", linestyle="-", alpha=0.7, linewidth=0.4)
        ax.grid(True, alpha=0.2)
        ax.scatter(X_train.flatten(), y_train.flatten(), marker="o", color="red", s=80, alpha=1, label="train data")
        ax.plot(lact_gd[:, 0], points_gd, linestyle="-", marker="x", label="gd output")
        X_test_sort = np.sort(X_test.flatten())
        y_test_sort = f(X_test_sort)
        ax.plot(X_test_sort,y_test_sort, linestyle="-", marker="|", label="test data")
        ax.set_xlim(X_train[0].flatten(), X_train[-1].flatten())
        ax.set_ylim(-1, 1)
        plt.legend()
        plt.show()
        sys.exit(0)

    # Skwdro robust shallow network
    print("Skwdro robust shallow network")
    mdl = RobustShallowNet(rho=0.1,solver="entropic_torch",fit_intercept=True, n_neurons=n_neurons, ly1=ly1, ly2=ly2)
    mdl.fit(X_train, y_train)
    ro_fit_train, ro_fit_test = Loss(mdl.predict(X_train), y_train), Loss(mdl.predict(X_test), y_test)
    print(f"Skwdro TRAIN {ro_fit_train:.6f} - TEST {ro_fit_test:.6f}")
    ly1_ro, ly2_ro = mdl.params() # (n_neurons, dim_data+1), (1, n_neurons)

    # Sklearn linear regression
    mdl = LinearRegression()
    mdl.fit(X_train, y_train)
    fit_train, fit_test = Loss(mdl.predict(X_train), y_train), Loss(mdl.predict(X_test), y_test)
    print(f"sklearn linear: TRAIN {fit_train:.6f} - TEST {fit_test:.6f}")

    # Sanity check
    X_train_b = np.array([[x, 1] for x in X_train.flatten()])
    points = np.maximum(X_train_b@ly1_gd.T, 0) @ ly2_gd.T
    gd_fit_train_2 = np.linalg.norm(points.flatten()-y_train)**2/len(y_train)
    if abs(gd_fit_train - gd_fit_train_2)/gd_fit_train_2 > 1e-5: # assert model to ly1/ly2 is correct
        print("(true gd_fit_train_2)", gd_fit_train, "!=", gd_fit_train_2)

    ## Plotting 1-dimensional data
    if dim_data!= 1 and not args.plot:
        sys.exit(0)
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
    ax.scatter(X_train.flatten(), y_train.flatten(), marker="o", color="red", s=80, alpha=1, label="train data")
    ax.plot(lact_gd[:, 0], points_gd, linestyle="-", marker="x", label="gd output")
    ax.plot(lact_ro[:, 0], points_ro, linestyle="-", marker="x", label="ro output")
    ax.text(0.05, 0.9, f"Test loss\ngd:{gd_fit_test:.5f}\nro: {ro_fit_test:.5f}", transform=ax.transAxes)
    X_test_sort = np.sort(X_test.flatten())
    y_test_sort = f(X_test_sort)
    ax.plot(X_test_sort,y_test_sort, linestyle="-", marker="|", label="test data")
    #ax.set_xlim(-10, 10)
    #ax.set_ylim(-10, 10)

    ax.set_xlim(X_train[0].flatten(), X_train[-1].flatten())
    ax.set_ylim(-1, 1)
    plt.legend()
    plt.show()
