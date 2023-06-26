import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributions as distributions

import numpy as np
    
import sklearn
import sklearn.datasets
import sklearn.model_selection
from sklearn.neighbors import KernelDensity

import seaborn as sns
import matplotlib.pyplot as plt
import tikzplotlib

import tqdm
import time
import pickle
import argparse

import skwdro
from skwdro.solvers.hybrid_sgd import HybridAdam

alt = True
if alt:
    from langevin_approx_dro_alt import LangevinApproxDRO
else:
    from langevin_approx_dro import LangevinApproxDRO


torch.multiprocessing.set_sharing_strategy("file_system")

class LogisticRegLoss(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.linear = nn.Linear(d, 1, bias=True)
        self.logsigmoid = nn.LogSigmoid()
        self.in_features = d

    def forward(self, x, y):
        # x, y = xi[:,:-1], xi[:,-1]
        b = y.size()
        assert x.size() == b + (self.d,), x.size()

        inner = y * self.linear(x).squeeze()
        assert inner.size() == b

        return -self.logsigmoid(inner) + 1e-3 * torch.linalg.norm(self.linear.weight)**2
    


def generate_data(d, n_train, n_test, cluster_std):
    X, y = sklearn.datasets.make_blobs(n_train + n_test, n_features=d, centers=np.array([np.zeros(d), np.ones(d)]), cluster_std=cluster_std)
    y = 2 * y - 1
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=n_test, train_size=n_train)
    return torch.from_numpy(X_train).to(torch.float32), torch.from_numpy(X_test).to(torch.float32), torch.from_numpy(y_train).to(torch.float32), torch.from_numpy(y_test).to(torch.float32)

def train(model, x, y):
    model.train()
    grads = []
    times = []
    start = time.time()
    opt = optim.Adam(model.model.parameters(), lr=1e-3)
    for _ in range(300):
        opt.zero_grad()
        loss = model.model.forward(x, y).mean()
        loss.backward()
        opt.step()


    # opt = optim.LBFGS(model.parameters(), line_search_fn=None, max_iter=1)
    opt = HybridAdam([{'params': model.model.parameters(), 'lr':1e-5}, {'params': [model.lbd], 'lr':1e0, 'non_neg':True}])
    for _ in range(500):
        if False:
            def closure():
                opt.zero_grad()
                loss = model.forward(x, y, reset=False).mean()
                loss.backward()
                return loss
            print("\nBefore BFGS")
            loss = model.forward(x, y, reset=True).mean()
            print(f"{loss=}")
            print(f"{model.lbd.grad=}")
            print(f"{model.lbd=}")
            opt.step(closure)
            loss = model.forward(x, y, reset=False).mean()
            print("After BFGS")
            print(f"{loss=}")
            print(f"{model.lbd.grad=}")
            print(f"{model.lbd=}\n")

        
        def zero_closure(lbd):
            with torch.no_grad():
                model.lbd.requires_grad = False
                model.lbd.data = torch.tensor([lbd])

                loss = model.forward(x, y, reset=False).mean()

                model.lbd.requires_grad = True
            return loss

        #X = np.geomspace(1e-3, 10, num=100)
        #Y = [zero_closure(x) for x in X]
        #plt.figure()
        #plt.plot(X, Y)
        #plt.xscale('log')
        #plt.show()
        opt.zero_grad()
        loss = model.forward(x, y).mean()
        loss.backward()
        print("\nBefore Step")
        print(f"{loss=}")
        print(f"{model.lbd.grad=}")
        grads.append(model.lbd.grad.item())
        times.append(time.time() - start)
        print(f"{model.lbd=}")
        opt.step()

    plt.figure()
    plt.plot(times, grads)
    plt.show()
    return loss.item().detach()






d = 10
n_train = 100
n_test = 100
rho = 0.01
pert_step = 3*rho
pert_sigma = 0.
cluster_std = 0.7
n_xp = 100
sigma=1e-3
eps=1e-10
p = 1

def make_xp(robust):
    logistic = LogisticRegLoss(d)
    if alt:
        model = LangevinApproxDRO(logistic, rho, n_train, eps, sigma, sample_y=False, p=p, lr=1e-5, T=200, m_train=200, sigma_langevin=sigma)
    else:
        model = LangevinApproxDRO(logistic, rho, n_train, eps, sigma, sample_y=False, p=p, lr=1e-5, T=1000, m_train=100, sigma_langevin=sigma)
    X_train, X_test, y_train, y_test = generate_data(d, n_train, n_test, cluster_std)
    eval_train = train(model, X_train, y_train)
    return None

def main(robust, filename):
    start = time.time()
    #with mp.Pool(processes=10) as pool:
    #    res = pool.imap_unordered(make_xp, [robust for _ in range(n_xp)], chunksize=2)
    #    res = list(tqdm.tqdm(res,total=n_xp))
    res = make_xp(robust)
    end = time.time()
    print("Total time = ", end - start)

if __name__ == "__main__":
    robust = True
    filname = None
    main(robust, filename)
    
    








        




