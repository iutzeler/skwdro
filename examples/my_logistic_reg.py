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

torch.multiprocessing.set_sharing_strategy("file_system")

class LogisticRegLoss(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.linear = nn.Linear(d, 1, bias=True)
        self.logsigmoid = nn.LogSigmoid()

    def forward(self, x, y):
        b, = y.size()
        assert x.size() == (b, self.d)

        inner = y * self.linear(x).squeeze()
        assert inner.size() == (b,)

        return -self.logsigmoid(inner).mean()
    
    def adv_grad(self, x, y):
        b, = y.size()
        assert x.size() == (b, self.d)

        inner = y * self.linear(x).squeeze()
        assert inner.size() == (b,)

        weight = self.linear.weight.squeeze()
        assert weight.size() == (self.d,)

        adv_gradient = - y[:, None] * weight[None,:] * F.sigmoid(-inner)[:, None]
        assert adv_gradient.size() == (b, self.d) 
        return adv_gradient

    def perturbed_data(self, x, y, sigma, step):
        b, d = x.size()
        assert y.size() == (b,)

        distr = distributions.Normal(0, 1)
        noise = distr.rsample(sample_shape=(b, d))
        assert noise.size() == (b, d)

        adv_grad = self.adv_grad(x, y)
        assert adv_grad.size() == (b, d)


        perturbed_x = x + step * adv_grad + sigma * noise
        assert perturbed_x.size() == (b, d)

        return perturbed_x, y

class RobustLogisticLogLoss(nn.Module):
    def __init__(self, d, rho, smoothing=1e-2):
        super().__init__()
        self.logisticregloss = LogisticRegLoss(d)
        self.rho = rho
        self.smoothing = smoothing

    def forward(self, x, y):
        if self.training:
            smoothednorm = torch.sqrt(self.smoothing**2 + torch.linalg.norm(self.logisticregloss.linear.weight, 2)**2)
            return self.logisticregloss(x, y) + self.rho * smoothednorm
        else:
            return self.logisticregloss(x, y)

    def adv_grad(self, x, y):
        return self.logisticregloss.adv_grad(x, y)

    def perturbed_data(self, x, y, sigma, step):
        return self.logisticregloss.perturbed_data(x, y, sigma, step)

def generate_data(d, n_train, n_test, cluster_std):
    X, y = sklearn.datasets.make_blobs(n_train + n_test, n_features=d, centers=np.array([np.zeros(d), np.ones(d)]), cluster_std=cluster_std)
    y = 2 * y - 1
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=n_test, train_size=n_train)
    return torch.from_numpy(X_train).to(torch.float32), torch.from_numpy(X_test).to(torch.float32), torch.from_numpy(y_train).to(torch.float32), torch.from_numpy(y_test).to(torch.float32)

def train(model, x, y):
    model.train()

    def closure():
        opt.zero_grad()
        loss = model(x, y)
        loss.backward()
        return loss

    opt = optim.LBFGS(model.parameters(), max_iter=15, line_search_fn='strong_wolfe')
    opt.step(closure)
    return closure().item()

def eval_perf(model, eval_train, X_test, y_test, pert_sigma, pert_step):
    model.eval()
    X_pert, y_pert = model.perturbed_data(X_test, y_test, pert_sigma, pert_step)
    eval_test  = model.forward(X_test,  y_test).item()
    eval_pert  = model.forward(X_pert,  y_pert).item()
    return eval_train, eval_test, eval_pert

def plot_metric(metric, name, color, bins, alpha=1):
    _, bins, _ = plt.hist(metric, bins=bins, density=True, label=name, alpha=alpha, edgecolor="w")
    return bins

# From https://github.com/nschloe/tikzplotlib/issues/557
def tikzplotlib_fix_ncols(obj):
    """
        workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)


def plot_metrics(robust, metrics, save_tex=False):
    train = [m[0] for m in metrics]
    test  = [m[1] for m in metrics]
    pert  = [m[2] for m in metrics]
    #bins = plot_metric(train, "Train", "blue", 100)
    #plot_metric(test, "Test", "orange", bins, alpha=0.3)
    #plot_metric(pert, "Pert", "red", bins)
    #data={'Train':train, 'Test':test, 'Pert':pert}
    #sns.histplot(data=[train, test, pert], bins=50, stat="probability", log_scale=False, kde="True")
    bandwidth=1e-2
    ktrain = KernelDensity(bandwidth=bandwidth).fit(np.array(train).reshape(-1, 1))
    ktest  = KernelDensity(bandwidth=bandwidth).fit(np.array(test).reshape(-1, 1))
    kpert  = KernelDensity(bandwidth=bandwidth).fit(np.array(pert).reshape(-1, 1))
    x = np.linspace(0., 0.2, 100).reshape(-1,1)
    plt.plot(x, np.exp(ktrain.score_samples(x)), label="Robust loss on train set" if robust else "Loss on train set", color="C1")
    plt.plot(x, np.exp(ktest.score_samples(x)), label="Loss on test set", color="C2")
    plt.plot(x, np.exp(kpert.score_samples(x)), label="Loss on perturbed test set", color="C0")
    if robust:
        plt.title("Smoothed histogram of losses with the WDRO predictor")
    else:
        plt.title("Smoothed histogram of losses with the ERM predictor")
    plt.legend()
    if not save_tex:
        plt.show()
    else:
        fig = plt.gcf()
        tikzplotlib_fix_ncols(fig)
        filename = "./robust_hist.tex" if robust else "./erm_hist.tex"
        tikzplotlib.save(filename, figure=fig)

d = 10
n_train = 100
n_test = 100
rho = 0.01
pert_step = 3*rho
pert_sigma = 0.
cluster_std = 0.7
n_xp = 100

def make_xp(robust):
    model = RobustLogisticLogLoss(d, rho) if robust else LogisticRegLoss(d)
    X_train, X_test, y_train, y_test = generate_data(d, n_train, n_test, cluster_std)
    eval_train = train(model, X_train, y_train)
    metrics = eval_perf(model, eval_train, X_test, y_test, pert_sigma, pert_step)
    #print(metrics, flush=True)
    return metrics

def main(robust, filename):
    start = time.time()
    #with mp.Pool(processes=10) as pool:
    #    res = pool.imap_unordered(make_xp, [robust for _ in range(n_xp)], chunksize=2)
    #    res = list(tqdm.tqdm(res,total=n_xp))
    res = (make_xp(robust) for _ in range(n_xp))
    res = list(tqdm.tqdm(res,total=n_xp))
    end = time.time()
    print("Total time = ", end - start)
    with open(filename, "wb") as file:
        pickle.dump(res, file)

def plot(robust, filename, save_tex=False):
    with open(filename, "rb") as file:
        res = pickle.load(file)
    plot_metrics(robust, res, save_tex=save_tex)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--robust", action="store_true")
    parser.add_argument("--compute", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--savetex", action="store_true")
    args = parser.parse_args()
    if args.robust:
        filename = "stored_data/logregrob.pck"
    else:
        filename = "stored_data/logreg.pck"
    if args.compute or args.all:
        main(args.robust, filename)
    if args.plot or args.all:
        plot(args.robust, filename, save_tex=args.savetex)

    
    








        




