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

import tqdm
import time
import pickle
import argparse

import skwdro
from skwdro.solvers.hybrid_sgd import HybridAdam

alt = False
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

def train(model, x, y, robust):
    model.train()
    grads = []
    times = []
    crits = []
    start = time.time()
    opt = optim.LBFGS(model.model.parameters(), line_search_fn="strong_wolfe")
    for _ in range(50):
        def closure():
            opt.zero_grad()
            loss = model.model.forward(x, y).mean()
            loss.backward()
            #opt.step()
            return loss
        opt.step(closure)

    if not robust:
        return model.model.forward(x, y).mean().item()


    # opt = optim.LBFGS(model.parameters(), line_search_fn=None, max_iter=1)
    opt = HybridAdam([{'params': model.model.parameters(), 'lr':1e-1}, {'params': [model.lbd], 'lr':1e0, 'non_neg':True}])
    for t in range(1000):
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
        #print(f"{loss=}")
        #print(f"{model.lbd.grad=}")
        grads.append(model.lbd.grad.item())
        times.append(time.time() - start)
        #print(f"{model.lbd=}")
        #print(f"{torch.linalg.norm(model.model.linear.weight.grad)=}")
        #print(f"{torch.linalg.norm(model.model.linear.bias.grad)=}\n")
        crit = sum([torch.linalg.norm(param.grad).item() for param in model.parameters()])
        crits.append(crit)
        if t == 0:
            crit_tol = tol * crit
        if t % 100 == 0:
            print(f"{crit=}")
        if crit < crit_tol:
            print(f"Break at {t=} with {crit=} / {crit_tol=}")
            break
        opt.step()
    if True:
        plt.figure()
        plt.plot(times, grads)
        plt.yscale('log')
        plt.title("Grad lbd")
        plt.show()
        plt.figure()
        plt.plot(times, crits)
        plt.yscale('log')
        plt.title("Grads")
        plt.show()
    return loss.item()

def eval_perf(model, eval_train, X_test, y_test, pert_sigma, pert_step):
    model.eval()
    pert_model = model.model
    X_pert, y_pert = pert_model.perturbed_data(X_test, y_test, pert_sigma, pert_step)
    eval_test  = pert_model.forward(X_test,  y_test).mean().item()
    eval_pert  = pert_model.forward(X_pert,  y_pert).mean().item()
    return eval_train, eval_test, eval_pert

def plot_metric(metric, name, color, bins, alpha=1):
    _, bins, _ = plt.hist(metric, bins=bins, density=True, label=name, alpha=alpha, edgecolor="w")
    return bins



def plot_metrics(robust, metrics):
    train = [m[0] for m in metrics]
    test  = [m[1] for m in metrics]
    pert  = [m[2] for m in metrics]
    plt.figure()
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
    plt.plot(x, np.exp(ktrain.score_samples(x)), label=f"Robust loss on train set" if robust else "Loss on train set")
    plt.plot(x, np.exp(ktest.score_samples(x)), label="Loss on test set")
    plt.plot(x, np.exp(kpert.score_samples(x)), label="Loss on perturbed test set")
    plt.ylim(ymax=30)
    if robust:
        plt.title("Smoothed histogram of losses with the WDRO predictor")
    else:
        plt.title("Smoothed histogram of losses with the ERM predictor")
    plt.legend()
    plt.show()

d = 10
n_train = 100
n_test = 100
rho = 0.02
pert_step = 3*rho
pert_sigma = 0.
cluster_std = 0.7
n_xp = 30
sigma=1e-3
eps=1e-8
p = 1
tol = 1/100

def make_xp(robust):
    # model = RobustLogisticLogLoss(d, rho) if robust else LogisticRegLoss(d)
    logistic = LogisticRegLoss(d)
    if alt:
        model = LangevinApproxDRO(logistic, rho, n_train, eps, sigma, sample_y=False, p=p, lr=1e-4, T=100, m_train=100, sigma_langevin=sigma)
    else:
        model = LangevinApproxDRO(logistic, rho, n_train, eps, sigma, sample_y=False, p=p, lr=1e-5, T=1000, m_train=100, sigma_langevin=sigma)
    X_train, X_test, y_train, y_test = generate_data(d, n_train, n_test, cluster_std)
    eval_train = train(model, X_train, y_train, robust)
    metrics = eval_perf(model, eval_train, X_test, y_test, pert_sigma, pert_step)
    print(metrics, flush=True)
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

def plot(robust, filename):
    with open(filename, "rb") as file:
        res = pickle.load(file)
    plot_metrics(robust, res)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--robust", action="store_true")
    parser.add_argument("--compute", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if args.robust:
        filename = "stored_data/logregrob.pck"
    else:
        filename = "stored_data/logreg.pck"
    if args.compute or args.all:
        main(args.robust, filename)
    if args.plot or args.all:
        plot(args.robust, filename)

    
    








        




