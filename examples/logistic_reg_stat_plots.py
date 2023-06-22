import torch
import torch.nn.functional as F
import torch.distributions as distributions
import torch.multiprocessing as mp

import numpy as np
    
import sklearn
import sklearn.datasets
import sklearn.model_selection
import sklearn.linear_model
from sklearn.neighbors import KernelDensity

import skwdro
import skwdro.linear_models

import seaborn as sns
import matplotlib.pyplot as plt
# sns.set_theme()

import tqdm
import time
import pickle
import argparse

mp.set_sharing_strategy("file_system")

at = lambda x : torch.as_tensor(x, dtype=torch.float32)
    
def adv_grad(coef, intercept, x, y):
    coef, intercept = at(coef).unsqueeze(0), at(intercept)
    assert coef.shape == (1, d)
    assert intercept.shape == (1,), intercept.shape

    x, y = at(x), at(y)
    b, = y.shape
    assert x.shape == (b, d)

    inner = y * F.linear(x, coef, intercept).squeeze()
    assert inner.shape == (b,)

    weight = coef.squeeze()
    assert weight.shape == (d,)

    adv_gradient = - y[:, None] * weight[None,:] * F.sigmoid(-inner)[:, None]
    assert adv_gradient.shape == (b, d) 
    return adv_gradient

def perturbed_data(coef, intercept, x, y, sigma, step):
    coef, intercept = at(coef), at(intercept)
    x, y = at(x), at(y)
    b, d = x.shape
    assert y.shape == (b,)

    distr = distributions.Normal(0, 1)
    noise = distr.rsample(sample_shape=(b, d))
    assert noise.shape == (b, d)

    adv_g = adv_grad(coef, intercept, x, y)
    assert adv_g.shape == (b, d)

    perturbed_x = x + step * adv_g + sigma * noise
    assert perturbed_x.shape == (b, d)

    assert not torch.isinf(perturbed_x).any()

    return perturbed_x, y

def generate_data(d, n_train, n_test, cluster_std):
    X, y = sklearn.datasets.make_blobs(n_train + n_test, n_features=d, centers=np.array([np.zeros(d), np.ones(d)]), cluster_std=cluster_std)
    y = 2 * y - 1
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=n_test, train_size=n_train)
    return X_train, X_test, y_train, y_test

def computeloss(coef, intercept, x, y):
    coef, intercept = at(coef).unsqueeze(0), at(intercept)
    assert coef.shape == (1, d)
    assert intercept.shape == (1,)

    x, y = at(x), at(y)
    b, = y.shape
    assert x.shape == (b, d)

    inner = y * F.linear(x, coef, intercept).squeeze()
    assert inner.shape == (b,), inner.shape

    return - F.logsigmoid(inner).mean().item()

def eval_perf(model, eval_train, X_test, y_test, pert_sigma, pert_step):
    X_pert, y_pert = perturbed_data(model.coef_, model.intercept_, X_test, y_test, pert_sigma, pert_step)
    eval_test  = computeloss(model.coef_, model.intercept_, at(X_test),  at(y_test))
    eval_pert  = computeloss(model.coef_, model.intercept_, at(X_pert),  at(y_pert))
    assert np.isclose(eval_test, model.logistic_loss(X_test, y_test)), (eval_test, model.logistic_loss(X_test, y_test))
    assert np.isclose(eval_pert, model.logistic_loss(X_pert, y_pert))
    assert eval_test != float('inf')
    assert eval_pert != float('inf')
    return eval_train, eval_test, eval_pert

def plot_metric(metric, name, color, bins, alpha=1):
    _, bins, _ = plt.hist(metric, bins=bins, density=True, label=name, alpha=alpha, edgecolor="w")
    return bins

def plot_metrics(robust, metrics, solver):
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
    plt.plot(x, np.exp(ktrain.score_samples(x)), label=f"Robust loss on train set with {solver=}" if robust else "Loss on train set")
    plt.plot(x, np.exp(ktest.score_samples(x)), label="Loss on test set")
    plt.plot(x, np.exp(kpert.score_samples(x)), label="Loss on perturbed test set")
    if robust:
        plt.title("Losses with the WDRO predictor")
    else:
        plt.title("Losses with the ERM predictor")
    plt.legend()
    plt.show()

d = 10
n_train = 100
n_test = 100
rho = 1e-2
pert_step = 3*rho
pert_sigma = 0.
cluster_std = 0.7
n_xp = 100
eps = 1e-3

def make_xp(robust, solver):
    new_rho = rho if robust else 0.
    model = skwdro.linear_models.LogisticRegression(rho=new_rho, solver_reg=eps, solver=solver, kappa=1, cost_power=1, n_zeta_samples=10)
    X_train, X_test, y_train, y_test = generate_data(d, n_train, n_test, cluster_std)
    model.fit(X_train, y_train)
    print(f"{model.dual_var_=}")
    eval_train = model.robust_loss_
    metrics = eval_perf(model, eval_train, X_test, y_test, pert_sigma, pert_step)
    print(metrics, flush=True)
    return metrics

def main(robust, filename, solver):
    start = time.time()
    make_xp(robust, solver)
    #with mp.Pool(processes=10) as pool:
    #    res = pool.imap_unordered(make_xp, [robust for _ in range(n_xp)], chunksize=2)
    #    res = list(tqdm.tqdm(res,total=n_xp))
    res = [make_xp(robust, solver) for _ in range(n_xp)]
    end = time.time()
    print("Total time = ", end - start)
    with open(filename, "wb") as file:
        pickle.dump(res, file)

def plot(robust, filename, solver):
    with open(filename, "rb") as file:
        res = pickle.load(file)
    plot_metrics(robust, res, solver)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--robust", action="store_true")
    parser.add_argument("--compute", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--solver", default="dedicated")
    args = parser.parse_args()
    if args.robust:
        filename = "stored_data/logregrob.pck"
    else:
        filename = "stored_data/logreg.pck"
    if args.compute or args.all:
        main(args.robust, filename, args.solver)
    if args.plot or args.all:
        plot(args.robust, filename, args.solver)

    
    








        




