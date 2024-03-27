"""
TODO
"""
import numpy as np
import argparse
import wandb
import math
import torch

from skwdro.linear_models import LogisticRegression

L2_REG = 1e-5 #Don't change

def fit_estimator(rho_norm, X, y, X_test, y_test, n_zeta, epsilon):
    estimator = LogisticRegression(
            rho=np.sqrt(2)*rho_norm,
            l2_reg=L2_REG,
            cost="t-NLC-2-2",
            fit_intercept=True,
            solver="entropic_torch",
            solver_reg=epsilon,
            sampler_reg=None,
            n_zeta_samples=n_zeta,
        )
    torch.manual_seed(int(rho_norm*1e6))
    estimator.fit(X, y)
    robust_loss = estimator.wdro_loss_.primal_loss(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)).mean()
    test_loss = estimator.wdro_loss_.primal_loss(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)).mean()
    return robust_loss, test_loss

def fit_cvx(rho_norm, X, y, X_test, y_test, n_zeta, *_):
    estimator = LogisticRegression(
            rho=np.sqrt(2)*rho_norm,
            l2_reg=L2_REG,
            cost="n-NLC-2-2",
            fit_intercept=True,
            solver="dedicated",
            solver_reg=None,
            sampler_reg=None,
            n_zeta_samples=n_zeta,
        )
    estimator.fit(X, y)
    robust_loss = estimator.result_
    lin = y_test * X_test.dot(estimator.coef_)
    test_loss = (np.log1p(np.exp(-np.abs(lin))) - np.minimum(lin, 0.)).mean()
    return robust_loss, test_loss


def gen_data(d, n, sigma):
    rng = np.random.default_rng()

    y = rng.binomial(1, 0.6, (n,))
    y = 2 * y - 1
    assert y.shape == (n,)

    X = y[:, None] + sigma * rng.standard_normal((n, d))
    assert X.shape == (n, d)

    return X, y

def one_run_reliability(rho, d, n_train, sigma, X_test, y_test, n_zeta, epsilon, baseline: bool=False):
    X, y = gen_data(d, n_train, sigma)
    fitter = fit_cvx if baseline else fit_estimator
    robust_loss, test_loss = fitter(rho, X, y, X_test, y_test, n_zeta, epsilon)
    return int((robust_loss >= test_loss).item())

def all_runs_reliability(rho, d, n_train, n_test, n_zeta, sigma, epsilon, repeat, baseline: bool=False):
    X_test, y_test = gen_data(d, n_test, sigma)
    avg = 0.
    xps = np.array([one_run_reliability(rho, d, n_train, sigma, X_test, y_test, n_zeta, epsilon, baseline) for _ in range(repeat)])
    avg = xps.mean()
    std = xps.std(ddof=1)
    del xps
    # for _ in range(repeat):
    #     avg += one_run_reliability(rho, d, n_train, sigma, X_test, y_test)
    # avg /= repeat
    # std = np.sqrt(avg * (1 - 1 / repeat))
    return avg, std

IS = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=3)
    parser.add_argument("--sigma", type=float, default=1.)
    parser.add_argument("--n_train", type=int, default=int(1e3))
    parser.add_argument("--n_zeta", type=int, default=int(1e1))
    parser.add_argument("--n_test", type=int, default=int(1e5))
    parser.add_argument("--logrho", type=float, default=math.log10(0.02))
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("-e", "--epsilon", type=float, default=1e-6)
    parser.add_argument("-c", "--cvx_wdro", action='store_true', help="Launch with specific cvxopt solver (baseline)")
    args = parser.parse_args()
    rho = math.pow(10, args.logrho)
    config = vars(args)
    config.update({"IS":IS})
    run = wandb.init(project="toolbox_eps", config=config, name=str(config))
    assert run is not None, "Failed to run wandb init"
    avg, sdev = all_runs_reliability(rho, args.d, args.n_train, args.n_test, args.n_zeta, args.sigma, args.epsilon, args.repeat, args.cvx_wdro)
    run.log({"avg_rel": avg, "sdev_rel": sdev})

