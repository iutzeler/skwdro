import numpy as np
import argparse
import wandb
import math
import torch

from skwdro.base.costs import NormLabelCost
from skwdro.linear_models import LogisticRegression
from skwdro.solvers.optim_cond import OptCond

L2_REG = 1e-5 #Don't change

def fit_estimator(rho_norm, X, y, X_test, y_test):
    estimator = LogisticRegression(
            rho=rho_norm,
            l2_reg=L2_REG,
            cost="t-NLC-2-2-1",
            fit_intercept=True,
            solver="entropic_torch",
            solver_reg=None,
            sampler_reg=None,
            n_zeta_samples=20,
        )
    estimator.fit(X, y)
    robust_loss = estimator.robust_loss_
    test_loss = estimator.problem_.loss.primal_loss(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)).mean()
    return robust_loss, test_loss

def gen_data(d, n, sigma):
    rng = np.random.default_rng()

    y = rng.binomial(1, 0.5, (n,))
    y = 2 * y - 1
    assert y.shape == (n,)

    X = y[:, None] + sigma * rng.standard_normal((n, d))
    assert X.shape == (n, d)

    return X, y

def one_run_reliability(rho, d, n_train, sigma, X_test, y_test):
    X, y = gen_data(d, n_train, sigma)
    robust_loss, test_loss = fit_estimator(rho, X, y, X_test, y_test)
    return int((robust_loss >= test_loss).item())

def all_runs_reliability(rho, d, n_train, n_test, sigma, repeat):
    X_test, y_test = gen_data(d, n_test, sigma)
    avg = 0.
    for _ in range(repeat):
        avg += one_run_reliability(rho, d, n_train, sigma, X_test, y_test)
    avg /= repeat
    std = np.sqrt(avg * (1 - 1 / repeat))
    return avg, std

IS = True
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=3)
    parser.add_argument("--sigma", type=float, default=1.5)
    parser.add_argument("--n", type=int, default=int(1e3))
    parser.add_argument("--n_test", type=int, default=int(1e5))
    parser.add_argument("--logrho", type=float, default=math.log10(0.02))
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    rho = math.pow(10, args.logrho)
    config = vars(args)
    config.update({"IS":IS})
    run = wandb.init(project="toolbox", config=config, name=str(config))
    avg, sdev = all_runs_reliability(rho, args.d, args.n, args.n, args.sigma, args.repeat)
    run.log({"avg_rel": avg, "sdev_rel": sdev})

