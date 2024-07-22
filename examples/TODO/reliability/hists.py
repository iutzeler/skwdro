"""
TODO
"""
import argparse
import wandb
import math

import numpy as np
import torch as pt

L2_REG = 1e-5 #Don't change

from examples.reliability.main import fit_estimator, fit_cvx, gen_data
from skwdro.operations_research._portfolio import Portfolio
# def fit_estimator(rho_norm, X, y, X_test, y_test, n_zeta, epsilon):
#     estimator = LogisticRegression(
#             rho=np.sqrt(2)*rho_norm,
#             l2_reg=L2_REG,
#             cost="t-NLC-2-2",
#             fit_intercept=True,
#             solver="entropic_torch",
#             solver_reg=epsilon,
#             sampler_reg=None,
#             n_zeta_samples=n_zeta,
#         )
#     torch.manual_seed(int(rho_norm*1e6))
#     estimator.fit(X, y)
#     robust_loss = estimator.robust_loss_
#     test_loss = estimator.problem_.loss.primal_loss(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)).mean()
#     return robust_loss, test_loss

# def fit_cvx(rho_norm, X, y, X_test, y_test, n_zeta, *_):
#     estimator = LogisticRegression(
#             rho=np.sqrt(2)*rho_norm,
#             l2_reg=L2_REG,
#             cost="n-NLC-2-2",
#             fit_intercept=True,
#             solver="dedicated",
#             solver_reg=None,
#             sampler_reg=None,
#             n_zeta_samples=n_zeta,
#         )
#     estimator.fit(X, y)
#     robust_loss = estimator.result_
#     lin = y_test * X_test.dot(estimator.coef_)
#     test_loss = (np.log1p(np.exp(-np.abs(lin))) - np.minimum(lin, 0.)).mean()
#     return robust_loss, test_loss


# def gen_data(d, n, sigma):
#     rng = np.random.default_rng()
# 
#     y = rng.binomial(1, 0.6, (n,))
#     y = 2 * y - 1
#     assert y.shape == (n,)
# 
#     X = y[:, None] + sigma * rng.standard_normal((n, d))
#     assert X.shape == (n, d)
# 
#     return X, y

def fit_portfolio(rho_norm, X, X_test, n_zeta, epsilon):
    estimator = Portfolio(
            rho=rho_norm,
            cost="t-NC-2-2",
            solver="entropic_torch",
            solver_reg=epsilon,
            n_zeta_samples=n_zeta,
            eta=10.,
            alpha=2e-1
        )
    estimator.fit(X)
    robust_loss = estimator.wdro_loss_.primal_loss(pt.tensor(X, dtype=pt.float32)).mean()
    test_loss = estimator.wdro_loss_.primal_loss(pt.tensor(X_test, dtype=pt.float32)).mean()
    return robust_loss, test_loss

def one_portfolio_run(rho, d, n_train, X_test, n_zeta, epsilon):
    X = np.random.randn(n_train, 1) * 1e-2 + (np.random.randn(n_train, d) * 2.5e-2 + 3e-2 * np.arange(d)[None, :])
    fitter = fit_portfolio
    tr_robust_loss, te_robust_loss = fitter(rho, X, X_test, n_zeta, epsilon)
    tr_loss, te_loss = fitter(0., X, X_test, 1, 1.)
    return tr_robust_loss, te_robust_loss, tr_loss, te_loss

def one_run(rho, d, n_train, sigma, X_test, y_test, n_zeta, epsilon, baseline: bool=False):
    X, y = gen_data(d, n_train, sigma)
    fitter = fit_cvx if baseline else fit_estimator
    tr_robust_loss, te_robust_loss = fitter(rho, X, y, X_test, y_test, n_zeta, epsilon)
    tr_loss, te_loss = fitter(0., X, y, X_test, y_test, 1, 1.)
    return tr_robust_loss, te_robust_loss, tr_loss, te_loss

def all_runs(run, rho, xp, d, n_train, n_test, n_zeta, sigma, epsilon, repeat, baseline: bool=False):
    if xp == "logreg":
        X_test, y_test = gen_data(d, n_test, sigma)
        for _ in range(repeat):
            trrl, terl, trl, tel = one_run(rho, d, n_train, sigma, X_test, y_test, n_zeta, epsilon, baseline)
            run.log({'train_loss': trl, 'test_loss': tel, '_robust_train_loss': trrl, '_robust_test_loss': terl})
    elif xp == "portfolio":
        X_test = np.random.randn(n_test, 1) * 1e-2 + (np.random.randn(n_test, d) * 2.5e-2 + 3e-2 * np.arange(d)[None, :])
        for _ in range(repeat):
            trrl, terl, trl, tel = one_portfolio_run(rho, d, n_train, X_test, n_zeta, epsilon)
            run.log({'train_loss': trl, 'test_loss': tel, '_robust_train_loss': trrl, '_robust_test_loss': terl})


IS = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xp", type=str, default="logreg")
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
    xp = args.xp
    config = vars(args)
    config.update({"IS":IS})
    del config["xp"]
    run = wandb.init(project="toolbox_hists_"+xp, config=config, name=str(config))
    assert run is not None, "Failed to run wandb init"
    all_runs(run, rho, xp, args.d, args.n_train, args.n_test, args.n_zeta, args.sigma, args.epsilon, args.repeat, args.cvx_wdro)

if __name__ == '__main__': main()
