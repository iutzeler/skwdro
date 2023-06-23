from typing import Optional, Union
import torch
import torch as pt

from skwdro.solvers.oracle_torch import _DualLoss

from skwdro.solvers.result import wrap_solver_result, SolverResult
from skwdro.solvers.utils import *
from skwdro.base.problems import EmpiricalDistributionWithoutLabels, WDROProblem

from scipy.optimize import line_search

import matplotlib.pyplot as plt
import numpy as np

@wrap_solver_result
def solve_dual(wdro_problem: WDROProblem, sigma: Union[float, pt.Tensor]=pt.tensor(.1)):

    rho = wdro_problem.rho
    if isinstance(rho, float): rho = pt.tensor(rho)
    if isinstance(sigma, float): sigma = pt.tensor(sigma)

    if wdro_problem.P.with_labels:
        xi = torch.Tensor(wdro_problem.P.samples_x)
        xi_labels  = torch.Tensor(wdro_problem.P.samples_y)
    else:
        xi = torch.Tensor(wdro_problem.P.samples)
        xi_labels = None

    loss = wdro_problem.loss
    assert loss is not None
    assert isinstance(loss, _DualLoss)
    if loss._sampler is None:
        loss.sampler = loss.default_sampler(xi, xi_labels, sigma)

    optimizer_main = torch.optim.Adam(loss.loss.parameters())
    optimizer_lam = torch.optim.LBFGS([loss._lam])

    if loss.presample:
        losses = optim_presample(30, optimizer, xi, xi_labels, loss)
        np.save(
                "test_pre.npy",
                losses
            )
    else:
        losses = optim_postsample(1000, optimizer_main, optimizer_lam ,  xi, xi_labels, loss)
        np.save(
                "test_post.npy",
                losses
            )

    theta = detach_tensor(loss.theta)
    intercept = loss.intercept
    if intercept is not None:
        intercept = detach_tensor(intercept)
    lambd = detach_tensor(loss.lam)
    return SolverResult(theta, intercept, lambd, losses[-1])

def optim_presample(
        n_iter: int,
        optimizer: pt.optim.Optimizer,
        xi: pt.Tensor,
        xi_labels: Optional[pt.Tensor],
        loss: _DualLoss):

    zeta, zeta_labels = loss.generate_zetas()

    def closure():
        optimizer.zero_grad()
        objective = loss(xi, xi_labels, zeta, zeta_labels)
        objective.backward()
        return objective

    losses = []
    for _ in range(n_iter):
        optimizer.step(closure)
        losses.append(loss(xi, xi_labels, zeta, zeta_labels).item())

    return losses

def optim_postsample(
        n_iter: int,
        optimizer_main: pt.optim.Optimizer,
        optimizer_lam: pt.optim.Optimizer,
        xi: pt.Tensor,
        xi_labels: Optional[pt.Tensor],
        loss: _DualLoss):
    losses = []

    for _ in range(n_iter):

        loss._lam.requires_grad = False
        loss._lam.data = torch.tensor([7.72])
        for param in loss.loss.parameters():
            param.requires_grad = True

        assert loss._lam.requires_grad == False

        optimizer_main.zero_grad()
        objective = loss(xi, xi_labels)
        objective.backward()
        optimizer_main.step()

    for _ in range(n_iter):

        loss._lam.requires_grad = False
        for param in loss.loss.parameters():
            param.requires_grad = True

        optimizer_main.zero_grad()
        objective = loss(xi, xi_labels)
        objective.backward()
        optimizer_main.step()

        loss._lam.requires_grad = True
        for param in loss.loss.parameters():
            param.requires_grad = False

        zeta, zeta_labels = loss.generate_zetas(loss.n_samples)
        

        def closure():
            optimizer_lam.zero_grad()
            objective = loss.compute_dual(xi, xi_labels, zeta, zeta_labels)
            objective.backward()
            print(f"{loss._lam.grad=}")
            return objective

        def deriv_closure(lam):
            loss._lam.requires_grad = False
            with torch.no_grad():
                loss._lam.data = torch.tensor(lam)
            loss._lam.requires_grad = True
            for param in loss.loss.parameters():
                param.requires_grad = False
            optimizer_lam.zero_grad()
            objective = loss.compute_dual(xi, xi_labels, zeta, zeta_labels)
            objective.backward()

            return loss._lam.grad.numpy()

        def zero_closure(lam):
            loss._lam.requires_grad = False
            with torch.no_grad():
                loss._lam.data = torch.tensor(lam)
                objective = loss.compute_dual(xi, xi_labels, zeta, zeta_labels)
            return objective.detach().item()

        X = np.geomspace(1e-10, 1e10, num=1000)
        Y = [zero_closure(x) for x in X]
        plt.figure()
        # plt.xscale('log')
        plt.loglog(X, Y)
        X = np.geomspace(1e-10, 1e15, num=1000)
        Y = [deriv_closure(x) for x in X]
        plt.figure()
        plt.xscale('log')
        plt.plot(X, Y)
        plt.show()



        # optimizer_lam.step(closure)
        lam0 = loss._lam.detach().item()
        gk = deriv_closure(lam0)
        res = line_search(zero_closure, deriv_closure, lam0, -gk, amax= lam0/gk if gk > 0 else float('inf'), maxiter=20)
        print(f"{res[0]=}")
        print(f"{gk=}")
        print(f"{res[4]=} {res[3]=}")
        print(f"newslope = {res[5]}")
        print(f"{lam0=}")
        print(f"{torch.tensor(lam0 - res[0] * gk)=}")
        loss._lam.data = torch.tensor(lam0 - res[0] * gk)
        print(f"{loss._lam=}")

        # print(f"{loss._lam.grad=}")
        with torch.no_grad():
            if loss._lam < 0 and False:
                loss._lam *= 0.
                loss._lam += 1e-10
        losses.append(objective.item())

    return losses
