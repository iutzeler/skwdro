from typing import Optional, Union
import torch
import torch as pt

from skwdro.solvers.oracle_torch import _DualLoss

from skwdro.solvers.result import wrap_solver_result, SolverResult
from skwdro.solvers.utils import *
from skwdro.base.problems import EmpiricalDistributionWithoutLabels, WDROProblem

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

    optimizer = loss.optimizer

    if loss.presample:
        losses = optim_presample(30, optimizer, xi, xi_labels, loss)
        np.save(
                "test_pre.npy",
                losses
            )
    else:
        losses = optim_postsample(1000, optimizer, xi, xi_labels, loss)
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
        optimizer: pt.optim.Optimizer,
        xi: pt.Tensor,
        xi_labels: Optional[pt.Tensor],
        loss: _DualLoss):
    losses = []

    for _ in range(n_iter):
        optimizer.zero_grad()
        objective = loss(xi, xi_labels)
        objective.backward()
        optimizer.step()
        # print(f"{loss._lam.grad=}")
        with torch.no_grad():
            if loss._lam < 0:
                loss._lam *= 0.
                loss._lam += 1e-10
        losses.append(objective.item())

    return losses
