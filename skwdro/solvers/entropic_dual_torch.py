from typing import Optional, Union
import torch
import torch as pt
import torch.optim as optim

import math
from skwdro.solvers.oracle_torch import _DualLoss

from skwdro.solvers.utils import *
from skwdro.base.problems import WDROProblem


def solve_dual(WDROProblem: WDROProblem, sigma: Union[float, pt.Tensor]=pt.tensor(.1), fit_intercept: bool=False):

    rho = WDROProblem.rho
    if isinstance(rho, float): rho = pt.tensor(rho)
    if isinstance(sigma, float): sigma = pt.tensor(sigma)

    NoLabels = WDROProblem.dLabel == 0

    if NoLabels:
        xi = torch.Tensor(WDROProblem.P.samples)
        xi_labels = None
    else:
        xi = torch.Tensor(WDROProblem.P.samplesX)
        xi_labels  = torch.Tensor(WDROProblem.P.samplesY)

    loss = WDROProblem.loss
    assert loss is not None
    if loss._sampler is None:
        loss.sampler = loss.default_sampler(xi, xi_labels, sigma)

    optimizer = loss.optimizer

    if loss.presample:
        np.save(
                "test_pre.npy",
                optim_presample(30, optimizer, xi, xi_labels, loss)
            )
    else:
        np.save(
                "test_post.npy",
                optim_postsample(1000, optimizer, xi, xi_labels, loss)
            )

    theta = detach_tensor(loss.theta)
    intercept = None if not fit_intercept else detach_tensor(loss.intercept)
    lambd = detach_tensor(loss.lam)
    return theta, intercept, lambd

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
        losses.append(objective.item())

    return losses
