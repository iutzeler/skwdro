from typing import List, Optional, Union

import numpy as np
import torch as pt

from skwdro.solvers.oracle_torch import _DualLoss

from skwdro.solvers.result import wrap_solver_result
from skwdro.solvers.utils import detach_tensor, interpret_steps_struct
from skwdro.base.problems import Distribution, WDROProblem
from skwdro.base.samplers.torch.base_samplers import BaseSampler

def extract_data(dist: Distribution):
    """
    Get torch tensors out of empirical distribution.

    Parameters
    ----------
    dist: Distribution
        Empirical distribution of data and optionally labels

    Returns
    -------
    xi: pt.Tensor
        data tensor
    xi_labels: Optional[pt.Tensor]
        label tensor if the distribution yields them, else ``None``

    Shapes
    ------
    xi: (m, d)
    xi_labels: None or (m, d')
    """
    if dist.with_labels:
        xi = pt.Tensor(dist.samples_x)
        xi_labels  = pt.Tensor(dist.samples_y)
        return xi, xi_labels
    else:
        xi = pt.Tensor(dist.samples)
        return xi, None


@wrap_solver_result
def solve_dual(wdro_problem: WDROProblem, seed: int, sigma_: Union[float, pt.Tensor]=pt.tensor(.1)):
    r""" Solve the dual problem with the loss-dependant grandient descent algorithm.

    Parameters
    ----------
    wdro_problem : WDROProblem
        Whole WDRO problem containing relevant parameters and data
    sigma_ : Union[float, pt.Tensor]
        variance of the :math:`\pi_0` adversarial sampler

    Returns
    -------
    theta: np.ndarray
        Concatenated array of the parameters of the model, except the intercept if there is one
    intercept: Optional[np.ndarray]
        If the model has specificaly an intercept as one of its parameters, it is stacked in this output
        tensor
    lambd: Union[np.ndarray, float]
        Dual variable :math:`\lambda` of the problem

    Shapes
    ------
    sigma_: (1,) or (d, d)
    theta: (n_params,)
    intercept: (n_intercepts,) or None
    lambd: (1,)
    """
    if isinstance(sigma_, float): sigma = pt.tensor(sigma_)
    elif isinstance(sigma_, pt.Tensor): sigma = sigma_
    elif sigma_ is None:
        sigma = wdro_problem.rho / 2
    else: raise ValueError("Please provide a valid type for sigma_ parameter in solve_dual.")

    # Cast our raw data into tensors
    xi, xi_labels = extract_data(wdro_problem.p_hat)

    loss = wdro_problem.loss

    # If user provides a numpy loss, fail.
    assert loss is not None
    assert isinstance(loss, _DualLoss)

    # Initialize sampler.
    if loss._sampler is None:
        loss.sampler = loss.default_sampler(xi, xi_labels, sigma, seed)
    assert isinstance(loss.sampler, BaseSampler)

    # If user wants to specify a custom optimizer, they need to register an instance
    # of a subclass of torch optimizers in the relevant attribute.
    optimizer: pt.optim.Optimizer = loss.optimizer

    # _DualLoss.presample determines the way the optimization is performed
    if loss.presample:
        losses = optim_presample(optimizer, xi, xi_labels, loss)
        np.save(
                "test_pre.npy",
                losses
            )
    else:
        losses = optim_postsample(optimizer, xi, xi_labels, loss)
        np.save(
                "test_post.npy",
                losses
            )
    theta = detach_tensor(loss.theta)
    intercept = loss.intercept
    if intercept is not None:
        intercept = detach_tensor(intercept)
    lambd = detach_tensor(loss.lam) if loss.rho > 0. else [0.]
    robust_loss = losses[-1]
    return theta, intercept, lambd, robust_loss

def optim_presample(
        optimizer: pt.optim.Optimizer,
        xi: pt.Tensor,
        xi_labels: Optional[pt.Tensor],
        loss: _DualLoss) -> List[float]:
    r""" Optimize the dual loss by sampling the :math:`zeta` values once at the begining of
    the optimization, the performing a deterministic gradient descent (e.g. BFGS style algorithm).

    Parameters
    ----------
    optimizer : pt.optim.Optimizer
        loss-dependant optimizer, can be customized if needed
    xi : pt.Tensor
        data tensor
    xi_labels : Optional[pt.Tensor]
        target tensor
    loss : _DualLoss
        dual loss instance

    Returns
    -------
    List[float]

    Shapes
    ------
    xi: (m, d)
    xi_labels: (m, d')
    """

    zeta, zeta_labels = loss.generate_zetas()

    def closure(back=True) -> float:
        """ Loss evaluation function, performing the forward pass for the autograd engine.
        """
        optimizer.zero_grad()

        # Forward pass
        objective = loss(xi, xi_labels, zeta, zeta_labels)
        assert isinstance(objective, pt.Tensor)

        # Backward pass
        if back: objective.backward()
        return objective.item()

    losses = []
    pretrain_iters, train_iters = interpret_steps_struct(loss.n_iter)

    # Pretrain ERM
    loss.erm_mode = True
    for _ in range(pretrain_iters):
        optimizer.step(closure)

    # Init lambda
    loss.get_initial_guess_at_dual(xi, xi_labels)
    loss.erm_mode = False

    # Train WDRO
    for _ in range(train_iters):
        # Do not resample, only step according to BFGS-style algo
        optimizer.step(closure)
        with pt.no_grad():
            _is = loss.imp_samp
            loss.imp_samp = not _is
            losses.append(closure(False))
            loss.imp_samp = _is
            del _is

    return losses

def optim_postsample(
        optimizer: pt.optim.Optimizer,
        xi: pt.Tensor,
        xi_labels: Optional[pt.Tensor],
        loss: _DualLoss) -> List[pt.Tensor]:
    r""" Optimize the dual loss by resampling the :math:`\zeta` values at each gradient descent step.

    Parameters
    ----------
    n_iter : int
        number of gradient descent iterations to perform
    optimizer : pt.optim.Optimizer
        loss-dependant optimizer, can be customized if needed
    xi : pt.Tensor
        data tensor
    xi_labels : Optional[pt.Tensor]
        target tensor
    loss : _DualLoss
        dual loss instance

    Returns
    -------
    List[float]

    Shapes
    ------
    xi: (m, d)
    xi_labels: (m, d')
    """
    losses = []

    pretrain_iters, train_iters = interpret_steps_struct(loss.n_iter)

    # Pretrain ERM
    loss.erm_mode = True
    for _ in range(pretrain_iters):
        optimizer.zero_grad()

        # Resamples zetas at forward pass
        objective = loss(xi, xi_labels)
        assert isinstance(objective, pt.Tensor)

        objective.backward()
        # Perform the stochastic step
        optimizer.step()
        losses.append(objective.item())

    # Init lambda
    loss.get_initial_guess_at_dual(xi, xi_labels)
    loss.erm_mode = False

    # Train WDRO
    for _ in range(train_iters):
        optimizer.zero_grad()

        # Resamples zetas at forward pass
        objective = loss(xi, xi_labels)
        assert isinstance(objective, pt.Tensor)

        objective.backward()
        # Perform the stochastic step
        optimizer.step()
        losses.append(objective.item())

    return losses
