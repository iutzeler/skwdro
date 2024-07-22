from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np

import torch as pt

from skwdro.solvers.oracle_torch import _DualLoss

from skwdro.solvers.result import wrap_solver_result
from skwdro.solvers.utils import detach_tensor, interpret_steps_struct
from skwdro.solvers.optim_cond import OptCondTorch
from skwdro.base.problems import Distribution
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
        xi_labels = pt.Tensor(dist.samples_y)
        return xi, xi_labels
    else:
        xi = pt.Tensor(dist.samples)
        return xi, None


# @wrap_solver_result
# def solve_dual(wdro_problem: WDROProblem):
#     r""" Solve the dual problem with the loss-dependant grandient descent algorithm.

#     Parameters
#     ----------
#     wdro_problem : WDROProblem
#         Whole WDRO problem containing relevant parameters and data
#     sigma_ : Union[float, pt.Tensor]
#         variance of the :math:`\pi_0` adversarial sampler

#     Returns
#     -------
#     theta: np.ndarray
#         Concatenated array of the parameters of the model, except the intercept if there is one
#     intercept: Optional[np.ndarray]
#         If the model has specificaly an intercept as one of its parameters, it is stacked in this output
#         tensor
#     lambd: Union[np.ndarray, float]
#         Dual variable :math:`\lambda` of the problem

#     Shapes
#     ------
#     sigma_: (1,) or (d, d)
#     theta: (n_params,)
#     intercept: (n_intercepts,) or None
#     lambd: (1,)
#     """

#     # Cast our raw data into tensors
#     xi, xi_labels = extract_data(wdro_problem.p_hat)

#     loss = wdro_problem.loss

#     # If user provides a numpy loss, fail.
#     assert loss is not None
#     assert isinstance(loss, _DualLoss)

#     # Initialize sampler.
#     assert isinstance(loss.sampler, BaseSampler)

#     # If user wants to specify a custom optimizer, they need to register an instance
#     # of a subclass of torch optimizers in the relevant attribute.
#     optimizer: pt.optim.Optimizer = loss.optimizer


#     # _DualLoss.presample determines the way the optimization is performed
#     optim_ = optim_presample if loss.presample else optim_postsample

#     opt_cond: OptCondTorch = wdro_problem.opt_cond

#     losses, lgrads, tgrads, lams = optim_(optimizer, xi, xi_labels, loss, opt_cond)

#     plt.rcParams.update({
#         "text.usetex": True,
#         "font.family": 'STIXGeneral',
#         "mathtext.fontset": 'cm'
#     })
#     fig, axes = plt.subplots(4, 1, sharex=True)
#     axes[0].plot(losses, label='Robust loss L',color='k')
#     axes[0].set_yscale('log')
#     axes[1].plot(range(len(losses) - len(lgrads), len(losses)), lgrads, label='$\\nabla_\\lambda L$',color='r')
#     # axes[1].set_yscale('log')
#     axes[2].plot(tgrads, label='$\\nabla_\\theta L$',color='g')
#     axes[2].set_yscale('log')
#     axes[3].plot(range(len(losses) - len(lgrads), len(losses)), lams, label='$\\lambda$',color='b')
#     axes[3].set_yscale('log')
#     fig.suptitle(f"$\\epsilon=${loss.epsilon.item()}")
#     fig.legend()
#     fig.savefig(f"epsilon{loss.epsilon.item()}.png", transparent=True)
#     plt.show()
#     theta = detach_tensor(loss.theta)
#     intercept = loss.intercept
#     if intercept is not None:
#         intercept = detach_tensor(intercept)
#     lambd = detach_tensor(loss.lam) if loss.rho > 0. else [0.]
#     robust_loss = losses[-1]
#     return theta, intercept, lambd, robust_loss


@wrap_solver_result
def solve_dual_wdro(loss: _DualLoss, p_hat: Distribution, opt: OptCondTorch):
    r""" Solve the dual problem with the loss-dependant grandient descent algorithm.

    Parameters
    ----------
    loss: _DualLoss
        Dual loss
    p_hat: Distribution
        Empirical distribution
    opt: OptCond
        Optimality conditions

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

    # Cast our raw data into tensors
    xi, xi_labels = extract_data(p_hat)

    # If user provides a numpy loss, fail.
    assert loss is not None
    assert isinstance(loss, _DualLoss)

    # Initialize sampler.
    assert isinstance(loss.sampler, BaseSampler)
    # If user wants to specify a custom optimizer, they need to register an instance
    # of a subclass of torch optimizers in the relevant attribute.
    optimizer: pt.optim.Optimizer = loss.optimizer

    # _DualLoss.presample determines the way the optimization is performed
    optim_ = optim_presample if loss.presample else optim_postsample

    opt_cond: OptCondTorch = opt

    losses, lgrads, tgrads, lams = optim_(
        optimizer, xi, xi_labels, loss, opt_cond)

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": 'STIXGeneral',
        "mathtext.fontset": 'cm'
    })
    fig, axes = plt.subplots(4, 1, sharex=True)
    axes[0].plot(losses, label='Robust loss L', color='k')
    axes[0].set_yscale('log')
    axes[1].plot(range(len(losses) - len(lgrads), len(losses)),
                 lgrads, label='$\\nabla_\\lambda L$', color='r')
    # axes[1].set_yscale('log')
    axes[2].plot(tgrads, label='$\\nabla_\\theta L$', color='g')
    axes[2].set_yscale('log')
    axes[3].plot(range(len(losses) - len(lgrads), len(losses)),
                 lams, label='$\\lambda$', color='b')
    axes[3].set_yscale('log')
    fig.suptitle(f"$\\epsilon=${loss.epsilon.item()}")
    fig.legend()
    fig.savefig(f"epsilon{loss.epsilon.item()}.png", transparent=True)
    plt.show()
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
        loss: _DualLoss,
        opt_cond: OptCondTorch
) -> List[float]:
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
        if back:
            objective.backward()
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

    if hasattr(optimizer, "reset_lbd_state"):
        optimizer.reset_lbd_state()  # type: ignore

    # Train WDRO
    for iteration in range(train_iters):
        # Do not resample, only step according to BFGS-style algo
        optimizer.step(closure)
        if opt_cond(loss, iteration):
            break
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
        loss: _DualLoss,
        opt_cond: OptCondTorch
) -> List[pt.Tensor]:
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
    lgrads, tgrads, lams = [], [], []

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
        tgrads.append(pt.linalg.norm(
            loss.primal_loss.loss.pos.grad.detach()).item())

    # Init lambda
    loss.get_initial_guess_at_dual(xi, xi_labels)

    if hasattr(optimizer, "reset_lbd_state") and loss.erm_mode:
        optimizer.reset_lbd_state()  # type: ignore

    # Train WDRO
    loss.erm_mode = False
    for iteration in range(train_iters):
        optimizer.zero_grad()

        # Resamples zetas at forward pass
        objective = loss(xi, xi_labels)
        assert isinstance(objective, pt.Tensor)

        objective.backward()
        # Perform the stochastic step
        optimizer.step()
        if opt_cond(loss, iteration):
            break
        losses.append(pt.abs(objective).item())
        lgrads.append(loss._lam.grad.item())
        lams.append(float(loss.lam.item()))
        tgrads.append(pt.linalg.norm(
            loss.primal_loss.loss.pos.grad.detach()).item())

    return np.array(losses), np.array(lgrads), np.array(tgrads), np.array(lams)
