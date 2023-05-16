r"""
Some functions for estimation of gradients in lambda and theta.
Backend: Numpy

Full objective:

.. math::
    \inf_{\lambda\ge 0, \theta} J(\theta, \lambda):=
        \lambda\rho + \epsilon
            \mathbb{E}_{\xi\sim\mathbb{P}^N}
            \ln\mathbb{E}_{\zeta\sim\mathcal{N}(\xi, \sigma)}
            e^{\frac{1}{\epsilon}(L_\theta(\zeta)-\lambda c(\zeta, \xi))}

Gradients:
.. math::
    \nabla_\theta J=
    \mathbb{E}_{\xi\sim\mathbb{P}^N}
        \mathbb{E}_{\zeta\sim\mathcal{N}(\xi, \sigma)}
            \nabla_\theta L_\theta(\zeta)\frac{e^{\dots}}{\mathcal{E}_\zeta e^{\dots}}\\
    \nabla_\lambda J=
    \rho - \mathbb{E}_{\xi\sim\mathbb{P}^N}
        \mathbb{E}_{\zeta\sim\mathcal{N}(\xi, \sigma)}
         c(\zeta, \xi)\frac{e^{\dots}}{\mathcal{E}_\zeta e^{\dots}}
"""

import numpy as np
from skwdro.solvers.utils import non_overflow_exp_mean

# LR - schedule
# #############
def lr_decay_schedule(iter_idx, offset: int=10, lr0=1e-1) -> float:
    r"""
    For gradient descent, usualy schedule looking like:

    .. math ::
        \nu_t=\frac{\nu_0}{(o+t)^k}

    Here we use default values :math:`o=10`, :math:`\nu_0=10^{-1}`, :math:`k=8.10^{-1}`
    """
    return lr0 * (iter_idx + offset)**-0.8


# ### Steps ####################################################
def step_lam_wol(xi, zeta, theta, lam, cost, loss, t, rho, epsilon):
    """
    Perform a gradient step for lambda, when no labels are provided.
    """
    # Compute the coefficients that we will exponentiate (the dots in the top formula)
    c = cost(xi[None, :, :], zeta) # (n_samples, m, 1)
    loss_outputs = loss(theta, zeta)
    exps_coefs = loss_outputs - lam * c
    exps_coefs /= epsilon

    # Use the non-overflowing average of exponentials weighted against the (negative) costs
    minus_full_grads = non_overflow_exp_mean(exps_coefs, c)
    grad_estimate = rho - minus_full_grads.mean()

    # Returned the scheduled step
    lr = lr_decay_schedule(t)
    return -lr * grad_estimate

def step_lam_wl(xi, xi_labels, zeta, zeta_labels, theta, lam, cost, loss, t, rho, epsilon):
    """
    Perform a gradient step for lambda, when labels are provided for the cost function.
    """
    # Compute the coefficients that we will exponentiate (the dots in the top formula)
    c = cost(xi[None, :, :], zeta, xi_labels[None, :, :], zeta_labels) # (n_samples, m, 1)
    loss_outputs = loss(theta, zeta, zeta_labels)
    exps_coefs = loss_outputs - lam * c
    exps_coefs /= epsilon

    # Use the non-overflowing average of exponentials weighted against the (negative) costs
    minus_full_grads = non_overflow_exp_mean(exps_coefs, c)
    grad_estimate = rho - minus_full_grads.mean()

    # Returned the scheduled step
    lr = lr_decay_schedule(t)
    return -lr * grad_estimate

def step_theta_wol(xi, zeta, theta, lam, cost, loss_fns, step_id, epsilon):
    """
    Perform a gradient step for theta, when no labels are provided.
    """
    loss, loss_grad = loss_fns
    grads_theta_loss = loss_grad(theta, zeta) # array of shape (n_samples, m, d)

    # Compute the coefficients that we will exponentiate (the dots in the top formula)
    c = cost(xi[None, :, :], zeta) # (n_samples, m, 1)
    loss_outputs = loss(theta, zeta)
    exps_coefs = loss_outputs - lam * c
    exps_coefs /= epsilon

    # Use the non-overflowing average of exponentials weighted against the theta gradients of the loss
    full_grads = non_overflow_exp_mean(exps_coefs, grads_theta_loss)
    grad_estimate = full_grads.mean(axis=0)

    # Returned the scheduled step
    lr = lr_decay_schedule(step_id)
    return -lr * grad_estimate

def step_theta_wl(xi, xi_labels, zeta, zeta_labels, theta, lam, cost, loss_fns, step_id, epsilon):
    """
    Perform a gradient step for theta, when labels are provided for the cost function.
    """
    loss, loss_grad = loss_fns
    grads_theta_loss = loss_grad(theta, zeta, zeta_labels) # array of shape (n_samples, m, d)

    # Compute the coefficients that we will exponentiate (the dots in the top formula)
    c = cost(xi[None, :, :], zeta, xi_labels[None, :, :], zeta_labels) # (n_samples, m, 1)
    loss_outputs = loss(theta, zeta, zeta_labels)
    exps_coefs = loss_outputs - lam * c
    exps_coefs /= epsilon

    # Use the non-overflowing average of exponentials weighted against the theta gradients of the loss
    full_grads = non_overflow_exp_mean(exps_coefs, grads_theta_loss)
    grad_estimate = full_grads.mean(axis=0)

    # Returned the scheduled step
    lr = lr_decay_schedule(step_id)
    return -lr * grad_estimate

def project_lambda(lam):
    return max(0., lam)

def step_wgx_wol(xi, zeta, theta, lam, cost, loss_fns, t, rho_eps):
    """
    Perform the step itself, provided a steping function without data labels
    """
    rho, epsilon = rho_eps

    K = 5
    step_theta = step_theta_wol(xi, zeta, theta, lam, cost, loss_fns, t, epsilon)
    for k in range(1, K):
        step_theta = step_theta_wol(xi, zeta, theta, lam, cost, loss_fns, t+k, epsilon)
        theta += step_theta
    step_lambda = step_lam_wol(xi, zeta, theta, lam, cost, loss_fns[0], t, rho, epsilon)
    lam += step_lambda

    # Gradient projection
    lam = project_lambda(lam)

    return theta, lam, (step_theta, step_lambda)

def step_wgx_wl(xi, xi_labels, zeta, zeta_labels, theta, lam, cost, loss_fns, t, rho_eps):
    """
    Perform the step itself, provided a steping function handling data labels for costs
    """
    rho, epsilon = rho_eps

    K = 5
    step_theta = step_theta_wl(xi, xi_labels, zeta, zeta_labels, theta, lam, cost, loss_fns, t, epsilon)
    for k in range(1, K):
        step_theta = step_theta_wl(xi, xi_labels, zeta, zeta_labels, theta, lam, cost, loss_fns, t+k, epsilon)
        theta += step_theta
    step_lambda = step_lam_wl(xi, xi_labels, zeta, zeta_labels, theta, lam, cost, loss_fns[0], t, rho, epsilon)
    lam += step_lambda

    # Gradient projection
    lam = project_lambda(lam)

    return theta, lam, (step_theta, step_lambda)
# ##############################################################
