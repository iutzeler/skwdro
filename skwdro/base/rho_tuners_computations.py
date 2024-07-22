import numpy as np
import torch as pt
from joblib import Parallel, delayed

from skwdro.operations_research import *
from skwdro.linear_models import *


def indicator_func(xii, theta, estimator):

    if isinstance(estimator, Portfolio):
        tau = estimator.tau_ if (isinstance(estimator.tau_, (np.ndarray, np.generic))) \
            else estimator.tau_
        result = np.sign(-(theta @ np.array(xii) + tau))
        assert result != 0, "The loss function is not differentiable"
        return result if result == 1 else 0
    else:
        raise NotImplementedError()


def scal_xi(xii, a):
    return np.append(xii, a)


def compute_h(xii, theta, estimator):
    # Computes the gradient of the loss on theta (only for the Portfolio
    # problem right now)

    if isinstance(estimator, Portfolio):

        eta = estimator.eta_
        alpha = estimator.alpha_

        xii_minus_eta = scal_xi(xii=xii, a=-eta)
        xii_1 = scal_xi(xii=xii, a=1)
        return - (xii_minus_eta + (eta / alpha) * xii_1 * indicator_func(xii=xii,
                                                                         theta=theta, estimator=estimator))

    else:
        raise NotImplementedError()


def compute_phi_star(X, y, z, diff_loss):

    def func_call_portfolio(X, y, theta_tilde, tau):
        params = dict(diff_loss.named_parameters())
        params['loss.loss._theta_tilde'] = theta_tilde
        params['loss._tau'] = tau
        return pt.func.functional_call(
            module=diff_loss, parameter_and_buffer_dicts=params, args=(X, y))

    def func_call_logistic(X, y, weight):
        params = dict(diff_loss.named_parameters())
        params['loss.linear.weight'] = weight
        return pt.func.functional_call(
            module=diff_loss, parameter_and_buffer_dicts=params, args=(X, y))

    def func_call(X, y, theta):
        params = dict(diff_loss.named_parameters())
        params['loss._theta'] = theta
        return pt.func.functional_call(
            module=diff_loss, parameter_and_buffer_dicts=params, args=(X, y))

    n_samples = len(X)

    one_dim = len(X[0]) == 1

    hessian_products = []

    for k in range(n_samples):

        yk = y[k] if y is not None else None
        Xk_conv, yk_conv = diff_loss.convert(X[k], yk)

        class_name = diff_loss.loss.__class__.__name__

        if "MeanRisk" in class_name:
            theta_tilde = diff_loss.get_parameter('loss.loss._theta_tilde')
            tau = diff_loss.get_parameter('loss._tau')

            hessians = pt.func.jacrev(pt.func.jacrev(func_call_portfolio, argnums=(
                0, 2, 3)), argnums=(0, 2, 3))(Xk_conv, yk_conv, theta_tilde, tau)
            hessian_theta_tilde = hessians[0][1][0][0].squeeze(1)
            hessian_tau = hessians[0][2][0][0].squeeze(1)
            hessian_loss = pt.cat((hessian_theta_tilde, hessian_tau), 1)

        elif "Logistic" in class_name or "Quadratic" in class_name:
            weight = diff_loss.get_parameter('loss.linear.weight')

            hessians = pt.func.jacrev(pt.func.jacrev(func_call_logistic, argnums=(
                0, 1, 2)), argnums=(0, 1, 2))(Xk_conv, yk_conv, weight)
            hessian_X = hessians[0][2][0].squeeze(1)
            hessian_y = hessians[1][2][0].squeeze(1)
            hessian_loss = pt.cat((hessian_X, hessian_y.T), 1)
        else:
            theta = diff_loss.get_parameter('loss._theta')
            if yk_conv is None:
                hessians = pt.func.jacrev(pt.func.jacrev(func_call, argnums=(
                    0, 2)), argnums=(0, 2))(Xk_conv, yk_conv, theta)
            else:
                hessians = pt.func.jacrev(pt.func.jacrev(func_call, argnums=(
                    0, 1, 2)), argnums=(0, 1, 2))(Xk_conv, yk_conv, theta)
            hessian_loss = hessians[0][1][0]

        hessian_product = (hessian_loss)**2 if one_dim is True \
            else pt.matmul((hessian_loss).T, hessian_loss)
        hessian_products.append(hessian_product)

    A = (1 / 2 * n_samples) * sum(hessian_products)

    if one_dim is True:  # A is a scalar
        if A != 0:
            alpha_opt = z / A
            return alpha_opt * z - (1 / 2) * alpha_opt * A * alpha_opt
        else:
            return pt.tensor([float("inf")]) if z != pt.tensor([0.]) else 0
    else:
        pseudo_inv_A = pt.linalg.pinv(A)
        alpha_opt = pt.matmul(pseudo_inv_A, z)
        # We consider in that case that z is not in range(A)
        if pt.isclose(pt.matmul(A, alpha_opt).squeeze(), z) is False:
            return pt.tensor([float("inf")])

    # Using directly a.T will be depreciated in a future release of PyTorch so
    # we use this technique to transpose alpha_opt
    alpha_opt_transpose = alpha_opt.permute(
        *pt.arange(alpha_opt.ndim - 1, -1, -1))

    return alpha_opt_transpose @ z - \
        (1 / 2) * pt.dot(pt.matmul(alpha_opt_transpose, A), alpha_opt)


def compute_phi_star_portfolio(X, z, theta, estimator):

    if isinstance(estimator, Portfolio):
        eta = estimator.eta_
        alpha = estimator.alpha_

        q = estimator.cost_.p  # The norm of the cost function is denoted q in Blanchet's paper
        n_samples = len(X)
        C = sum([1 + (eta / alpha) * indicator_func(xii=X[k], theta=theta,
                estimator=estimator) for k in range(n_samples)])

        return (n_samples * np.linalg.norm(x=z, ord=q)**2) / C
    else:
        raise NotImplementedError()
