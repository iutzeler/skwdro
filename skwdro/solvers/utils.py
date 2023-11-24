from typing import Union, Optional, Tuple
import numpy as np
import torch as pt


def weightedExpAverage(a,b):

    n = a.shape[0]

    bmax= np.max(b)

    num = 0.0
    den = 0.0
    for i in range(n):
        num += a[i]*np.exp(b[i]-bmax)
        den += np.exp(b[i]-bmax)
    return num/den

def non_overflow_exp_mean(coefs, multipliers):
    r"""
    Given coefficients :math:`b_i` that we wish to exponentiate and weight coefficients :math:`a_i` to put before, we wish to compute the weighted mean of the coefs' exponentials:

    .. math::
        \frac{1}{N}\sum_{i<N}a_ie^{b_i}=e^{b_{max}-\ln(N)}\sum_{i<N}a_ie^{b_i-b_{max}}

    The average is taken on the first dimension of the arrays (n_samples in the shape, i.e. the number of zeta samples).

    Parameters
    ==========
    coefs: shape (n_samples, m, 1)
        coefficients :math:`b_i` to exponentiate
    multipliers: (n_samples, m, d)
        weights of the average denoted :math:`a_i`

    Returns
    =======
    avg: (m, d)
        properly scaled averaged exponentials
    """
    coef_max = coefs.max(axis=0, keepdims=True) # (1, m, 1)
    exps = np.exp(
            coefs - coef_max
        )
    scaling = np.sum(exps)
    unscaled_avg = np.einsum("ijk,ijk->jk", multipliers, coefs)
    return unscaled_avg / scaling



# ### Initializations and sampling #############################
def init_theta(d):
    # Glorot init
    # Fan_in = n or n+1 depending on intercept
    theta = np.random.randn(d) / np.sqrt(.5 * (d+1))
    return theta

def prepare_data(samples, m, d, n_samples, epsilon, fit_intercept):
    """
    Draw the ``n_samples` zeta vectors from the xi vectors, and initiate theta
    """
    if fit_intercept:
        xi = np.concatenate((np.ones((m, 1)), samples), axis=1)
        theta = init_theta(d + 1)
    else:
        xi = samples
        theta = init_theta(d)
    zeta = sample_pi_0(epsilon, n_samples, xi)
    return xi, theta, zeta

def sample_pi_0(epsilon, n_samples, xi):
    """
    Sample from (truncated) normal distribution centered on the xi vectors
    """

    m = xi.shape[0]
    d = xi.shape[1]

    sigma = epsilon
    pi_0_noise = np.random.randn(n_samples, m, d)
    pi_0_raw_samples = pi_0_noise * sigma + xi[None, :, :]

    # TODO: constrain to Xi_bounds?
    zeta = pi_0_raw_samples
    return zeta
# ##############################################################

Steps = Union[int, Tuple[int, int]]

def detach_tensor(tensor: pt.Tensor) -> np.ndarray:
    out = tensor.detach().cpu().numpy().flatten()
    return out # float(out) if len(out) == 1 else out

def maybe_detach_tensor(tensor: Optional[pt.Tensor]) -> Optional[np.ndarray]:
    return None if tensor is None else detach_tensor(tensor)

def diff_opt_tensor(tensor: Optional[pt.Tensor], us_dim: Optional[int]=0) -> Optional[pt.Tensor]:
    if tensor is None:
        return None
    else:
        return diff_tensor(tensor, us_dim)

def diff_tensor(tensor: pt.Tensor, us_dim: Optional[int]=0) -> pt.Tensor:
    if us_dim is not None:
        return tensor.clone().detach().unsqueeze(us_dim).requires_grad_(True)
    else:
        return tensor.clone().detach().requires_grad_(True)

def maybe_unsqueeze(tensor: Optional[pt.Tensor], dim: int=0) -> Optional[pt.Tensor]:
    return None if tensor is None else tensor.unsqueeze(dim)

def normalize_maybe_vects(tensor: Optional[pt.Tensor], threshold: float=1., dim: int=0) -> Optional[pt.Tensor]:
    return None if tensor is None else normalize_just_vects(tensor, threshold, dim)

def normalize_just_vects(tensor: pt.Tensor, threshold: float=1., dim: int=0) -> pt.Tensor:
    n = pt.linalg.norm(tensor, dim=dim, keepdims=True)
    return tensor / n * pt.min(pt.tensor(threshold), n)

class NoneGradError(ValueError): pass

def maybe_flatten_grad_else_raise(tensor: pt.Tensor) -> pt.Tensor:
    if tensor.grad is None:
        raise NoneGradError(tensor.shape)
    else:
        return tensor.grad.flatten()

def maybe_relocate(eval_points: Optional[pt.Tensor], old_center: Optional[pt.Tensor], new_center: Optional[pt.Tensor]) -> Optional[pt.Tensor]:
    if eval_points is None:
        return None
    else:
        assert old_center is not None
        assert new_center is not None
        return eval_points + old_center.unsqueeze(dim=0) - new_center

def interpret_steps_struct(steps_spec: Steps, default_split: float=.3) -> Tuple[int, int]:
    if isinstance(steps_spec, int):
        assert 0 <= default_split <= 1.
        pretrain_iters = int(steps_spec * default_split)
        return pretrain_iters, steps_spec
    else: # already tuple
        return steps_spec
