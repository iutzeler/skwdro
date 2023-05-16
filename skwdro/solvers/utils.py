import numpy as np


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
