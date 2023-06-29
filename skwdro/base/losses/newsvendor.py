from .base_loss import Loss

import numpy as np

class NewsVendorLoss(Loss):

    def __init__(self, k=5, u=7, name="NewsVendor loss"):
        self.k = k
        self.u = u
        self.name = name

    def value(self,theta,xi):
        return self.k*theta-self.u*np.minimum(theta,xi)

    def _parallel_grad_theta(self, theta, X):
        # New parallelized:
        # shapes in:
        # X(:=zeta): (n_samples, m, 1)
        # theta: (1,)
        # shapes out:
        # grad: (n_samples, m, 1)
        # NOTE: no mean on m !!!!
        grads = self.k*np.ones_like(X) - self.u*(X>theta).astype(int)
        return grads

    def grad_theta(self,theta,xi):
        if len(xi) >= 2:
            # Parallelized
            return self._parallel_grad_theta(theta, xi)
        else:
            if theta>=xi :
                return self.k
            else:
                return self.k-self.u
