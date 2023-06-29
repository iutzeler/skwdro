from .base_loss import Loss

import numpy as np
import torch as pt

#Strategy when manipulating torch tensor during parallelization
pt.multiprocessing.set_sharing_strategy('file_system')

from sqwash import SuperquantileReducer

class PortfolioLoss(Loss):
    """
    [ WIP ]
    """

    def __init__(self, l2_reg=None, name="Portfolio loss", eta=0, alpha=.95,\
            fit_intercept="False"):

        self.l2_reg = l2_reg
        self.name = name
        self.eta = eta
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def value(self, theta, X):

        #Define coefficients linked to the problem
        a1 = -1
        a2 = -1 - self.eta/self.alpha
        b1 = self.eta
        b2 = self.eta*(1-(1/self.alpha))

        #Transform theta to respect the simplex condition
        '''
        tmax_vector = np.amax(theta) * np.ones(len(theta))
        theta_tilde = softmax(theta - tmax_vector)

        return_cost = np.dot(theta_tilde.T, X)
        return max(a1*return_cost+b1*self.tau, a2*return_cost+b2*self.tau)
        '''

        return NotImplementedError("TODO: Create the loss after the Cvxopt part")

    def grad_theta(self, theta, xi):
        return NotImplementedError("TODO: Compute the gradient for this loss")

class PortfolioLoss_torch(Loss):

    def __init__(self, eta, alpha, name="Portfolio loss"):
        super(PortfolioLoss_torch, self).__init__()
        self.eta = eta
        self.alpha = alpha
        self.name = name
        self.reducer = SuperquantileReducer(superquantile_tail_fraction=self.alpha)

    def value(self, theta, xi):
        #Conversion np.array to torch.tensor if necessary
        if isinstance(theta, (np.ndarray,np.generic)):
            theta = pt.from_numpy(theta)
        if isinstance(xi, (np.ndarray,np.generic)):
            xi = pt.from_numpy(xi)

        N = xi.size()[0]

        #We add a double cast in the dot product to solve torch type issues for torch.dot
        in_sample_products = -pt.matmul(pt.t(theta), pt.t(xi.double()))
        expected_value = (1/N) * pt.sum(in_sample_products)
        reduce_loss = self.reducer(in_sample_products)

        return expected_value + self.eta*reduce_loss
