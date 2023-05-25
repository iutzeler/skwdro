"""
WDRO Estimators
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from skwdro.base.problems import WDROProblem, EmpiricalDistribution
from skwdro.base.losses import PortfolioLoss 
# from skwdro.base.losses_torch import *
from skwdro.base.costs import Cost, NormCost

# TODO: Import Portfolio's specific solver when implemented
import skwdro.solvers.entropic_dual_solvers as entS
import skwdro.solvers.entropic_dual_torch as entTorch

class Portfolio(BaseEstimator):
    """ A Wasserstein Distributionally Robust portfolio regressor.
    
    The cost function is XXX
    Uncertainty is XXX

    Parameters
    ----------
    rho : float, default=1e-2
        Robustness radius
    eta: float > 0, default=0
        Risk-aversion parameter linked to the Conditional Value at Risk
    alpha: float in (0,1], default=0.95
        Confindence level of the Conditional Value at Risk
    fit_intercept: boolean, default=None
        Determines if an intercept is fit or not
    cost: Loss, default=NormCost(p=1)
        Transport cost
    solver: str, default='entropic'
        Solver to be used: 'entropic' or 'dedicated'
    solver_reg: float, default=1.0
        regularization value for the entropic solver

    Examples(TODO)
    --------
    
    """

    def __init__(self,
                 rho=1e-2,
                 eta=0,
                 alpha=.95,
                 fit_intercept=None,
                 cost: Cost=NormCost(p=1),
                 solver="entropic",
                 solver_reg=1.0
                 ):
        
        #Verifying conditions on eta and alpha
        if (eta < 0):
            raise ValueError("Risk-aversion error eta cannot be negative")
        elif(alpha <= 0 or alpha > 1):
            raise ValueError("Confidence level alpha needs to be in the (0,1] interval")
        
        self.rho = rho
        self.eta = eta
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.cost = cost
        self.solver = solver
        self.solver_reg = solver_reg

        self.problem = WDROProblem(
                cost=cost,
                Xi_bounds=[-1e8,1e8],
                Theta_bounds=[-1e8,1e8],
                rho=rho,
                loss=PortfolioLoss(l2_reg=None, eta=eta, alpha=alpha)
            )

    def fit(self, X):
        """Fits the WDRO regressor.

        Parameters
        ----------
        X : array-like, shape (n_samples,)
            The training input samples.

        """

        # Check that X and y have correct shape
        #X, y = check_X_y(X, y)

        # Store data
        self.X_ = X

        return NotImplementedError("TODO: fit function to finish")



    