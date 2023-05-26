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

import skwdro.solvers.specific_solvers as spS
import skwdro.solvers.entropic_dual_solvers as entS
import skwdro.solvers.entropic_dual_torch as entTorch

class Portfolio(BaseEstimator):
    """ A Wasserstein Distributionally Robust portfolio estimator.
    
    The cost function is XXX
    Uncertainty is XXX

    Parameters
    ----------
    rho : float, default=1e-2
        Robustness radius
    eta: float > 0, default=0
        Risk-aversion parameter linked to the Conditional Value at Risk
    alpha: float in (0,1], default=0.95
        Confidence level of the Conditional Value at Risk
    fit_intercept: boolean, default=None
        Determines if an intercept is fit or not
    cost: Loss, default=NormCost(p=1)
        Transport cost
    solver: str, default='entropic'
        Solver to be used: 'entropic' or 'dedicated'
    solver_reg: float, default=1.0
        regularization value for the entropic solver
        
    Attributes
    ----------
    C : (nb_constraints, nb_assets), default=np.zeros((nb_constraints, nb_assets))
        Matrix of constraints observed by the user.
    d : (nb_constraints,), default=np.zeros((nb_constraints, nb_assets))
        Vector of constraints observed by the user.

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
                Xi_bounds=[-np.inf,np.inf],
                Theta_bounds=[-np.inf,np.inf],
                rho=rho,
                loss=PortfolioLoss(l2_reg=None, eta=eta, alpha=alpha)
            )

    def fit(self, X, C=np.array([]), d=np.array([])):
        """Fits the WDRO regressor.

        Parameters
        ----------
        X : array-like, shape (n_samples,m)
            The training input samples.

        """

        # Check that X has correct shape
        X = check_array(X)

        # Store data
        self.X_ = X

        # Setup problem parameters
        m = np.shape(X)[0]
        emp = EmpiricalDistribution(m=m, samples=X)

        self.problem.P = emp
        self.problem.d = m
        self.problem.n = m

        # Setup values C and d that define the polyhedron of xi_maj

        if (C is None or d is None) or (np.logical_xor(np.any(C), np.any(d)) is True): #Treating missing constraints
            raise ValueError("Constraints linked to xi are missing")
        elif(np.any(C) is False and np.any(d) is False): #Treating default values
            self.C_ = np.zeros((1,m))
            self.d = np.zeros((1,1))
        else:
            self.C_ = C
            self.d_ = d

        if np.shape(C)[1] != m: #Check that the matrix-vector product is well-defined
            raise ValueError("The number of columns of C don't match the number of lines of any xi")

        if self.solver == "entropic":
            raise NotImplementedError("Entropic solver for Portfolio not implemented yet")
        elif self.solver == "dedicated":
            self.coef_, _, self.dual_var_ = spS.WDROPortfolioSolver(self.problem, self.C_, \
                                                                    self.d_, self.eta, self.alpha)
        else:
            raise NotImplementedError("Designation for solver not recognized")
        
        self.is_fitted_ = True

        #Return the estimator
        return self




    