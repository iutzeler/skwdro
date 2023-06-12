"""
WDRO Estimators
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted, check_random_state

from skwdro.base.problems import WDROProblem, EmpiricalDistribution
from skwdro.base.losses import PortfolioLoss
from skwdro.base.losses_torch import *
from skwdro.base.costs_torch import NormCost as NormCostTorch
from skwdro.base.costs import NormCost
from skwdro.solvers.oracle_torch import DualLoss

import skwdro.solvers.specific_solvers as spS
import skwdro.solvers.entropic_dual_solvers as entS
import skwdro.solvers.entropic_dual_torch as entTorch

class Portfolio(BaseEstimator):
    """ A Wasserstein Distributionally Robust Mean-Risk Portfolio estimator.

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
    random_state: int, default=None
        Seed used by the random number generator when using non-deterministic methods
        on the estimator

    Attributes
    ----------
    C : (nb_constraints, nb_assets)
        Matrix of constraints observed by the user.
    d : (nb_constraints,)
        Vector of constraints observed by the user.

    Examples
    >>> from skwdro.estimators import Portfolio
    >>> import numpy as np
    >>> X = np.random.normal(size=(10,2))
    >>> estimator = Portfolio()
    >>> estimator.fit(X)
    Portfolio()

    --------

    """

    def __init__(self,
                 rho=1e-2,
                 eta=0,
                 alpha=.95,
                 fit_intercept=None,
                 cost=None,
                 solver="dedicated",
                 solver_reg: float=1e-2,
                 n_zeta_samples: int=10,
                 random_state=None
                 ):

        #Verifying conditions on rho, eta and alpha
        if rho < 0:
            raise ValueError("The Wasserstein radius cannot be negative")
        elif eta < 0:
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
        self.random_state = random_state
        self.n_samples = n_zeta_samples

    def fit(self, X, y=None, C=None, d=None):
        """Fits the WDRO regressor.

        Parameters
        ----------
        X : array-like, shape (n_samples_train,m)
            The training input samples.

        """

        # Check that X has correct shape
        X = check_array(X)

        # Check random state
        self.random_state_ = check_random_state(self.random_state)

        # Store data
        self.X_ = X

        # Setup problem parameters
        N = np.shape(X)[0] #N samples for the empirical distribution
        m = np.shape(X)[1] #m assets
        self.n_features_in_ = m
        emp = EmpiricalDistribution(m=N, samples=X)

        self.cost_ = NormCost(1, 1., "L1 cost")
        self.problem_ = WDROProblem(
                # TODO: PUT PortfolioLoss INSTEAD, ONLY USE TORCH VERSION IF SOLVER IS ENTROPIC. DOESN'T PASS TYPECHECK BECAUSE loss IS NEITHER A DUAL LOSS NOR A NUMPY LOSS, AND PortfolioLoss DOESN'T WORK (for some error with a multiplication, I'll let u investigate)
                loss=PortfolioLoss_torch(eta=self.eta, alpha=self.alpha),
                cost=self.cost_,
                Xi_bounds=[-np.inf,np.inf],
                Theta_bounds=[-np.inf,np.inf],
                rho=self.rho,
                P=emp,
                d=m,
                n=m
            )

        # Joyeux noel
        torch_loss = DualLoss(
            PortfolioLoss_torch(eta=self.eta, alpha=self.alpha),
            NormCostTorch(1, 1., "L1 cost"),
            n_samples=self.n_samples,
            epsilon_0=pt.tensor(self.solver_reg),
            rho_0=pt.tensor(self.rho))



        # Setup values C and d that define the polyhedron of xi_maj

        if (C is None or d is None):
            self.C_ = np.zeros((1,m))
            self.d_ = np.zeros((1,1))
        else:
            self.C_ = C
            self.d_ = d

        if np.shape(self.C_)[1] != m: #Check that the matrix-vector product is well-defined
            raise ValueError("The number of columns of C don't match the number of lines of any xi")

        if self.solver == "entropic":
            raise NotImplementedError("Entropic solver for Portfolio not implemented yet")
        elif self.solver == "dedicated":
            self.coef_, _, self.dual_var_, self.result_ = spS.WDROPortfolioSolver(self.problem_, self.cost_, self.C_, \
                                                                    self.d_, self.eta, self.alpha)
        else:
            raise NotImplementedError("Designation for solver not recognized")

        self.is_fitted_ = True

        #Return the estimator
        return self

    def eval(self, X):
        '''
        Evaluates the loss with the theta obtained from the fit function.

        Parameters
        ----------
        X : array-like, shape (n_samples_test,m)
            The testing input samples.
        '''

        assert self.is_fitted_ == True #We have to fit before evaluating

        return self.problem_.loss.value(xi=X)


