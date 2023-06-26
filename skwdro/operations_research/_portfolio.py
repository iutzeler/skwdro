"""
WDRO Estimators
"""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted, check_random_state

import skwdro.solvers.specific_solvers as spS
import skwdro.solvers.entropic_dual_solvers as entS
import skwdro.solvers.entropic_dual_torch as entTorch
from skwdro.base.problems import WDROProblem, EmpiricalDistributionWithoutLabels
#USELESS FOR NOW: from skwdro.base.losses import PortfolioLoss 
from skwdro.base.losses import PortfolioLoss_torch
from skwdro.base.losses_torch import *
from skwdro.base.costs_torch import NormCost as NormCostTorch
from skwdro.base.costs import NormCost
from skwdro.solvers.oracle_torch import DualLoss
from skwdro.base.cost_decoder import cost_from_str

from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV, KFold

class Portfolio(BaseEstimator):
    r""" A Wasserstein Distributionally Robust Mean-Risk Portfolio estimator.

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
    cost: str, default="n-NC-1-2"
        Tiret-separated code to define the transport cost: "<engine>-<cost id>-<k-norm type>-<power>" for :math:`c(x, y):=\|x-y\|_k^p`
    solver: str, default='entropic'
        Solver to be used: 'entropic' or 'dedicated'
    solver_reg: float, default=1.0
        regularization value for the entropic solver
    reparam: str, default="softmax"
        Reparametrization method of theta for the entropic torch loss
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
                 cost="n-NC-1-1",
                 solver="dedicated",
                 solver_reg=1e-3,
                 reparam="softmax",
                 n_zeta_samples: int=10,
                 random_state=None,
                 seed: int=0
                 ):

        #Verifying conditions on rho, eta and alpha
        if rho < 0:
            raise ValueError("The Wasserstein radius cannot be negative")
        elif eta < 0:
            raise ValueError("Risk-aversion error eta cannot be negative")
        elif alpha <= 0 or alpha > 1:
            raise ValueError("Confidence level alpha needs to be in the (0,1] interval")
        elif solver_reg < 0:
            raise ValueError("The regularization parameter cannot be negative")
        elif n_zeta_samples < 0:
            raise ValueError("Cannot sample a negative number of zetas")
        
        self.rho = float(rho) #Conversion to float to prevent torch.nn conversion errors
        self.eta = float(eta)
        self.alpha = float(alpha)
        self.fit_intercept = fit_intercept
        self.cost = cost
        self.solver = solver
        self.solver_reg = float(solver_reg)
        self.reparam = reparam
        self.random_state = random_state
        self.n_zeta_samples = n_zeta_samples
        self.seed = seed

    def fit(self, X, y=None, C=None, d=None):
        """Fits the WDRO regressor.

        Parameters
        ----------
        X : array-like, shape (n_samples_train,m)
            The training input samples.
        y : None
            The prediction. Always none for a portfolio estimator.

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
        emp = EmpiricalDistributionWithoutLabels(m=N, samples=X)

        self.cost_ = cost_from_str(self.cost)# NormCost(1, 1., "L1 cost")
        self.problem_ = WDROProblem(
                loss=PortfolioLoss_torch(eta=self.eta, alpha=self.alpha),
                cost=self.cost_,
                xi_bounds=[-np.inf,np.inf],
                theta_bounds=[-np.inf,np.inf],
                rho=self.rho,
                p_hat=emp,
                d=m,
                n=m
            )
        
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
        elif self.solver == "entropic_torch" or self.solver == "entropic_torch_pre":
            epsilon = pt.tensor(self.solver_reg)

            self.problem_.loss = DualPreSampledLoss(
                    MeanRisk_torch(loss=RiskPortfolioLoss_torch(cost=self.cost_, xi=pt.tensor(X),
                                                                epsilon=epsilon, 
                                                                m=m, 
                                                                reparam=self.reparam),
                    eta=pt.as_tensor(self.eta), 
                    alpha=pt.as_tensor(self.alpha)),
                    cost = self.cost_,
                    n_samples=self.n_zeta_samples,
                    epsilon_0 = epsilon,
                    rho_0 = pt.as_tensor(self.rho)
                )

        elif self.solver == "entropic_torch_post":
            self.problem_.loss = DualPostSampledLoss(
                    MeanRisk_torch(loss=RiskPortfolioLoss_torch(cost=self.cost_, xi=pt.as_tensor(X), epsilon=pt.tensor(self.solver_reg),
                                                                m=m, reparam=self.reparam), eta=pt.as_tensor(self.eta),
                                                                alpha=pt.as_tensor(self.alpha)),
                    cost = self.cost_,
                    n_samples=self.n_zeta_samples,
                    epsilon_0 = pt.tensor(self.solver_reg),
                    rho_0 = pt.as_tensor(self.rho)
                )
        else:
            raise NotImplementedError("Designation for solver not recognized")
        
        if self.solver in {"entropic_torch_pre", "entropic_torch_post"}:
        
            #Define the optimizer

            '''
            self.problem_.loss.optimizer = hybrid_opt.HybridAdam([
            {'params': [self.problem_.loss.loss.loss._theta_tilde], 'lr':1e-10, 'mwu_simplex':True},
            {'params': [self.problem_.loss.loss.tau]},
            {'params': [self.problem_.loss._lam], 'non_neg':True}
            ], lr=1e-5, betas=(.99, .999), weight_decay=0., amsgrad=True, foreach=True)
            '''
                      
            self.coef_, _, self.dual_var_ = entTorch.solve_dual(
                    self.problem_,
                    sigma = pt.tensor(self.solver_reg)
            ) 

            #We optimize on tau once again
            self.reducer_loss_ = PortfolioLoss_torch(eta=self.eta, alpha=self.alpha)

            self.result_ = self.reducer_loss_.value(theta=self.coef_, xi=X).mean()

        self.is_fitted_ = True

        #Return the estimator
        return self
    
    def optimize_parameters(self, X, y=None):

        #Tuning rho using grid search
        param_grid = {"rho": [10**(-i) for i in range(4,-4,-1)]}
        grid_cv = KFold(n_splits=5, shuffle=True)

        grid_estimator= GridSearchCV(estimator=self, param_grid=param_grid, cv=grid_cv, refit=True, n_jobs=-1, verbose=3)
        #grid_estimator= HalvingGridSearchCV(estimator=estimator, param_grid=param_grid, cv=grid_cv,n_jobs=-1, refit=True, verbose=3, min_resources="smallest")

        grid_estimator.fit(X) #Fit on the new estimator

        best_params = grid_estimator.best_params_
        best_score = grid_estimator.best_score_

        print("Best params: ", best_params)
        print("Best score: ", best_score)

        best_estimator = grid_estimator.best_estimator_
        print("Solver reg value: ",best_estimator.solver_reg)  

        return best_estimator
    
    def score(self, X, y=None):
        '''
        Score method to estimate the quality of the model.

        Parameters
        ----------
        X : array-like, shape (n_samples_test,m)
            The testing input samples.
        y : None
            The prediction. Always none for a portfolio estimator.
        '''
        return -self.eval(X)

    def eval(self, X):
        '''
        Evaluates the loss with the theta obtained from the fit function.

        Parameters
        ----------
        X : array-like, shape (n_samples_test,m)
            The testing input samples.
        '''

        assert self.is_fitted_ == True #We have to fit before evaluating

        def entropic_case(X):
            if isinstance(X, (np.ndarray,np.generic)):
                X = pt.from_numpy(X)

            #We optimize on tau once again
            reducer_loss = PortfolioLoss_torch(eta=self.eta, alpha=self.alpha)

            return reducer_loss.value(theta=self.coef_, xi=X).mean()

        match self.solver:
            case "dedicated":
                return self.problem_.loss.value(theta=self.coef_, X=X)
            case "entropic":
                return NotImplementedError("Entropic solver for Portfolio not implemented yet")
            case "entropic_torch":
                return entropic_case(X)
            case "entropic_torch_pre":
                return entropic_case(X)
            case "entropic_torch_post":
                return entropic_case(X)            
            case _:
                return ValueError("Solver not recognized")

