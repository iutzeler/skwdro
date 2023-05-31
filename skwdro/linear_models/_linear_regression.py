"""
Linear Regression
"""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted



from skwdro.base.problems import WDROProblem, EmpiricalDistributionWithLabels
from skwdro.base.losses import QuadraticLoss
# from skwdro.base.losses_torch import *
from skwdro.base.costs import Cost, NormCost
from skwdro.solvers.optim_cond import OptCond

#import skwdro.solvers.specific_solvers as spS
import skwdro.solvers.entropic_dual_solvers as entS
#import skwdro.solvers.entropic_dual_torch as entTorch


class LinearRegression(BaseEstimator):
    """ A Wasserstein Distributionally Robust linear regression.


    The cost function is XXX
    Uncertainty is XXX

    Parameters
    ----------
    rho : float, default=1e-2
        Robustness radius
    l2_reg  : float, default=None
        l2 regularization
    fit_intercept : boolean, default=True
        Determines if an intercept is fit or not
    cost: Loss, default=NormCost(p=2)
        Transport cost
    solver: str, default='entropic'
        Solver to be used: 'entropic' or 'dedicated'
    solver_reg: float, default=1.0
        regularization value for the entropic solver

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        parameter vector (:math:`w` in the cost function formula)
    intercept_ : float
        constant term in decision function.

    Examples
    --------
    >>> import numpy as np
    >>> from skwdro.linear_models import LinearRegression
    >>> XXX
    >>> estimator = LinearRegression()
    >>> estimator.fit(X_train,y_train)
    LinearRegression()
    >>> estimator.predict(X_test)
    >>> estimator.score(X_test,y_test)
    """

    def __init__(self,
                 rho=1e-2,
                 l2_reg=None,
                 fit_intercept=True,
                 cost: Cost=NormCost(p=2),
                 solver="entropic",
                 solver_reg=1.0,
                 solver_cond=OptCond(2)
                 ):

        self.rho    = rho
        self.l2_reg = l2_reg
        self.cost   = cost
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.solver_reg = solver_reg
        self.opt_cond = solver_cond

        self.problem = WDROProblem(
                cost=cost,
                Xi_bounds=[-1e8,1e8],
                Theta_bounds=[-1e8,1e8],
                rho=rho,
                loss=QuadraticLoss(l2_reg=l2_reg)
            )


    def fit(self, X, y):
        """Fits the WDRO classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int. Only -1 or +1 are currently supported

        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # Store the classes seen during fit
        #self.classes_ = unique_labels(y)

        # Store data
        self.X_ = X
        self.y_ = y

        # Setup problem parameters ################
        m, d = np.shape(X)
        emp = EmpiricalDistributionWithLabels(m=m,samplesX=X,samplesY=y)
        self.problem.n = d
        self.problem.d = d
        self.problem.dLabel = 1
        self.problem.P = emp
        # #########################################

        if self.solver=="entropic":
            self.coef_ , self.intercept_, self.dual_var_ = entS.WDROEntropicSolver(
                    self.problem,
                    fit_intercept=self.fit_intercept,
                    opt_cond=self.opt_cond
            )
        elif self.solver=="dedicated":
            raise NotImplementedError
        else:
            raise NotImplementedError

        self.is_fitted_ = True

        # Return the classifier
        return self

  


    def predict(self, X):
        """ Robust prediction.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The prediction
        """
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)

        return self.intercept_ + X@self.coef_

