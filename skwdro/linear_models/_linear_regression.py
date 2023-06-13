"""
Linear Regression
"""
import numpy as np
import torch as pt

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import ConvergenceWarning, DataConversionWarning
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from skwdro.base.problems import WDROProblem, EmpiricalDistributionWithLabels
from skwdro.base.losses import QuadraticLoss
from skwdro.base.losses_torch import QuadraticLoss as QuadraticLossTorch
from skwdro.base.costs import NormCost
from skwdro.base.costs_torch import NormLabelCost
from skwdro.solvers.optim_cond import OptCond

import skwdro.solvers.specific_solvers as spS
import skwdro.solvers.entropic_dual_solvers as entS
import skwdro.solvers.entropic_dual_torch as entTorch
from skwdro.solvers.oracle_torch import DualLoss, DualPreSampledLoss

class LinearRegression(BaseEstimator, RegressorMixin):
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
                 cost=None,
                 solver="entropic",
                 solver_reg=1.0,
                 n_zeta_samples: int=10,
                 opt_cond=None
                 ):

        if rho is not float:
            try:
                rho = float(rho)
            except:
                raise TypeError(f"The uncertainty radius rho should be numeric, received {type(rho)}")

        if rho < 0:
            raise ValueError(f"The uncertainty radius rho should be non-negative, received {rho}")

        self.rho    = rho
        self.l2_reg = l2_reg
        self.cost   = cost
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.solver_reg = solver_reg
        self.opt_cond = opt_cond
        self.n_zeta_samples = n_zeta_samples




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
        X, y = check_X_y(X, y, y_numeric=True)
        X = np.array(X)
        y = np.array(y)

        if len(y.shape) != 1:
            y.flatten()
            raise DataConversionWarning(f"y expects a shape (n_samples,) but receiced shape {y.shape}")

        # Store data
        self.X_ = X
        self.y_ = y

        m, d = np.shape(X)
        self.n_features_in_ = d

        # Setup problem parameters ################
        emp = EmpiricalDistributionWithLabels(m=m,samples_x=X,samples_y=y[:,None])

        self.problem_ = WDROProblem(
                loss=QuadraticLoss(l2_reg=self.l2_reg),
                cost=NormCost(p=2),
                Xi_bounds=[-1e8,1e8],
                Theta_bounds=[-1e8,1e8],
                rho=self.rho,
                P=emp,
                dLabel=1,
                d=d,
                n=d
            )

        # #########################################

        if self.solver=="entropic":
            self.coef_ , self.intercept_, self.dual_var_ = entS.WDROEntropicSolver(
                    self.problem_,
                    fit_intercept=self.fit_intercept,
                    opt_cond=OptCond(2,max_iter=int(1e9),tol_theta=1e-6,tol_lambda=1e-6)
            )

            if np.isnan(self.coef_).any() or (self.intercept_ is not None and np.isnan(self.intercept_)):
                raise ConvergenceWarning(f"The entropic solver has not converged: theta={self.coef_} intercept={self.intercept_} lambda={self.dual_var_} ")
        elif self.solver == "entropic_torch" or self.solver == "entropic_torch_post":
            self.problem_.loss = DualLoss(
                    QuadraticLossTorch(None, d=self.problem_.d, fit_intercept=self.fit_intercept),
                    NormLabelCost(2., 1., 1e8),
                    n_samples=10,
                    epsilon_0=pt.tensor(self.solver_reg),
                    rho_0=pt.tensor(self.rho)
                )

            self.coef_, self.intercept_, self.dual_var_ = entTorch.solve_dual(
                    self.problem_,
                    sigma=self.solver_reg,
                    fit_intercept=self.fit_intercept,
                )
        elif self.solver == "entropic_torch_pre":
            self.problem_.loss = DualPreSampledLoss(
                    QuadraticLossTorch(None, d=self.problem_.d, fit_intercept=self.fit_intercept),
                    NormLabelCost(2., 1., 1e8),
                    n_samples=10,
                    epsilon_0=pt.tensor(self.solver_reg),
                    rho_0=pt.tensor(self.rho)
                )

            self.coef_, self.intercept_, self.dual_var_ = entTorch.solve_dual(
                    self.problem_,
                    sigma=self.solver_reg,
                    fit_intercept=self.fit_intercept,
                )
        elif self.solver=="dedicated":
            self.coef_ , self.intercept_, self.dual_var_ = spS.WDROLinRegSpecificSolver(
                    rho=self.problem_.rho,
                    X=X,
                    y=y,
                    fit_intercept=self.fit_intercept
            )
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

