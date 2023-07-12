"""
Linear Regression
"""
import warnings

import numpy as np
import torch as pt

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import ConvergenceWarning, DataConversionWarning
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

import skwdro.solvers.specific_solvers as spS
import skwdro.solvers.entropic_dual_solvers as entS
import skwdro.solvers.entropic_dual_torch as entTorch
from skwdro.base.problems import WDROProblem, EmpiricalDistributionWithLabels
from skwdro.base.losses import QuadraticLoss
from skwdro.base.losses_torch import QuadraticLoss as QuadraticLossTorch
from skwdro.base.samplers.torch import LabeledCostSampler
from skwdro.solvers.optim_cond import OptCond
from skwdro.solvers.oracle_torch import DualLoss, DualPreSampledLoss
from skwdro.base.cost_decoder import cost_from_str

class LinearRegression(BaseEstimator, RegressorMixin):
    r""" A Wasserstein Distributionally Robust linear regression.


    The cost function is

    .. math::
        \ell(\theta,\xi=(x,y)) = \frac{1}{2}(\langle \theta,x \rangle - y)^2


    The WDRO problem solved at fitting is

    .. math::
        \min_{\theta} \max_{\mathbb{Q} : W(\mathbb{P}_n,\mathbb{Q})} \mathbb{E}_{\xi\sim\mathbb{Q}} \ell(\theta,\xi=(x,y))

    Parameters
    ----------
    rho : float, default=1e-2
        Robustness radius
    l2_reg : float, default=0.
        l2 regularization
    fit_intercept : boolean, default=True
        Determines if an intercept is fit or not
    cost: str, default="n-NC-1-2"
        Tiret-separated code to define the transport cost: "<engine>-<cost id>-<k-norm type>-<power>" for :math:`c(x, y):=\|x-y\|_k^p`
    solver: str, default='entropic'
        Solver to be used: 'entropic', 'entropic_torch' (_pre or _post) or 'dedicated'
    solver_reg: float, default=1.0
        regularization value for the entropic solver
    n_zeta_samples: int, default=10
        number of adversarial samples to draw
    opt_cond: Optional[OptCond]
        optimality condition, see :py:class:`OptCond`

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        parameter vector (:math:`w` in the cost function formula)
    intercept_ : float
        constant term in decision function.

    Examples
    --------
    >>> import numpy as np
    >>> from skwdro.linear_models import LinearRegression as RobustLinearRegression
    >>> from sklearn.model_selection import train_test_split
    >>> d = 10; m = 100
    >>> x0 = np.random.randn(d)
    >>> X = np.random.randn(m,d)
    >>> y = X.dot(x0) +  np.random.randn(m)
    >>> X_train, X_test, y_train, y_test = train_test_split(X,y)
    >>> rob_lin = RobustLinearRegression(rho=0.1,solver="entropic",fit_intercept=True)
    >>> rob_lin.fit(X_train, y_train)
    LinearRegression(rho=0.1)
    >>> y_pred_rob = rob_lin.predict(X_test)
    """

    def __init__(self,
                 rho=1e-2,
                 l2_reg=.0,
                 fit_intercept=True,
                 cost="n-NC-1-2",
                 solver="entropic",
                 solver_reg=1.0,
                 n_zeta_samples: int=10,
                 random_state: int=0,
                 opt_cond=None
                 ):

        if rho is not float:
            try:
                rho = float(rho)
            except:
                raise TypeError(f"The uncertainty radius rho should be numeric, received {type(rho)}")

        if rho < 0:
            raise ValueError(f"The uncertainty radius rho should be non-negative, received {rho}")

        self.rho            = rho
        self.l2_reg         = l2_reg
        self.cost           = cost
        self.fit_intercept  = fit_intercept
        self.solver         = solver
        self.solver_reg     = solver_reg
        self.opt_cond       = opt_cond
        self.n_zeta_samples = n_zeta_samples
        self.random_state   = random_state



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
            warnings.warn(f"y expects a shape (n_samples,) but receiced shape {y.shape}", DataConversionWarning)

        # Store data
        self.X_ = X
        self.y_ = y

        m, d = np.shape(X)
        self.n_features_in_ = d

        # Setup problem parameters ################
        emp = EmpiricalDistributionWithLabels(m=m,samples_x=X,samples_y=y[:,None])

        cost = cost_from_str(self.cost)
        self.problem_ = WDROProblem(
                loss=QuadraticLoss(l2_reg=self.l2_reg),
                cost=cost,
                xi_bounds=[-1e8,1e8],
                theta_bounds=[-1e8,1e8],
                rho=self.rho,
                p_hat=emp,
                d_labels=1,
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
        elif "torch" in self.solver:
            custom_sampler = LabeledCostSampler(
                    cost,
                    pt.Tensor(self.problem_.p_hat.samples_x),
                    pt.Tensor(self.problem_.p_hat.samples_y),
                    epsilon=pt.tensor(self.rho),
                    seed=self.random_state
                )

            if self.solver == "entropic_torch" or self.solver == "entropic_torch_post":
                self.problem_.loss = DualLoss(
                        QuadraticLossTorch(custom_sampler, d=self.problem_.d, fit_intercept=self.fit_intercept),
                        cost,
                        n_samples=10,
                        epsilon_0=pt.tensor(self.solver_reg),
                        rho_0=pt.tensor(self.rho)
                    )

            elif self.solver == "entropic_torch_pre":
                self.problem_.loss = DualPreSampledLoss(
                        QuadraticLossTorch(custom_sampler, d=self.problem_.d, fit_intercept=self.fit_intercept),
                        cost,
                        n_samples=10,
                        epsilon_0=pt.tensor(self.solver_reg),
                        rho_0=pt.tensor(self.rho)
                    )
            else:
                raise NotImplementedError

            self.coef_, self.intercept_, self.dual_var_ = entTorch.solve_dual(
                    self.problem_,
                    seed=self.random_state,
                    sigma_=self.solver_reg,
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

