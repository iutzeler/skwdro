"""
Linear Regression
"""
import warnings
from typing import Optional

import numpy as np
import torch as pt
import torch.nn as nn

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import DataConversionWarning
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

import skwdro.solvers.specific_solvers as spS
import skwdro.solvers.entropic_dual_torch as entTorch
from skwdro.solvers.utils import detach_tensor, maybe_detach_tensor
from skwdro.base.problems import EmpiricalDistributionWithLabels
from skwdro.base.costs_torch import Cost as TorchCost
from skwdro.solvers.optim_cond import OptCondTorch
from skwdro.base.cost_decoder import cost_from_str
from skwdro.wrap_problem import dualize_primal_loss


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
    cost: str, default="t-NLC-2-2"
        Tiret-separated code to define the transport cost: "<engine>-<cost id>-<k-norm type>-<power>" for :math:`c(x, y):=\|x-y\|_k^p`
    solver: str, default='entropic'
        Solver to be used: 'entropic', 'entropic_torch' (_pre or _post) or 'dedicated'
    solver_reg: float, default=1.0
        regularization value for the entropic solver
    n_zeta_samples: int, default=10
        number of adversarial samples to draw
    opt_cond: Optional[OptCondTorch]
        optimality condition, see :py:class:`OptCondTorch`

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
                 cost="t-NLC-2-2",
                 solver="entropic_torch",
                 solver_reg=None,
                 sampler_reg=None,
                 n_zeta_samples: int = 10,
                 random_state: int = 0,
                 opt_cond: Optional[OptCondTorch] = OptCondTorch(2)
                 ):

        if rho < 0:
            raise ValueError(
                f"The uncertainty radius rho should be non-negative, received {rho}")

        self.rho = rho
        self.l2_reg = l2_reg
        self.cost = cost
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.solver_reg = solver_reg
        self.sampler_reg = sampler_reg
        self.opt_cond = opt_cond
        self.n_zeta_samples = n_zeta_samples
        self.random_state = random_state

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

        # Type checking for rho
        if self.rho is not float:
            try:
                self.rho = float(self.rho)
            except BaseException:
                raise TypeError(
                    f"The uncertainty radius rho should be numeric, received {type(self.rho)}")

        if len(y.shape) != 1:
            y.flatten()
            warnings.warn(
                f"y expects a shape (n_samples,) but receiced shape {y.shape}", DataConversionWarning)

        # Store data
        self.X_ = X
        self.y_ = y

        m, d = np.shape(X)
        self.n_features_in_ = d

        # Setup problem parameters ################
        emp = EmpiricalDistributionWithLabels(
            m=m, samples_x=X, samples_y=y[:, None])

        self.cost_ = cost_from_str(self.cost)

        # #########################################
        if self.solver == "entropic":
            raise (DeprecationWarning(
                "The entropic (numpy) solver is now deprecated"))
        elif "torch" in self.solver:
            assert isinstance(self.cost_, TorchCost)

            if self.opt_cond is None:
                self.opt_cond = OptCondTorch(2)

            _post_sample = self.solver == "entropic_torch" or self.solver == "entropic_torch_post"
            _wdro_loss = dualize_primal_loss(
                nn.MSELoss(reduction="none"),
                nn.Linear(d, 1),
                pt.tensor(self.rho),
                pt.Tensor(emp.samples_x),
                pt.Tensor(emp.samples_y),
                _post_sample,
                self.cost,
                self.n_zeta_samples,
                self.random_state,
                l2reg=self.l2_reg
            )

            self.coef_, self.intercept_, self.dual_var_, self.robust_loss_ = entTorch.solve_dual_wdro(
                _wdro_loss,
                emp,
                self.opt_cond  # type: ignore
            )
            self.coef_ = detach_tensor(
                _wdro_loss.primal_loss.transform.weight)  # type: ignore
            self.intercept_ = maybe_detach_tensor(
                _wdro_loss.primal_loss.transform.bias)  # type: ignore
        elif self.solver == "dedicated":
            self.coef_, self.intercept_, self.dual_var_ = spS.WDROLinRegSpecificSolver(
                rho=self.rho,
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

        return self.intercept_ + X @ self.coef_
