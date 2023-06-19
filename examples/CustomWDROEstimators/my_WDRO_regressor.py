"""
Linear Regression
"""
import numpy as np
import torch as pt
import torch.nn as nn

from typing import Optional

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import ConvergenceWarning, DataConversionWarning
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from skwdro.base.problems import WDROProblem, EmpiricalDistributionWithLabels
from skwdro.base.costs_torch import NormLabelCost
from skwdro.base.losses_torch import Loss
from skwdro.base.samplers.torch.base_samplers import LabeledSampler
from skwdro.base.samplers.torch.classif_sampler import ClassificationNormalNormalSampler


import skwdro.solvers.entropic_dual_torch as entTorch
from skwdro.solvers.oracle_torch import DualLoss, DualPreSampledLoss

class MyWDRORegressor(BaseEstimator, RegressorMixin):
    r""" My WDRO Regressor.


    The cost function is

    .. math:: 
        \ell(\theta,\xi=(x,y)) = ...


    The WDRO problem solved at fitting is 

    .. math::
        \min_{\theta} \max_{\mathbb{Q} : W(\mathbb{P}_n,\mathbb{Q})} \mathbb{E}_{\xi\sim\mathbb{Q}} \ell(\theta,\xi=(x,y))

    Parameters
    ----------
    rho : float, default=1e-2
        Robustness radius
    cost: Loss, default=NormCost(p=2)
        Transport cost
    fit_intercept : boolean, default=True
        Determines if an intercept is fit or not
    solver: str, default='entropic'
        Solver to be used: 'entropic' 
    solver_reg: float, default=1.0
        regularization value for the entropic solver

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        parameter vector (:math:`\theta` in the cost function formula)

    """

    def __init__(self,
                 rho=1e-2,
                 cost=None,
                 fit_intercept=True,
                 solver="entropic",
                 solver_reg=1.0,
                 n_zeta_samples: int=10
                 ):

        if rho is not float:
            try:
                rho = float(rho)
            except:
                raise TypeError(f"The uncertainty radius rho should be numeric, received {type(rho)}")

        if rho < 0:
            raise ValueError(f"The uncertainty radius rho should be non-negative, received {rho}")

        self.rho    = rho
        self.cost   = cost
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.solver_reg = solver_reg
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

        cost = NormLabelCost(2., 1., 1e8)

        self.problem_ = WDROProblem(
                loss = None,
                cost = cost,
                rho=self.rho,
                P=emp
            )

        # #########################################

        if self.solver == "entropic_torch" or "entropic_torch_pre":
            self.problem_.loss = DualPreSampledLoss(
                    MyLoss(None, d=self.n_features_in_, fit_intercept=self.fit_intercept),
                    NormLabelCost(2., 1., 1e8),
                    n_samples=10,
                    epsilon_0=pt.tensor(self.solver_reg),
                    rho_0=pt.tensor(self.rho)
                )

            self.coef_, self.intercept_, self.dual_var_ = entTorch.solve_dual(
                    self.problem_,
                    sigma=self.solver_reg,
                )
        elif self.solver == "entropic_torch_post":
            self.problem_.loss = DualLoss(
                    MyLoss(None, d=self.problem_.d, fit_intercept=self.fit_intercept),
                    NormLabelCost(2., 1., 1e8),
                    n_samples=10,
                    epsilon_0=pt.tensor(self.solver_reg),
                    rho_0=pt.tensor(self.rho)
                )

            self.coef_, self.intercept_, self.dual_var_ = entTorch.solve_dual(
                    self.problem_,
                    sigma=self.solver_reg,
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

        # Prediction
        prediction = np.zeros(X.shape[0]) # [CUSTOMIZE] Dummy prediction

        return prediction



class MyLoss(Loss):
    def __init__(
            self,
            sampler: Optional[LabeledSampler]=None,
            *,
            d: int=0,
            fit_intercept: bool=False) -> None:
        
        super(MyLoss, self).__init__(sampler)
        assert d > 0, "Please provide a valid data dimension d>0"

        self.d = d
        self.fit_intercept = fit_intercept

        # Internal structure
        self.linear = nn.Linear(d, 1, bias=fit_intercept) # [CUSTOMIZE] Dummy linear regression


    def value(self, xi: pt.Tensor, xi_labels: pt.Tensor):

        # Loss value
        prediction = self.linear(xi)                # [CUSTOMIZE] Dummy linear regression
        error = nn.MSELoss(reduction='none')        # [CUSTOMIZE] Dummy linear regression
        loss_value = error( prediction, xi_labels)  # [CUSTOMIZE] Dummy linear regression

        return  loss_value

    @classmethod
    def default_sampler(cls, xi, xi_labels, epsilon):
        return ClassificationNormalNormalSampler(xi, xi_labels, sigma=epsilon, l_sigma=epsilon)

    @property
    def theta(self) -> pt.Tensor:
        return self.linear.weight       # [CUSTOMIZE] Optimized parameters

    @property
    def intercept(self) -> pt.Tensor:
        return self.linear.bias         # [CUSTOMIZE] Intercept