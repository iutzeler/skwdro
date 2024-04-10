"""
WDRO Estimators
"""
import numpy as np
import torch as pt
import torch.nn as nn
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array

from typing import Optional

from skwdro.solvers.optim_cond import OptCondTorch

from skwdro.base.problems import EmpiricalDistributionWithoutLabels
import skwdro.solvers.specific_solvers as spS
import skwdro.solvers.entropic_dual_torch as entTorch
from skwdro.base.cost_decoder import cost_from_str
from skwdro.wrap_problem import dualize_primal_loss
from skwdro.solvers.utils import detach_tensor


class CustomNewsvendorLoss(nn.Module):
    def __init__(self, k: float, u: float):
        super().__init__()
        self.k = pt.tensor(k)
        self.u = pt.tensor(u)
        self.theta_ = nn.Parameter(pt.rand(1))

    def forward(self, x):
        gains = self.k * self.theta_
        losses = self.u * pt.minimum(self.theta_, x)
        return (gains - losses).mean(dim=-1, keepdim=True)


class NewsVendor(BaseEstimator):
    r""" A NewsVendor Wasserstein Distributionally Robust Estimator.

    The cost function is XXX
    Uncertainty is XXX

    Parameters
    ----------
    rho : float, default=1e-2
        Robustness radius
    k   : float, default=5
        Buying cost
    u   : float, default=7
        Selling cost
    cost: str, default="n-NC-1-2"
        Tiret-separated code to define the transport cost: "<engine>-<cost id>-<k-norm type>-<power>" for :math:`c(x, y):=\|x-y\|_k^p`
    solver: str, default='entropic'
        Solver to be used: 'entropic', 'entropic_torch' (_pre or _post) or 'dedicated'
    n_zeta_samples: int, default=10
        number of adversarial samples to draw
    opt_cond: Optional[OptCondTorch]
        optimality condition, see :py:class:`OptCondTorch`

    Attributes
    ----------
    coef_ : float
        parameter vector (:math:`w` in the cost function formula)

    Examples
    --------
    >>> from skwdro.operations_research import NewsVendor
    >>> import numpy as np
    >>> X = np.random.exponential(scale=2.0,size=(20,1))
    >>> estimator = NewsVendor()
    >>> estimator.fit(X)
    NewsVendor()
    """

    def __init__(
            self,
            rho: float = 1e-2,
            k: float = 5,
            u: float = 7,
            cost: str = "t-NC-1-2",
            l2_reg: float = 0.,
            solver_reg: float = .01,
            n_zeta_samples: int = 10,
            solver: str = "entropic",
            random_state: int = 0,
            opt_cond: Optional[OptCondTorch] = OptCondTorch(2)
    ):

        if rho < 0:
            raise ValueError(
                f"The uncertainty radius rho should be non-negative, received {rho}")

        self.rho = rho
        self.k = k
        self.u = u
        self.cost = cost
        self.l2_reg = l2_reg
        self.solver = solver
        self.solver_reg = solver_reg
        self.n_zeta_samples = n_zeta_samples
        self.random_state = random_state
        self.opt_cond = opt_cond

    def fit(self, X, y=None):
        """Fits a WDRO model

        Parameters
        ----------
        X : array-like, shape (n_samples,) or (n_samples,1)
            The training input samples.

        Returns
        -------
        self : object
            Returns self.
        """
        del y

        # Input validation
        X = check_array(X)
        X = np.array(X)

        # Type checking for rho
        if self.rho is not float:
            try:
                self.rho = float(self.rho)
            except ValueError:
                raise TypeError(
                    f"The uncertainty radius rho should be numeric, received {type(self.rho)}")

        m, d = np.shape(X)

        # if d>1:
        #     raise ValueError(f"The input X should be one-dimensional, got {d}")
        X = X.mean(axis=1, keepdims=True)
        self.n_features_in_ = d

        self.cost_ = cost_from_str(self.cost)

        emp = EmpiricalDistributionWithoutLabels(m=m, samples=X)
        # #################################

        if "torch" in self.solver:
            self._wdro_loss = dualize_primal_loss(
                CustomNewsvendorLoss(self.k, self.u),
                None,
                pt.tensor(self.rho),
                xi_batchinit=pt.Tensor(emp.samples),
                xi_labels_batchinit=None,
                post_sample=self.solver == "entropic_torch_post",
                cost_spec=self.cost,
                n_samples=self.n_zeta_samples,
                seed=self.random_state,
                epsilon=self.solver_reg,
            )
            # Solve dual problem
            self.coef_, self.intercept_, self.dual_var_, self.robust_loss_ = entTorch.solve_dual_wdro(
                self._wdro_loss,
                emp,
                self.opt_cond,  # type: ignore
            )
            self.coef_ = detach_tensor(
                self._wdro_loss.primal_loss.loss.assets.weight)  # type: ignore

        elif self.solver == "dedicated":
            # Use cvx solver to solve Kuhn MP formulation
            # self.problem_.p_hat.samples = self.problem_.p_hat.samples.flatten()[:, None]
            self.coef_, self.dual_var_ = spS.WDRONewsvendorSpecificSolver(
                k=self.k, u=self.u, rho=self.rho, samples=emp.samples)
            if self.coef_ == 0.0:
                # If theta is 0, so is lambda (constraint non-active)
                self.dual_var_ = 0.0
            else:
                self.dual_var_ = self.u

        else:
            raise NotImplementedError()

        self.is_fitted_ = True

        # `fit` should always return `self`
        return self

    def score(self, X, y=None):
        '''
        Score method to estimate the quality of the model.

        Parameters
        ----------
        X : array-like, shape (n_samples_test,m)
            The testing input samples.
        y : None
            The prediction. Always None for a Newsvendor estimator.
        '''
        del y
        return -self.eval(X)

    def eval(self, X):
        '''
        Evaluates the loss with the theta obtained from the fit function.

        Parameters
        ----------
        X : array-like, shape (n_samples_test,m)
            The testing input samples.
        '''

        assert self.is_fitted_  # We have to fit before evaluating

        # Check that X has correct shape
        X = check_array(X)

        if "entropic" in self.solver:
            return self._wdro_loss.primal_loss.forward(pt.from_numpy(X)).mean()
        elif self.solver == "dedicated":
            gains = self.k * self.coef_
            losses = self.u * np.minimum(self.coef_, X)
            return np.mean(gains - losses)
        else:
            raise (ValueError("Solver not recognized"))
