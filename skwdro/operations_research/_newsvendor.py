"""
WDRO Estimators
"""
import numpy as np
import torch as pt
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances



from skwdro.base.problems import WDROProblem, EmpiricalDistributionWithoutLabels
from skwdro.base.losses import NewsVendorLoss
from skwdro.base.losses_torch import NewsVendorLoss_torch
from skwdro.base.costs import NormCost, Cost
from skwdro.base.costs_torch import NormCost as NormCostTorch
import skwdro.solvers.specific_solvers as spS
import skwdro.solvers.entropic_dual_solvers as entS
import skwdro.solvers.entropic_dual_torch as entTorch
from skwdro.solvers.oracle_torch import DualLoss, DualPreSampledLoss
from skwdro.base.cost_decoder import cost_from_str



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
    opt_cond: Optional[OptCond]
        optimality condition, see :py:class:`OptCond`

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
            rho: float=1e-2,
            k: float=5,
            u: float=7,
            cost: str="n-NC-1-2",
            solver_reg: float=.01,
            n_zeta_samples: int=10,
            solver: str="entropic"):


        if rho is not float:
            try:
                rho = float(rho)
            except:
                raise TypeError(f"The uncertainty radius rho should be numeric, received {type(rho)}")

        if rho < 0:
            raise ValueError(f"The uncertainty radius rho should be non-negative, received {rho}")

        self.rho    = rho
        self.k      = k
        self.u      = u
        self.cost   = cost
        self.solver = solver
        self.solver_reg = solver_reg
        self.n_zeta_samples = n_zeta_samples



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


        # Input validation
        X = check_array(X)
        X = np.array(X)

        m,d = np.shape(X)

        if d>1:
            raise ValueError(f"The input X should be one-dimensional, got {d}")

        cost = cost_from_str(self.cost)
        # Define problem w/ hyperparameters
        self.problem_ = WDROProblem(
                loss=NewsVendorLoss(k=self.k, u=self.u),
                cost=cost,
                d=1,
                xi_bounds=[0, 20],
                n=1,
                theta_bounds=[0, np.inf],
                rho=self.rho,
                p_hat=EmpiricalDistributionWithoutLabels(m=m,samples=X))
        # #################################

        if "torch" in self.solver:
            # Use torch backend to solve the entropy-regularized version
            self.cost = NormCostTorch(1, 1)
            if self.solver == "entropic_torch" or self.solver == "entropic_torch_pre":
                # Default is to sample once the zetas
                self.problem_.loss = DualPreSampledLoss(
                        NewsVendorLoss_torch(k=self.k,u=self.u),
                        cost,
                        self.n_zeta_samples,
                        epsilon_0=pt.tensor(self.solver_reg),
                        rho_0=pt.tensor(self.rho),
                        )
            elif self.solver == "entropic_torch_post":
                # Use this option to resample the zetas at each gradient step
                self.problem_.loss = DualLoss(
                        NewsVendorLoss_torch(k=self.k,u=self.u),
                        cost,
                        self.n_zeta_samples,
                        epsilon_0=pt.tensor(self.solver_reg),
                        rho_0=pt.tensor(self.rho),
                        )
            else:
                raise NotImplementedError()
            # Solve dual problem
            self.coef_ , self.intercept_, self.dual_var_ = entTorch.solve_dual(
                    self.problem_,
                    sigma_=self.solver_reg)

        elif self.solver=="dedicated":
            # Use cvx solver to solve Kuhn MP formulation
            self.coef_ = spS.WDRONewsvendorSolver(self.problem_)
            if self.coef_ == 0.0:
                # If theta is 0, so is lambda (constraint non-active)
                self.dual_var_ = 0.0
            else:
                self.dual_var_ = self.u

        elif self.solver=="entropic":
            # Numpy entropic solver, soon deprecated
            self.coef_ , self.intercept_, self.dual_var_ = entS.WDROEntropicSolver(self.problem_,epsilon=0.1)

        else:
            raise NotImplementedError()


        self.is_fitted_ = True

        self.coef_ = float(self.coef_)

        # `fit` should always return `self`
        return self

