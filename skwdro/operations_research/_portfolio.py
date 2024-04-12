"""
WDRO Estimators
"""

import numpy as np
import torch as pt
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array

from typing import Optional

from skwdro.base.problems import EmpiricalDistributionWithoutLabels
from skwdro.base.losses_torch.portfolio import SimplePortfolio
from skwdro.solvers.utils import detach_tensor
from skwdro.wrap_problem import dualize_primal_loss

from skwdro.base.cost_decoder import cost_from_str

from skwdro.solvers.optim_cond import OptCondTorch

import skwdro.solvers.specific_solvers as spS
import skwdro.solvers.entropic_dual_torch as entTorch


class Portfolio(BaseEstimator):
    r""" A Wasserstein Distributionally Robust Mean-Risk Portfolio estimator.

    Model for the portfolio optimization problem

    .. math::

        \mathbb{E}[ - \langle x ; \xi \rangle ] + \eta \mathrm{CVar}_\alpha[- \langle x ; \xi \rangle]

    which amounts to using the following loss function

    .. math::

        \ell(x,\tau;\xi) =  - \langle x ; \xi \rangle + \eta \tau + \frac{1}{\alpha} \max( - \langle x ; \xi \rangle - \tau ; 0)

    where :math:`\tau` is an extra real parameter accounting for the threshold of the CVaR (see [Rockafellar and Uryasev (2000)]). The parameter :math:`x` is constrained to live in the simplex (This is encoded in the constraints of the problem in [Esfahani et al. (2018)] and by an exponential reparametrization for the entropy-regularized version).

    Parameters
    ----------
    rho : float, default=1e-2
        Robustness radius
    eta: float > 0, default=0
        Risk-aversion parameter linked to the Conditional Value at Risk
    alpha: float in (0,1], default=0.95
        Confidence level of the Conditional Value at Risk
    C : (nb_constraints, nb_assets), default=None
        Matrix of constraints observed by the user.
    d : (nb_constraints,), default=None
        Vector of constraints observed by the user.
    fit_intercept: boolean, default=None
        Determines if an intercept is fit or not
    cost: str, default="t-NC-1-1"
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


    Examples
    >>> from skwdro.estimators import Portfolio
    >>> import numpy as np
    >>> X = np.random.normal(size=(10,2))
    >>> estimator = Portfolio()
    >>> estimator.fit(X)
    Portfolio()

    --------

    """

    def __init__(
        self,
        rho=1e-2,
        eta=0.,
        alpha=.95,
        C=None,
        d=None,
        fit_intercept=None,
        cost="t-NC-1-1",
        solver="dedicated",
        solver_reg=1e-3,
        reparam="softmax",
        n_zeta_samples: int = 10,
        seed: int = 0,
        opt_cond: Optional[OptCondTorch] = OptCondTorch(
            2
        )  # type: ignore
    ):

        # Verifying conditions on rho, eta and alpha
        if rho < 0:
            raise ValueError("The Wasserstein radius cannot be negative")
        elif eta < 0:
            raise ValueError("Risk-aversion error eta cannot be negative")
        elif alpha <= 0 or alpha > 1:
            raise ValueError(
                "Confidence level alpha needs to be in the (0,1] interval")
        elif solver_reg < 0:
            raise ValueError("The regularization parameter cannot be negative")
        elif n_zeta_samples < 0:
            raise ValueError("Cannot sample a negative number of zetas")
        self.rho = rho
        self.eta = eta
        self.alpha = alpha
        self.C = C
        self.d = d
        self.fit_intercept = fit_intercept
        self.cost = cost
        self.solver = solver
        self.solver_reg = solver_reg
        self.reparam = reparam
        self.n_zeta_samples = n_zeta_samples
        self.seed = seed
        self.opt_cond = opt_cond

    def fit(self, X, y=None):
        """Fits the WDRO regressor.

        Parameters
        ----------
        X : array-like, shape (n_samples_train,m)
            The training input samples.
        y : None
            The prediction. Always none for a portfolio estimator.

        """
        del y

        # Conversion to float to prevent torch.nn conversion errors
        self.rho_ = float(self.rho)
        self.eta_ = float(self.eta)
        self.alpha_ = float(self.alpha)
        self.solver_reg_ = float(self.solver_reg)

        # Check that X has correct shape
        X = check_array(X)

        # Store data
        self.X_ = X

        # Setup problem parameters
        N = np.shape(X)[0]  # N samples for the empirical distribution
        m = np.shape(X)[1]  # m assets
        self.n_features_in_ = m
        emp = EmpiricalDistributionWithoutLabels(m=N, samples=X)

        self.cost_ = cost_from_str(self.cost)  # NormCost(1, 1., "L1 cost")
        p = self.cost_.power

        # Setup values C and d that define the polyhedron of xi_maj
        if (self.C is None or self.d is None):
            self.C_ = np.zeros((1, m))
            self.d_ = np.zeros((1, 1))
        else:
            self.C_ = self.C
            self.d_ = self.d

        # Check that the matrix-vector product is well-defined
        if np.shape(self.C_)[1] != m:
            raise ValueError(
                ' '.join([
                    "The number of columns",
                    "of C don't match the",
                    "number of lines of",
                    "any xi"
                ])
            )

        if self.solver == "entropic":
            raise (DeprecationWarning(
                "The entropic (numpy) solver is now deprecated"
            ))
        elif self.solver == "dedicated":
            _res = spS.WDROPortfolioSpecificSolver(
                C=self.C_,
                d=self.d_,
                m=self.n_features_in_,
                p=p,
                eta=self.eta,
                alpha=self.alpha,
                rho=self.rho,
                samples=emp.samples
            )
            self.coef_, self.tau_, self.dual_var_, self.result_ = _res
        elif "torch" in self.solver:
            self._wdro_loss = dualize_primal_loss(
                SimplePortfolio(m, risk_aversion=self.eta_,
                                risk_level=self.alpha_),
                None,
                pt.tensor(self.rho_),
                pt.Tensor(emp.samples),
                None,
                "post" not in self.solver,
                self.cost,
                self.n_zeta_samples,
                self.seed,
                epsilon=self.solver_reg_,
                l2reg=0.
            )
            _res = entTorch.solve_dual_wdro(
                self._wdro_loss,
                emp,
                self.opt_cond,  # type: ignore
            )
            (
                self.coef_,
                self.intercept_,
                self.dual_var_,
                self.robust_loss_
            ) = _res
            self.coef_ = detach_tensor(
                self._wdro_loss.primal_loss.loss.assets.weight  # type: ignore
            )

        else:
            raise NotImplementedError("Designation for solver not recognized")

        self.is_fitted_ = True

        # Return the estimator
        return self

    def score(self, X, y=None):
        '''
        Score method to estimate the quality of the model.

        Parameters
        ----------
        X : array-like, shape (n_samples_test,m)
            The testing input samples.
        y : None
            The prediction. Always None for a Portfolio estimator.
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

        # Check that X has correct shape
        X = check_array(X)

        assert self.is_fitted_  # We have to fit before evaluating

        if "entropic" in self.solver:
            return self._wdro_loss.primal_loss.forward(pt.from_numpy(X)).mean()
        elif self.solver == "dedicated":
            return -np.mean(X, axis=0) @ self.coef_
        else:
            raise (ValueError("Solver not recognized"))
