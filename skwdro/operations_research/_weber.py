"""
Weber problem
"""
import numpy as np
import torch as pt

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array

from typing import Optional

from skwdro.solvers.optim_cond import OptCondTorch

from skwdro.solvers.utils import detach_tensor
from skwdro.base.problems import EmpiricalDistributionWithLabels
from skwdro.base.losses_torch.weber import SimpleWeber
from skwdro.wrap_problem import dualize_primal_loss
import skwdro.solvers.entropic_dual_torch as entTorch


class Weber(BaseEstimator):
    """ A Weber Wasserstein Distributionally Robust Estimator.

    The cost function is XXX
    Uncertainty is XXX

    Parameters
    ----------
    rho : float, default=1e-1
        Robustness radius
    kappa: float, default=10.
        For the cost
    solver_reg: float, default=1e-2
        regularization value for the entropic solver
    n_zeta_samples: int, default=10
        number of adversarial samples to draw
    solver: str, default='entropic_torch'
        Solver to be used: 'entropic_torch' (only this is implemented for now)

    Attributes
    ----------
    position_ : float
        parameter vector (:math:`w` in the cost function formula)

    Examples
    --------
    >>> from skwdro.operations_research import Weber
    >>> import numpy as np
    >>> m = 20
    >>> X = np.random.exponential(scale=2.0,size=(m,2))
    >>> w = np.ones(m)
    >>> estimator = Weber()
    >>> estimator.fit(X,w)
    Weber()
    """

    def __init__(
        self,
        rho: float = 1e-1,
        kappa: float = 10.0,
        solver_reg: float = 1e-2,
        sampler_reg: float = 1e-2,
        l2_reg: float = 0.,
        n_zeta_samples: int = 10,
        cost: str = "t-NLC-2-2",
        solver="entropic_torch",
        random_state: int = 0,
        opt_cond: Optional[OptCondTorch] = OptCondTorch(2)
    ):

        if rho < 0:
            raise ValueError(
                f"The uncertainty radius rho should be non-negative, received {rho}")

        self.rho = rho
        self.kappa = kappa
        self.solver = solver
        self.solver_reg = solver_reg
        self.sampler_reg = sampler_reg
        self.l2_reg = l2_reg
        self.n_zeta_samples = n_zeta_samples
        self.random_state = random_state
        self.cost = cost
        self.opt_cond = opt_cond

    def fit(self, X, y):
        """Fits a Weber WDRO model

        Parameters
        ----------
        X : array-like, shape (n_samples,2)
            The training input positions.
        y : array-like, shape (n_samples,) or (n_samples,1)
            The training input importance weights

        Returns
        -------
        self : object
            Returns self.
        """

        X, y = check_X_y(X, y, y_numeric=True)

        if self.rho is not float:
            try:
                self.rho = float(self.rho)
            except ValueError:
                raise TypeError(
                    f"The uncertainty radius rho should be numeric, received {type(self.rho)}")

        m, d = np.shape(X)

        emp = EmpiricalDistributionWithLabels(
            m=m, samples_x=X, samples_y=y.reshape(-1, 1))

        if "torch" in self.solver:
            assert self.opt_cond is not None
            _post_sample = self.solver == "entropic_torch" or self.solver == "entropic_torch_post"
            self._wdro_loss = dualize_primal_loss(
                SimpleWeber(d),
                None,
                pt.tensor(self.rho),
                pt.Tensor(emp.samples_x),
                pt.Tensor(emp.samples_y),
                _post_sample,
                self.cost,
                self.n_zeta_samples,
                self.random_state,
                epsilon=self.solver_reg,
                sigma=self.sampler_reg,
                l2reg=self.l2_reg
            )
            self._wdro_loss.n_iter = 300
            self.coef_, self.intercept_, self.dual_var_, self.robust_loss_ = entTorch.solve_dual_wdro(
                self._wdro_loss,
                emp,
                self.opt_cond,
            )
            self.coef_ = detach_tensor(
                self.problem_.loss.primal_loss.loss.pos).flatten()  # type: ignore

        else:
            raise NotImplementedError

        self.is_fitted_ = True

        self.n_features_in_ = d

        self.position_ = self.coef_

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
        else:
            raise (ValueError("Solver not recognized"))
