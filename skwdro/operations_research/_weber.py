"""
Weber problem
"""
import numpy as np
import torch as pt
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances



from skwdro.base.problems import WDROProblem, EmpiricalDistributionWithLabels
from skwdro.base.losses_torch import WeberLoss_torch
from skwdro.base.costs import NormLabelCost
from skwdro.base.costs_torch import NormLabelCost as NormLabelCostTorch

import skwdro.solvers.entropic_dual_solvers as entS
import skwdro.solvers.entropic_dual_torch as entTorch
from skwdro.solvers.oracle_torch import DualLoss



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
    >>> X = np.random.exponential(scale=2.0,size=(20,1))
    >>> estimator = Weber()
    >>> estimator.fit(X,w)
    Weber()
    """

    def __init__(
            self,
            rho: float=1e-1,
            kappa: float=10.0,
            solver_reg: float=1e-2,
            n_zeta_samples: int=10,
            solver="entropic_torch"):

        if rho is not float:
            try:
                rho = float(rho)
            except:
                raise TypeError(f"The uncertainty radius rho should be numeric, received {type(rho)}")

        if rho < 0:
            raise ValueError(f"The uncertainty radius rho should be non-negative, received {rho}")

        self.rho    = rho
        self.kappa  = kappa
        self.solver = solver
        self.solver_reg = solver_reg
        self.n_zeta_samples = n_zeta_samples

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

        m,d = np.shape(X)

        emp = EmpiricalDistributionWithLabels(m=m,samples_x=X,samples_y=y.reshape(-1,1))
        cost = NormLabelCostTorch(kappa=self.kappa)

        self.problem_ = WDROProblem(
                loss=DualLoss(
                    WeberLoss_torch(),
                    cost,
                    n_samples=self.n_zeta_samples,
                    epsilon_0=pt.tensor(self.solver_reg),
                    rho_0=pt.tensor(self.rho)),
                cost = cost,
                Xi_bounds=[0,20],
                Theta_bounds=[0,np.inf],
                rho=self.rho,
                P=emp,
                d=d,
                dLabel=1,
                n=d
                )




        if self.solver=="entropic_torch":
            self.coef_ , _, self.dual_var_ = entTorch.solve_dual(self.problem_, sigma=0.1)
        else:
            raise(NotImplementedError)


        self.is_fitted_ = True

        self.n_features_in_ = d

        self.position_ = self.coef_

        # `fit` should always return `self`
        return self

