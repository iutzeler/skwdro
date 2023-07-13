"""
Weber problem
"""
import numpy as np
import torch as pt

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y

import skwdro.solvers.entropic_dual_torch as entTorch
from skwdro.base.problems import WDROProblem, EmpiricalDistributionWithLabels
from skwdro.base.costs_torch import NormLabelCost as NormLabelCostTorch
from skwdro.base.samplers.torch import LabeledCostSampler
from skwdro.solvers.oracle_torch import DualLoss, DualPreSampledLoss
from skwdro.base.losses_torch.weber import WeberLoss



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
            rho: float=1e-1,
            kappa: float=10.0,
            solver_reg: float=1e-2,
            n_zeta_samples: int=10,
            cost: str="t-NLC-2-2",
            solver="entropic_torch",
            random_state: int=0
            ):

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
        self.random_state = random_state
        self.cost = cost

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
        cost = NormLabelCostTorch(p=2,power=2,kappa=self.kappa)



        custom_sampler = LabeledCostSampler(
                        cost,
                        pt.Tensor(emp.samples_x),
                        pt.Tensor(emp.samples_y),
                        epsilon=pt.tensor(self.rho),
                        seed=self.random_state
                    )



        if "torch" in self.solver:

            if self.solver == "entropic_torch" or self.solver == "entropic_torch_post":
                self.problem_ = WDROProblem(
                                    loss=DualLoss(
                                        WeberLoss(custom_sampler,d=d),
                                        cost,
                                        n_samples=self.n_zeta_samples,
                                        epsilon_0=pt.tensor(self.solver_reg),
                                        rho_0=pt.tensor(self.rho)),
                                    cost = cost,
                                    xi_bounds=[0,20],
                                    theta_bounds=[0,np.inf],
                                    rho=self.rho,
                                    p_hat=emp,
                                    d=d,
                                    d_labels=1,
                                    n=d
                                    )

            elif self.solver == "entropic_torch_pre":
                self.problem_ = WDROProblem(
                                    loss=DualPreSampledLoss(
                                        WeberLoss(custom_sampler,d=d),
                                        cost,
                                        n_samples=self.n_zeta_samples,
                                        epsilon_0=pt.tensor(self.solver_reg),
                                        rho_0=pt.tensor(self.rho)),
                                    cost = cost,
                                    xi_bounds=[0,20],
                                    theta_bounds=[0,np.inf],
                                    rho=self.rho,
                                    p_hat=emp,
                                    d=d,
                                    d_labels=1,
                                    n=d
                                    )

            else:
                raise NotImplementedError

            self.coef_, self.intercept_, self.dual_var_ = entTorch.solve_dual(
                    self.problem_,
                    seed=self.random_state,
                    sigma_=self.solver_reg,
                )
        else:
            raise NotImplementedError
        


        self.is_fitted_ = True

        self.n_features_in_ = d

        self.position_ = self.coef_

        # `fit` should always return `self`
        return self
