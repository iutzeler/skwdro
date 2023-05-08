"""
WDRO Estimators
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances



from skwdro.base.problems import *
from skwdro.base.losses import *
from skwdro.base.losses_torch import *
from skwdro.base.costs import *

import skwdro.solvers.specific_solvers as spS
import skwdro.solvers.entropic_dual_solvers as entS
import skwdro.solvers.entropic_dual_torch as entTorch



class NewsVendor(BaseEstimator):
    """ A NewsVendor Wasserstein Distributionally Robust Estimator.

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
    cost: Loss, default=NormCost()
        Transport cost
    solver: str, default='entropic'
        Solver to be used: 'entropic' or 'dedicated' 

    Attributes
    ----------
    coef_ : float
        parameter vector (:math:`w` in the cost function formula)

    Examples
    --------
    >>> from skwdro.estimators import NewsVendor
    >>> import numpy as np
    >>> X = np.random.exponential(scale=2.0,size=(20,1))
    >>> estimator = NewsVendor()
    >>> estimator.fit(X)
    NewsVendor()
    """

    def __init__(self, rho = 1e-2,  k=5, u=7, cost = NormCost(), solver="entropic"):

        self.rho    = rho
        self.k      = k
        self.u      = u
        self.cost   = cost
        self.solver = solver

        self.problem = WDROProblem(d=1,Xi_bounds=[0,20],n=1,Theta_bounds=[0,np.inf],rho=rho,loss=NewsVendorLoss(k=k,u=u), cost = cost)

        if solver == "entropic_torch":
            self.problem.loss = NewsVendorLoss_torch(k=k,u=u)
        

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
        
        # TODO: assert X has the right shape


        m = np.size(X)
        emp = EmpiricalDistribution(m=m,samples=X)

        self.problem.P = emp

        if self.solver=="dedicated":
            self.coef_ = spS.WDRONewsvendorSolver(self.problem)
        elif self.solver=="entropic":
            self.coef_ , self.intercept_, self.dual_var_ = entS.WDROEntropicSolver(self.problem,epsilon=0.1)
        elif self.solver=="entropic_torch":
            self.coef_ , self.intercept_, self.dual_var_ = entTorch.WDROEntropicSolver(self.problem,epsilon=0.1)
        else:
            raise(NotImplementedError)


        self.is_fitted_ = True

        self.coef_ = float(self.coef_)

        # `fit` should always return `self`
        return self

