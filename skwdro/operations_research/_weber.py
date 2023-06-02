"""
Weber problem
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances



from skwdro.base.problems import WDROProblem, EmpiricalDistributionWithLabels
#from skwdro.base.losses import WeberLoss
from skwdro.base.losses_torch import WeberLoss_torch
from skwdro.base.costs import NormLabelCost

#import skwdro.solvers.specific_solvers as spS
import skwdro.solvers.entropic_dual_solvers as entS
import skwdro.solvers.entropic_dual_torch as entTorch



class Weber(BaseEstimator):
    """ A Weber Wasserstein Distributionally Robust Estimator.

    The cost function is XXX
    Uncertainty is XXX

    Parameters
    ----------
    rho : float, default=1e-2
        Robustness radius
    cost: Loss, default=NormCost()
        Transport cost
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

    def __init__(self, rho = 1e-1, cost = NormLabelCost(kappa=10.0), solver="entropic_torch"):

        self.rho    = rho
        self.cost   = cost
        self.solver = solver

        self.problem = WDROProblem(d=1,Xi_bounds=[0,20],n=1,Theta_bounds=[0,np.inf],rho=rho,loss=WeberLoss_torch(), cost = cost)
        

    def fit(self, X, w):
        """Fits a Weber WDRO model

        Parameters
        ----------
        X : array-like, shape (n_samples,2)
            The training input positions.
        w : array-like, shape (n_samples,) or (n_samples,1)
            The training input importance weights 

        Returns
        -------
        self : object
            Returns self.
        """
        
        # TODO: assert X has the right shape


        m = np.shape(X)[0]
        emp = EmpiricalDistributionWithLabels(m=m,samplesX=X,samplesY=w)

        self.problem.P = emp

        self.problem.d = 2
        self.problem.dLabel = 1
        self.problem.n = 2

        if self.solver=="entropic_torch":
            self.coef_ , _, self.dual_var_ = entTorch.WDROEntropicSolver(self.problem,epsilon=0.1)
        else:
            raise(NotImplementedError)


        self.is_fitted_ = True

        self.position_ = self.coef_

        # `fit` should always return `self`
        return self



