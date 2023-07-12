from sklearn.base import BaseEstimator

import joblib as jb
import numpy as np

import torch.nn as nn
import torch as pt

from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV, KFold

from dask.distributed import Client 

from skwdro.base.losses_torch.base_loss import Loss
import skwdro.base.rho_tuners_computations as cpt

from skwdro.operations_research import *
from skwdro.linear_models import *

class RhoTunedEstimator(BaseEstimator):
    """A custom estimator that tunes Wasserstein radius rho."""

    def __init__(self,
                 estimator,
                 min_power=-4,
                 max_power=4):
        
        if estimator is None:
            raise ValueError("Estimator cannot be None for rho tuning")

        self.estimator = estimator
        self.min_power = min_power
        self.max_power = max_power

    def fit(self, X, y):

        #Verify that estimator has a score method
        assert hasattr(self.estimator, "score"), "Score method not found for estimator for rho tuning"

        #Fitting on the estimator
        self.estimator.fit(X,y)

        #Tuning rho using grid search
        param_grid_ = {"rho": [10**(-i) for i in range(self.max_power,self.min_power,-1)]}
        grid_cv_ = KFold(n_splits=5, shuffle=False)

        client_ = Client(processes=False) 

        #grid_estimator_= GridSearchCV(estimator=self.estimator, param_grid=param_grid_, cv=grid_cv_, 
        #                               refit=True, n_jobs=-1, verbose=3, error_score="raise")
        grid_estimator_= HalvingGridSearchCV(estimator=self.estimator, param_grid=param_grid_, cv=grid_cv_,
                                    refit=True, n_jobs=-1, verbose=3, min_resources="smallest",
                                    error_score="raise")
 
        with jb.parallel_backend("dask", scatter=[X, y]):  
            grid_estimator_.fit(X, y) #Fit on the new estimator

        best_params_ = grid_estimator_.best_params_
        best_score_ = grid_estimator_.best_score_

        print("Best params: ", best_params_)
        print("Best score: ", best_score_)

        self.best_estimator_ = grid_estimator_.best_estimator_
        print("Solver reg value: ",self.best_estimator_.solver_reg)  

        return self
    
class DiffLoss(Loss):
    """Intermediary loss created from another loss.
    Useful for differentation on theta and X for Blanchet's algorithm."""

    def __init__(self, loss, X, y):
        super(DiffLoss, self).__init__()
        self.loss = loss

        #In this case we have to treat theta and tau as one parameter
        '''
        if "Portfolio" in loss.__class__.__name__:
            pass
        '''

        self.X = nn.Parameter(pt.as_tensor(X), requires_grad=False)
        self.y = nn.Parameter(pt.as_tensor(y), requires_grad=False) if y is not None else None
    def value(self):
        return self.loss.value(xi=self.X, xi_labels=self.y)
    
    def theta(self) -> pt.Tensor:
        return self._theta
    
class BlanchetRhoTunedEstimator(BaseEstimator):
    """A custom general estimator based on statistical analysis (Blanchet 2021) 
    that tunes Wasserstein radius rho. Takes norm cost equal to 2.
    Every differential computations are done using torch's autodiff methods."""

    def __init__(self, estimator):

        if estimator is None:
            raise ValueError("Estimator cannot be None for rho tuning")

        self.estimator = estimator

    def fit(self, X, y):

        #Confidence level for the presence of a minimizer in the Wasserstein ball
        confidence_level = 0.95

        #Creating ERM decision: for that we fix rho = 0 to solve the SAA problem

        rho = 0 if self.estimator.solver == "dedicated" else 0.1 #TODO: Adapt value of rho for entropic cases
        
        self.estimator.rho = rho
        self.estimator.fit(X,y)

        self.theta_erm_ = self.estimator.coef_

        #Data-driven evaluations for the estimation of rho

        self.n_samples_ = len(X)

        #Creation of an intermediary loss for differentiation
        loss = self.estimator.problem_.loss.primal_loss
        diff_loss = DiffLoss(loss=loss, X=X, y=y)
        output = diff_loss.value()

        print("Output value: ", diff_loss.value().detach().numpy())

        diff_loss.loss.theta.retain_grad()

        #output.backward(retain_graph=True, gradient=pt.tensor([1 for _ in range(self.n_samples_)]).unsqueeze(-1))

        for name, param in diff_loss.named_parameters():
            if param.requires_grad:
                print(name, param.data)

        output[0].backward(retain_graph=True)
        self.h_samples_ = np.array([diff_loss.loss.theta.grad.numpy().astype(float)])
        print("Before: ", self.h_samples_)

        for i in range(1,self.n_samples_):
            output[i].backward(retain_graph=True)
            assert diff_loss.loss.theta.grad is not None, "Issue with the differentiation w.r.t theta"
            self.h_samples_ = np.vstack((self.h_samples_, np.array([diff_loss.loss.theta.grad.numpy().astype(float)])))

        print("After: ", self.h_samples_)
        print(self.h_samples_.shape)
        print(type(self.h_samples_[0][0]))
            
        self.cov_matrix_ = np.cov(m=self.h_samples_, bias=True) if self.h_samples_.shape[1] == 1 else np.cov(m=self.h_samples_)
        print("Cov matrix:", self.cov_matrix_)
        self.normal_samples_ = np.random.multivariate_normal(mean=np.array([0 for _ in range(self.n_samples_)]),
                                                            cov=self.cov_matrix_,
                                                            size=self.n_samples_)

        #We now differentiate w.r.t X
        diff_loss.X.requires_grad = True
        diff_loss.loss._theta.requires_grad = False

        for name, param in diff_loss.named_parameters():
            if param.requires_grad:
                print(name, param.data)

        output.backward(gradient=pt.as_tensor(self.h_samples_)) #To adapt

        print(diff_loss.X.grad)

        assert any(parameter.grad is not None for parameter in diff_loss.parameters())
        self.conjugate_samples_ =  np.array([cpt.compute_phi_star(X=X, z=self.normal_samples_[i], 
                                                                  diff_loss=diff_loss)
                                                                for i in range(len(self.normal_samples_))])

        self.samples_quantile_ = np.quantile(a=self.conjugate_samples_, q=confidence_level)

        #Compute rho thanks to the statistical analysis and the DRO estimator
        #Taking the square root as a transformation of rho as Blanchet uses a squared cost function
        self.estimator.rho = np.sqrt((1/self.n_samples_)*self.samples_quantile_)
        self.estimator.fit(X,y)

        self.best_estimator_ = self.estimator

        return self
    
class PortfolioBlanchetRhoTunedEstimator(BaseEstimator):
    """A custom portfolio estimator based on statistical analysis (Blanchet 2021) 
    that tunes Wasserstein radius rho. Explicit formulae takes norm cost equal to 1."""

    def __init__(self, estimator):

        if estimator is None:
            raise ValueError("Estimator cannot be None for rho tuning")

        self.estimator = estimator

    def fit(self, X, y):

        #Confidence level for the presence of a minimizer in the Wasserstein ball
        confidence_level = 0.95

        #Creating ERM decision: for that we fix rho = 0 to solve the SAA problem

        rho = 0 if self.estimator.solver == "dedicated" else 0.1 #TODO: Adapt value of rho for entropic cases
        
        self.estimator.rho = rho
        self.estimator.fit(X,y)

        self.theta_erm_ = self.estimator.coef_

        #Data-driven evaluations for the estimation of rho

        self.n_samples_ = len(X)

        self.h_samples_ = np.array([cpt.compute_h(xii=X[i], theta=self.theta_erm_, estimator=self.estimator)
                                    for i in range(self.n_samples_)])

        self.cov_matrix_ = np.cov(self.h_samples_)
        print("Cov matrix:", self.cov_matrix_)
        print(self.cov_matrix_.shape)

        self.normal_samples_ = np.random.multivariate_normal(mean=np.array([0 for _ in range(self.n_samples_)]),
                                                            cov=self.cov_matrix_,
                                                            size=self.n_samples_)
        self.conjugate_samples_ =  np.array([cpt.compute_phi_star_portfolio(X=X, z=self.normal_samples_[i], 
                                                                  theta=self.theta_erm_, 
                                                                  estimator=self.estimator)
                                                                for i in range(len(self.normal_samples_))])
        
        self.samples_quantile_ = np.quantile(a=self.conjugate_samples_, q=confidence_level)

        #Compute rho thanks to the statistical analysis and the DRO estimator
        #Taking the square root as a transformation of rho as Blanchet uses a squared cost function
        self.estimator.rho = np.sqrt((1/self.n_samples_)*self.samples_quantile_)
        self.estimator.fit(X,y)

        self.best_estimator_ = self.estimator

        return self










    
