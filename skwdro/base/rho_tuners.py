from sklearn.base import BaseEstimator

import joblib as jb
import numpy as np

from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV, KFold

from dask.distributed import Client 

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

        grid_estimator_= GridSearchCV(estimator=self.estimator, param_grid=param_grid_, cv=grid_cv_, 
                                       refit=True, n_jobs=-1, verbose=3)
        #grid_estimator_= HalvingGridSearchCV(estimator=self.estimator, param_grid=param_grid_, cv=grid_cv_,
        #                            refit=True, n_jobs=-1, verbose=3, min_resources="smallest")
 
        with jb.parallel_backend("dask", scatter=[X, y]):  
            grid_estimator_.fit(X, y) #Fit on the new estimator

        best_params_ = grid_estimator_.best_params_
        best_score_ = grid_estimator_.best_score_

        print("Best params: ", best_params_)
        print("Best score: ", best_score_)

        self.best_estimator_ = grid_estimator_.best_estimator_
        print("Solver reg value: ",self.best_estimator_.solver_reg)  

        return self
    
class BlanchetRhoTunedEstimator(BaseEstimator):
    """A custom estimator based on statistical analysis (Blanchet 2021) 
    that tunes Wasserstein radius rho."""

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
        self.normal_samples_ = np.random.multivariate_normal(mean=np.array([0 for _ in range(self.n_samples_)]),
                                                            cov=self.cov_matrix_,
                                                            size=self.n_samples_)
        self.conjugate_samples_ =  np.array([cpt.compute_phi_star(X=X, z=self.normal_samples_[i], 
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










    
