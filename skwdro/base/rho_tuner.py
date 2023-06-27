from sklearn.base import BaseEstimator

import joblib as jb

from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV, KFold

from dask.distributed import Client 

class RhoTunedEstimator(BaseEstimator):
    """A custom estimator that tunes Wasserstein radius rho."""

    def __init__(self,
                 estimator):
        self.estimator = estimator

    def fit(self, X, y):
        #Tuning rho using grid search
        param_grid_ = {"rho": [10**(-i) for i in range(4,-4,-1)]}
        grid_cv_ = KFold(n_splits=5, shuffle=True)

        client_ = Client(processes=False) 

        grid_estimator_= GridSearchCV(estimator=self.estimator, param_grid=param_grid_, cv=grid_cv_, 
                                     refit=True, n_jobs=-1, verbose=3)
        #grid_estimator= HalvingGridSearchCV(estimator=self, param_grid=param_grid_, cv=grid_cv_,
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