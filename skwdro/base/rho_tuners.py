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
        if "MeanRisk" in loss.__class__.__name__:
            self.theta_tau = nn.Parameter(pt.cat((self.loss.loss.theta_tilde.data, self.loss.tau.data), 0))

        for name, param in self.named_parameters(): #Set other parameters than theta to false
            if param.requires_grad:
                print(name)
                print(type(name))

        self.X = nn.Parameter(pt.as_tensor(X))
        self.y = nn.Parameter(pt.as_tensor(y), requires_grad=False) if y is not None else None

    def convert(self):
        return (self.X.data.float(), self.y) if self.y is None else \
            (self.X.data.float(), self.y.data.float().unsqueeze(-1).mean())

    def value(self):
        X_conv, y_conv = self.convert()
        return self.loss.value(xi=X_conv, xi_labels=y_conv)
    
    def value_idx(self, idx):

        #Torch conversions
        X_conv, y_conv = self.convert()
        idx = int(idx.item()) if type(idx) == pt.Tensor else idx

        print(idx)
        if y_conv is None:
            return self.loss.value(xi=X_conv[idx], xi_labels=None)
        return self.loss.value(xi=X_conv[idx], xi_labels=y_conv[idx])
    
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

        print("Output: ", output)
        print(output.size())

        class_name = self.estimator.__class__.__name__

        if "MeanRisk" not in class_name:
            diff_loss.loss.theta.retain_grad()
        else:
            diff_loss.loss.loss.theta_tilde.retain_grad()
            diff_loss.loss.tau.retain_grad()

        '''
        for k in range(self.n_samples_):
            print("Gradient: ", pt.autograd.grad(diff_loss.value_idx(idx=k), diff_loss.theta_tau, 
                                   grad_outputs=pt.ones_like(diff_loss.value_idx(idx=k)), allow_unused=True))
        '''
            
        #output.backward(retain_graph=True, gradient=pt.tensor([1 for _ in range(self.n_samples_)]).unsqueeze(-1))

        output[0].backward(retain_graph=True)
        grad_theta = diff_loss.loss.theta.grad.numpy().astype(float) if "MeanRisk" not in class_name \
                    else diff_loss.theta_tau.grad.numpy().astype(float)
        self.h_samples_ = np.array([grad_theta])

        for i in range(1,self.n_samples_):
            output[i].backward(retain_graph=True)
            #print("Theta tau: ", diff_loss.theta_tau.grad)
            #print("Theta tilde:", diff_loss.loss.loss.theta_tilde.grad)
            #print("Theta:", diff_loss.loss.loss.theta.grad)
            #print("Tau: ", diff_loss.loss.tau.grad)
            print("\n")
            #assert diff_loss.loss.theta.grad is not None, "Issue with the differentiation w.r.t theta"
            grad_theta = diff_loss.loss.theta.grad.numpy().astype(float) if "MeanRisk" not in class_name \
                        else diff_loss.theta_tau.grad.numpy().astype(float)
            self.h_samples_ = np.vstack((self.h_samples_, np.array([grad_theta])))

        #CASE OF NEWSVENDOR WITH ONLY ONE FEATURE!!
        #self.h_samples_ = np.squeeze(self.h_samples_)
        print("h_samples: ", self.h_samples_)

        self.cov_matrix_ = np.cov(m=self.h_samples_, bias=True) if self.h_samples_.shape[1] == 1 else np.cov(m=self.h_samples_)
        self.normal_samples_ = np.random.multivariate_normal(mean=np.array([0 for _ in range(self.n_samples_)]),
                                                            cov=self.cov_matrix_,
                                                            size=self.n_samples_)

        #Differentiate w.r.t theta and X. We thus compute the hessian matrix
        diff_loss.X.requires_grad = True

        #diff_loss.loss.theta.grad.zero_()

        print("PHASE 2: HESSIAN COMPUTATIONS")

        self.conjugate_samples_ =  pt.tensor([cpt.compute_phi_star(X=X, z=self.normal_samples_[i], 
                                                                  diff_loss=diff_loss)
                                                                for i in range(len(self.normal_samples_))])

        self.samples_quantile_ = pt.quantile(a=self.conjugate_samples_, q=confidence_level)

        #Compute rho thanks to the statistical analysis and the DRO estimator
        #Taking the square root as a transformation of rho as Blanchet uses a squared cost function
        self.estimator.rho = pt.sqrt((1/self.n_samples_)*self.samples_quantile_)
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
        print(self.n_samples_)

        self.h_samples_ = np.array([cpt.compute_h(xii=X[i], theta=self.theta_erm_, estimator=self.estimator)
                                    for i in range(self.n_samples_)])
        print(self.h_samples_.shape)

        self.cov_matrix_ = np.cov(self.h_samples_)

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










    
