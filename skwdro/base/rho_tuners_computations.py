import numpy as np 
from skwdro.operations_research import *
from skwdro.linear_models import *

def indicator_func(xii, theta, estimator):

    if isinstance(estimator, Portfolio):
        tau = estimator.tau_ if (isinstance(estimator.tau_, (np.ndarray,np.generic))) \
            else estimator.tau_
        result = np.sign(-(theta@np.array(xii) + tau))
        assert result != 0, "The loss function is not differentiable"
        return result if result == 1 else 0
    else:
        raise NotImplementedError()
    
def scal_xi(xii, a):
    return np.append(xii, a)

def compute_h(xii, theta, estimator):
    #Computes the gradient of the loss on theta (only for the Portfolio problem right now)
    
    if isinstance(estimator, Portfolio):

        eta = estimator.eta_
        alpha = estimator.alpha_

        xii_minus_eta = scal_xi(xii=xii, a=-eta)
        xii_1 = scal_xi(xii=xii, a=1)
        return - (xii_minus_eta + (eta/alpha)*xii_1*indicator_func(xii=xii, 
                    theta=theta, estimator=estimator))
    
    else:
        raise NotImplementedError()
    
def compute_phi_star(X, z, diff_loss): 

    n_samples = len(X)
    A = (1/n_samples)*np.sum([np.matmul((diff_loss.value(idx=k)).T,diff_loss.value(idx=k))
            for k in range(n_samples)])
    return np.linalg.norm(x=z, ord=2)/np.matmul(z.T,A)@z
    
def compute_phi_star_portfolio(X, z, theta, estimator):

    if isinstance(estimator, Portfolio):
        eta = estimator.eta_
        alpha = estimator.alpha_       

        q = estimator.cost_.p #The norm of the cost function is denoted q in Blanchet's paper
        n_samples = len(X)
        C = sum([1 + (eta/alpha)*indicator_func(xii=X[k], theta=theta, estimator=estimator) for k in range(n_samples)])

        return (n_samples*np.linalg.norm(x=z, ord=q)**2)/C
    else:
        raise NotImplementedError()
