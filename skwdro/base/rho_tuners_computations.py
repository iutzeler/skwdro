import numpy as np 
import torch as pt

from skwdro.operations_research import *
from skwdro.linear_models import *

def func_call(loss, X, y):
    return pt.func.functional_call(module=loss, parameter_and_buffer_dicts=dict(loss.named_parameters()), args=(X, y))

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
    
def compute_phi_star(X, y, z, diff_loss): 

    n_samples = len(X)

    hessian_products = []

    for k in range(n_samples):

        yk = y[k] if y is not None else None
        one_dim = diff_loss.value(X=X[k], y=yk).size() == pt.Size([1])
        
        #Needs to take a float input due to autograd restrictions even if index should be int
        hessian_loss = pt.func.hessian(func_call)(diff_loss, X[k], yk)
        print("Hessian value: ", hessian_loss)

        hessian_product = (hessian_loss)**2 if one_dim is True \
            else pt.matmul((hessian_loss).T,hessian_loss)
        hessian_products.append(hessian_product)

    A = (1/n_samples)*pt.sum(pt.tensor(hessian_products))

    print("Value of A:", A)
    print("Size of A: ", A.size())

    if one_dim is True: #A is a scalar
        if A != 0:
            alpha_opt = z/A
        else:
            return -pt.tensor([float("inf")]) if z != pt.tensor([0.]) else 0
    else:
        pseudo_inv_A = pt.linalg.pinv(A)
        alpha_opt = pseudo_inv_A@z
        if pt.isclose(A@alpha_opt, z) is False: #We consider in that case that z is not in range(A)
            return -pt.tensor([float("inf")])
        
    return alpha_opt.T@z - (1/2)*np.matmul(alpha_opt.T,A)@alpha_opt
            
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
