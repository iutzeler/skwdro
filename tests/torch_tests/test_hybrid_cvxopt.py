'''
Basic test for the specific solver linked to the Mean-Risk Portfolio Problem.
We compare the optimal values returned by cvxopt and by our hybrid solver.
'''

N = 100 #Number of samples

import numpy as np
import torch as pt
from skwdro.operations_research import Portfolio

def generate_data(N, solver):
    if solver == "dedicated":
        X = np.array([1,0]) 
        return np.tile(X,(N,1)) #Duplicate the above line N times
    elif solver == "entropic_torch_post":
        X = pt.tensor([1.,0.]) 
        return pt.tile(X,(N,1)) #Duplicate the above line N times     

def test_fit_low_radius():
    '''
    Fitting test with a low ambiguity Wasserstein ball radius.
    The decision expected will be to place everything on the first asset.
    '''

    X_cvxopt = generate_data(10,"dedicated")

    estimator_cvxopt = Portfolio(solver="dedicated", cost="n-NC-1-1", rho=1e-10)
    estimator_cvxopt.fit(X_cvxopt)

    X_hybrid_opt = generate_data(10,"dedicated")

    estimator_hybrid_opt = Portfolio(solver="entropic_torch_post", cost="t-NC-1-1", reparam="none", \
                                    n_zeta_samples=10*N, rho=1e-10, solver_reg=1e-10)
    estimator_hybrid_opt.fit(X_hybrid_opt)

    theta_cvxopt = estimator_cvxopt.coef_
    theta_hybrid_opt = estimator_hybrid_opt.coef_

    print("theta cvxopt: ", estimator_cvxopt.coef_)
    print("theta hybrid_opt: ", estimator_hybrid_opt.coef_)

    #Assertions on optimal decisions
    #assert np.isclose(theta_cvxopt[0], theta_hybrid_opt[0], rtol=1e-3)
    #assert np.isclose(theta_cvxopt[1], theta_hybrid_opt[1], rtol=1e-3)

    print("Cvxopt value: ", estimator_cvxopt.problem_.loss.value(theta=theta_cvxopt, xi=X_cvxopt).double())
    print("Hybrid_opt value:", estimator_hybrid_opt.problem_.loss.primal_loss.value(xi=X_hybrid_opt).mean().double())

    #Assertions on optimal value
    assert pt.isclose(estimator_cvxopt.problem_.loss.value(theta=theta_cvxopt, xi=X_cvxopt).double(), estimator_hybrid_opt.problem_.loss.primal_loss.value(xi=X_hybrid_opt).mean().double())

def test_fit_high_radius():
    '''
    Fitting test with a high ambiguity Wasserstein ball radius.
    Due to the high ambiguity radius and the p-norm associated to the cost,
    Proposition 7.2 of DRO Kuhn 2017 leads us to expect to obtain the equally
    weighted portfolio decision.
    '''

    X_cvxopt = generate_data(10,"dedicated")

    estimator_cvxopt = Portfolio(solver="dedicated", cost="n-NC-1-1", rho=10)
    estimator_cvxopt.fit(X_cvxopt)

    X_hybrid_opt = generate_data(10,"entropic_torch_post")

    estimator_hybrid_opt = Portfolio(solver="entropic_torch_post", cost="t-NC-1-1", reparam="none", \
                                     n_zeta_samples=10*N, rho=10, solver_reg=10)
    estimator_hybrid_opt.fit(X_hybrid_opt)

    theta_cvxopt = estimator_cvxopt.coef_
    theta_hybrid_opt = estimator_hybrid_opt.coef_

    print("theta cvxopt: ", estimator_cvxopt.coef_)
    print("theta hybrid_opt: ", estimator_hybrid_opt.coef_)

    #Assertions on optimal decisions
    #assert np.isclose(theta_cvxopt[0], theta_hybrid_opt[0], rtol=1e-3)
    #assert np.isclose(theta_cvxopt[1], theta_hybrid_opt[1], rtol=1e-3)

    print("Cvxopt value: ", estimator_cvxopt.problem_.loss.value(theta=theta_cvxopt, xi=X_cvxopt).double())
    print("Hybrid_opt value:", estimator_hybrid_opt.problem_.loss.primal_loss.value(xi=X_hybrid_opt).mean().double())

    #Assertions on optimal value
    assert pt.isclose(estimator_cvxopt.problem_.loss.value(theta=theta_cvxopt, xi=X_cvxopt).double(), estimator_hybrid_opt.problem_.loss.primal_loss.value(xi=X_hybrid_opt).mean().double())