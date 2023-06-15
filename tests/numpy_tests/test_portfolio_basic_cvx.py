'''
Basic test for the specific solver linked to the Mean-Risk Portfolio Problem.
We test with m = 2 assets, and only one of them gives money. The best 
decision thus is to invest all of the portfolio on the one that generates
benefit.
'''

import numpy as np
from skwdro.operations_research import Portfolio

def generate_data(N):
    X = np.array([1,0]) 
    return np.tile(X,(N,1)) #Duplicate the above line N times

def test_fit_low_radius():
    '''
    Fitting test with a low ambiguity Wasserstein ball radius.
    The decision expected will be to place everything on the first asset.
    '''

    X = generate_data(10)

    estimator = Portfolio(solver="dedicated", rho=1e-10)
    estimator.fit(X)

    theta = estimator.coef_

    #Assertions on optimal decisions
    assert np.isclose(theta[0], 1)
    assert np.isclose(theta[1], 0)

    #Assertions on optimal value
    assert np.isclose(estimator.problem_.loss.value(theta=theta, X=X), -theta[0])

def test_fit_high_radius():
    '''
    Fitting test with a high ambiguity Wasserstein ball radius.
    Due to the high ambiguity radius and the p-norm associated to the cost,
    Proposition 7.2 of DRO Kuhn 2017 leads us to expect to obtain the equally
    weighted portfolio decision.
    '''

    X = generate_data(10)

    estimator = Portfolio(solver="dedicated", rho=10)
    estimator.fit(X)

    theta = estimator.coef_

    #Assertions on optimal decisions
    assert np.isclose(theta[0], 0.5)
    assert np.isclose(theta[1], 0.5)

    #Assertions on optimal value
    assert np.isclose(estimator.problem_.loss.value(theta=theta, X=X), -theta[0])
