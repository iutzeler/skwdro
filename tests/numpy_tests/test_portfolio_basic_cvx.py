'''
Basic test for the specific solver linked to the Mean-Risk Portfolio Problem.
We test with m = 2 assets, and only one of them gives money. The best 
decision thus is to invest all of the portfolio on the one that generates
benefit.
'''

import numpy as np
from skwdro.operations_research import Portfolio

def generate_data(N, val_1, val_2):
    X = np.array([val_1,val_2]) 
    return np.tile(X,(N,1)) #Duplicate the above line N times

def test_fit_low_radius():
    '''
    Fitting test with a low ambiguity Wasserstein ball radius.
    The decision expected will be to place everything on the first asset.
    '''

    X = generate_data(10,1,0)

    estimator = Portfolio(solver="dedicated", rho=1e-10)
    estimator.fit(X)

    theta = estimator.coef_

    #Assertions on optimal decisions
    assert np.isclose(theta[0], 1)
    assert np.isclose(theta[1], 0)

    #Assertions on optimal value
    assert np.isclose(estimator.problem_.loss.value(theta=theta, xi=X), -theta[0])

def test_fit_high_radius():
    '''
    Fitting test with a high ambiguity Wasserstein ball radius.
    Due to the high ambiguity radius and the p-norm associated to the cost,
    Proposition 7.2 of DRO Kuhn 2017 leads us to expect to obtain the equally
    weighted portfolio decision.
    '''

    X = generate_data(10,1,0)

    estimator = Portfolio(solver="dedicated", rho=10)
    estimator.fit(X)

    theta = estimator.coef_

    #Assertions on optimal decisions
    assert np.isclose(theta[0], 0.5)
    assert np.isclose(theta[1], 0.5)

    #Assertions on optimal value
    assert np.isclose(estimator.problem_.loss.value(theta=theta, xi=X), -theta[0])

def test_fit_with_polyhedron_constraints_low_radius():
    '''
    Fitting test with a low ambiguity Wasserstein radius and polyhedron constraints 
    on the input data. The first asset cannot give us more than 0.2 and the second one cannot give
    us more than 0.5. Since the observed data lead us to observe that these constraints are binding,
    the best expected decision thus is to put everything on the second asset that gives us more money.
    '''

    X = generate_data(10,0.2,0.5)

    C = np.eye(2)
    d = np.array([0.2,0.5])

    estimator = Portfolio(solver="dedicated", rho=1e-10, C=C, d=d)

    estimator.fit(X)

    theta = estimator.coef_

    #Assertions on optimal decisions
    assert np.isclose(theta[0], 0)
    assert np.isclose(theta[1], 1)

def test_fit_with_polyhedron_constraints_high_radius():
    '''
    Fitting test with a high ambiguity Wasserstein radius and polyhedron constraints 
    on the input data. Checks if the equally weighted portfolio stills is the best decision
    even with constraints on X.
    '''

    X = generate_data(10,0.2,0.5)

    C = np.eye(2)
    d = np.array([0.2,0.5])

    estimator = Portfolio(solver="dedicated", rho=10, C=C, d=d)

    estimator.fit(X)

    theta = estimator.coef_

    #Assertions on optimal decisions
    assert np.isclose(theta[0], 0.5)
    assert np.isclose(theta[1], 0.5)

def test_fit_no_solution():
    '''
    Fitting test that raises a ValueError if no solution exists for the problem.
    Here, we observe data on the second asset that gives 0.6, and it contradicts the fact that
    the second asset gives at most 0.5. The specific solver gives a ValueError if no decision exists.
    '''

    X = generate_data(10,0.2,0.6)

    C = np.eye(2)
    d = np.array([0.2,0.5])

    estimator = Portfolio(solver="dedicated", rho=1e-10, C=C, d=d)

    np.testing.assert_raises(ValueError, estimator.fit, X)