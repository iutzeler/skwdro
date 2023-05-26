import numpy as np
from cvxopt import matrix, solvers
import cvxpy as cp 

from skwdro.base.costs import Cost, NormCost

def WDRONewsvendorSolver(WDROProblem):
    return WDRONewsvendorSpecificSolver(k=WDROProblem.loss.k,u=WDROProblem.loss.u,rho=WDROProblem.rho,samples=WDROProblem.P.samples)

def WDRONewsvendorSpecificSolver(k=5,u=7,rho=1.0,samples=None):
    z = np.sort(samples, axis=0) 
    n = z.shape[0]
    a = np.array([sum(z[:i, 0]) for i in range(n-1)])
    b = np.array([n * rho - z[i+1, 0] for i in range(n-1)])
    c = np.array([n * rho]*(n-1))

    lower_bound = b < a
    upper_bound = a <= c

    if not lower_bound.any():
        lambda_star = u 
        return SAANewsvendorSpecificSolver(k=k,u=u,samples=samples)
    elif not upper_bound.any():
        lambda_star = 0
        return 0
    else:
        condition = [upper_bound[i] and lower_bound[i] for i in range(n-1)]
        i_star = condition.index(True)+1
        
    s  = np.minimum( z/z[i_star, 0] , np.ones((n, 1)))    

    T = u * rho / z[i_star, 0] + k - u * np.mean(s)
        
    if T>=0:
        return 0.0
    else:
        return SAANewsvendorSpecificSolver(k=k,u=u,samples=samples)



def SAANewsvendorSolver(WDROProblem):
    return SAANewsvendorSpecificSolver2(k=WDROProblem.loss.k,u=WDROProblem.loss.u,samples=WDROProblem.P.samples)

def SAANewsvendorSpecificSolver(k=5,u=7,samples=None):

    z = np.sort(samples, axis=0) 

    # Values useful for the following computations
    n     = z.shape[0]
    i     = np.ones((n, 1))
    o     = np.zeros((n, 1))
    oT    = [0]*n
    I     = np.eye(n)
    O     = np.zeros((n, n))
    
    
    #####################################################
    #           COMPUTE AND SOLVE LP PROBLEM
    #####################################################
    
    # ___________________ computing c ___________________
    
    c = np.vstack([0, 
                   i/n])
    c = matrix(c)
    
    # ___________________ computing h ___________________
                    
    h = np.vstack([ o,
                    u*z])
    h = matrix(h)
    
    # ___________________ computing G ___________________

    G = np.vstack([np.hstack([ (k-u)*i, -I]), 
                   np.hstack([ k*i, -I])])
    
    G = matrix(G)
    
    # _____________ solving the LP problem ______________
    
    solvers.options['show_progress'] = False
    solution = solvers.lp(c, G, h)
    theta    = np.array(solution['x'])[0]
    s        = np.array(solution['x'])[1:n]
    dual_fun = np.array(solution['primal objective'])
    
    return theta

def SAANewsvendorSpecificSolver2(k=5,u=7,samples=None):

    z = np.sort(samples, axis=0) 

    n = z.size 

    beta = cp.Variable(n+1)

    loss = k*beta[n] - u*1/n*cp.sum(beta[:n])

    constraints = [beta[n]>=0]
    for i in range(n):
        constraints.append(beta[i]<=beta[n])
        constraints.append(beta[i]<=z[i])


    problem = cp.Problem(cp.Minimize(loss),constraints=constraints)

    problem.solve(verbose=False)



    return beta.value[n]




def WDROLogisticSolver(WDROProblem,fit_intercept=False):
    return WDROLogisticSpecificSolver(k=WDROProblem.loss.k,u=WDROProblem.loss.u,rho=WDROProblem.rho,samples=WDROProblem.P.samples)

def WDROLogisticSpecificSolver(rho=1.0,kappa=1000,X=None,y=None,fit_intercept=False):
    n,d = X.shape 

    if fit_intercept:
        beta = cp.Variable(d+1+n+1)

        loss = beta[d]*rho + 1/n*cp.sum(beta[d+1:d+1+n])

        constraints = [cp.norm(beta[:d])<=beta[d]]
        for i in range(n):
            constraints.append(cp.logistic(y[i]*(X[i,:]@beta[:d] + beta[d+1+n] ) )-kappa*beta[d]<=beta[d+1+i])
            constraints.append(cp.logistic(-y[i]*(X[i,:]@beta[:d] + beta[d+1+n] ) )<=beta[d+1+i])


        problem = cp.Problem(cp.Minimize(loss),constraints=constraints)

        problem.solve(verbose=False)

        return beta.value[:d], beta.value[d+1+n] , beta.value[d]
    else:
        beta = cp.Variable(d+1+n)

        loss = beta[d]*rho + 1/n*cp.sum(beta[d+1:])

        constraints = [cp.norm(beta[:d])<=beta[d]]
        for i in range(n):
            constraints.append(cp.logistic(y[i]*X[i,:]@beta[:d])-kappa*beta[d]<=beta[d+1+i])
            constraints.append(cp.logistic(-y[i]*X[i,:]@beta[:d])<=beta[d+1+i])


        problem = cp.Problem(cp.Minimize(loss),constraints=constraints)

        problem.solve(verbose=False)

        return beta.value[:d], 0.0 , beta.value[d]

def WDROPortfolioSolver(WDROProblem, C, d, eta, alpha, fit_intercept=None):
    return WDROPortfolioSpecificSolver(C=C, d=d, m=WDROProblem.n, cost=WDROProblem.cost, eta=eta, \
                                       alpha=alpha, rho=WDROProblem.rho, samples=WDROProblem.P.samples)


def WDROPortfolioSpecificSolver(C, d, m, cost, eta=0, alpha=.95, rho=1.0, samples=None, fit_intercept=None):
    '''
    Solver for the dual program linked to Mean-Risk portfolio problem (Kuhn 2017).
    '''

    #Problem data
    a = [-1, -1 - eta/alpha]
    b = [eta, eta(1-(1/alpha))]
    N = samples.size
    K = 2

    #Decision variables of the problem
    lam = cp.Variable(1)
    s = cp.Variable(N)
    theta = cp.Variable(m)
    tau = cp.Variable(1)
    gamma = cp.Variable(N*K)

    #Objective function
    obj = lam*rho + (1/N)*np.sum(s)

    #Constraints
    constraints = [np.sum(theta) == 1]

    for j in range(len(theta)):
        constraints.append(theta[j] >= 0)

    if isinstance(cost, NormCost): #Obtain the q-norm for the dual norm
        p = cost.p
        if p != 1:
            q = 1 - (1/p)
        elif p == 1:
            q = np.inf
    else:
        raise TypeError("Please define NormCost instance for cost attribute to define dual norm")

    for i in range(N):
        xii_hat = samples[i]
        for k in range(K):
            constraints.append(b[k]*tau + a[k]*np.dot(theta,xii_hat) + np.dot(gamma[i][k], d - np.dot(C,xii_hat)) <= s[i])
            constraints.append(cp.norm(np.dot(C.T,gamma[i][k]) - a[k]*theta, q) <= lam)
            constraints.append(gamma[i][k] >= 0)

    #Solving the problem
    problem = cp.Problem(cp.Minimize(obj), constraints=constraints)
    problem.solve(verbose=False)

    return theta, fit_intercept, lam

