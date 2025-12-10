import numpy as np
from cvxopt import matrix, solvers
import cvxpy as cp

from skwdro.solvers.result import wrap_solver_result


@wrap_solver_result
def WDRONewsvendorSpecificSolver(k=5., u=7., rho=1.0, samples=None):
    assert samples is not None
    z = np.sort(samples, axis=0)
    n = z.shape[0]
    a = np.array([sum(z[:i, 0]) for i in range(n - 1)])
    b = np.array([n * rho - z[i + 1, 0] for i in range(n - 1)])
    c = np.array([n * rho] * (n - 1))

    lower_bound = b < a
    upper_bound = a <= c

    if not lower_bound.any():
        lambda_star = u
        return SAANewsvendorSpecificSolver2(k=k, u=u, samples=samples)
    elif not upper_bound.any():
        lambda_star = 0
        return 0, lambda_star
    else:
        condition = [upper_bound[i] and lower_bound[i] for i in range(n - 1)]
        i_star = condition.index(True) + 1

    s = np.minimum(z / z[i_star, 0], np.ones((n, 1)))

    T = u * rho / z[i_star, 0] + k - u * np.mean(s)

    if T >= 0:
        return 0.0, 0.
    else:
        return SAANewsvendorSpecificSolver2(k=k, u=u, samples=samples)


@wrap_solver_result
def SAANewsvendorSpecificSolver(k=5., u=7., samples=None):
    assert samples is not None

    z = np.sort(samples, axis=0)

    # Values useful for the following computations
    n = z.shape[0]
    i = np.ones((n, 1))
    o = np.zeros((n, 1))
    # oT = [0] * n
    I = np.eye(n)
    # O = np.zeros((n, n))

    #####################################################
    #           COMPUTE AND SOLVE LP PROBLEM
    #####################################################

    # ___________________ computing c ___________________

    c = np.vstack([0,
                   i / n])
    c = matrix(c)

    # ___________________ computing h ___________________

    h = np.vstack([o,
                   u * z])
    h = matrix(h)

    # ___________________ computing G ___________________

    G = np.vstack([np.hstack([(k - u) * i, -I]),
                   np.hstack([k * i, -I])])

    G = matrix(G)

    # _____________ solving the LP problem ______________

    solvers.options['show_progress'] = False
    solution = solvers.lp(c, G, h)
    theta = np.array(solution['x'])[0]
    # s = np.array(solution['x'])[1:n]
    dual_fun = np.array(solution['primal objective'])

    return theta, dual_fun


@wrap_solver_result
def SAANewsvendorSpecificSolver2(k=5., u=7., samples=None):
    assert samples is not None

    z = np.sort(samples, axis=0)

    n = z.size

    beta = cp.Variable(n + 1)

    loss = k * beta[n] - u * 1 / n * cp.sum(beta[:n])

    constraints = [beta[n] >= 0]
    for i in range(n):
        constraints.append(beta[i] <= beta[n])
        constraints.append(beta[i] <= z[i])

    problem = cp.Problem(cp.Minimize(loss), constraints=constraints)

    problem.solve(verbose=False)

    return beta.value[n], 0.


@wrap_solver_result
def WDROLogisticSpecificSolver(
        rho=1.0, kappa=1000, X=None, y=None, fit_intercept=False):
    assert X is not None and y is not None
    n, d = X.shape

    if fit_intercept:
        beta = cp.Variable(d + 1 + n + 1)

        loss = beta[d] * rho + 1 / n * cp.sum(beta[d + 1:d + 1 + n])

        constraints = [cp.norm(beta[:d]) <= beta[d]]
        for i in range(n):
            constraints.append(cp.logistic(
                y[i] * (X[i, :] @ beta[:d] + beta[d + 1 + n])) - kappa * beta[d] <= beta[d + 1 + i])
            constraints.append(
                cp.logistic(-y[i] * (X[i, :] @ beta[:d] + beta[d + 1 + n])) <= beta[d + 1 + i])

        problem = cp.Problem(cp.Minimize(loss), constraints=constraints)

        result = problem.solve(verbose=False)

        return beta.value[:d], beta.value[d + 1 + n], beta.value[d], result
    else:
        beta = cp.Variable(d + 1 + n)

        loss = beta[d] * rho + 1 / n * cp.sum(beta[d + 1:])

        constraints = [cp.norm(beta[:d]) <= beta[d]]
        for i in range(n):
            constraints.append(cp.logistic(
                y[i] * X[i, :] @ beta[:d]) - kappa * beta[d] <= beta[d + 1 + i])
            constraints.append(
                cp.logistic(-y[i] * X[i, :] @ beta[:d]) <= beta[d + 1 + i])

        problem = cp.Problem(cp.Minimize(loss), constraints=constraints)

        result = problem.solve(verbose=False)

        return beta.value[:d], 0.0, beta.value[d], result


@wrap_solver_result
def WDROLinRegSpecificSolver(rho: float = 1.0, X: np.ndarray = np.array(
        None), y: np.ndarray = np.array(None), fit_intercept: bool = False):
    n, d = X.shape

    assert rho > 0

    coeff = cp.Variable(d)
    intercept = cp.Variable(1)

    loss = cp.norm(X @ coeff + intercept - y, 2) / \
        cp.sqrt(n) + rho * (cp.norm(coeff))

    constraints = []

    if not fit_intercept:
        constraints.append(intercept == 0.0)

    problem = cp.Problem(cp.Minimize(loss), constraints=constraints)

    problem.solve(verbose=False)

    return coeff.value, intercept.value, None


@wrap_solver_result
def WDROPortfolioSpecificSolver(
        C, d, m, p, eta=.0, alpha=.95, rho=1.0, samples=None, fit_intercept=None):
    '''
    Solver for the dual program linked to Mean-Risk portfolio problem (Kuhn 2017).
    '''
    assert samples is not None

    # Problem data
    a = np.array([-1, -1 - eta / alpha])
    b = np.array([eta, eta * (1 - (1 / alpha))])
    N = samples.shape[0]
    K = 2

    # Decision variables of the problem
    lam = cp.Variable(1)
    s = cp.Variable(N)
    theta = cp.Variable(m)
    tau = cp.Variable(1)
    # The gamma[i][k] variables are vectors
    gamma = [cp.Variable(d.shape[0]) for _ in range(N * K)]

    # Objective function
    obj = lam * rho + (1 / N) * cp.sum(s)

    # Constraints
    constraints = [cp.sum(theta) == 1]

    constraints.append(lam >= 0)

    for j in range(m):
        constraints.append(theta[j] >= 0)

    if p != 1:
        q = 1 / (1 - (1 / p))
    elif p == 1:
        q = np.inf
        pass

    for i in range(N):
        xii_hat = samples[i]
        for k in range(K):
            constraints.append(b[k] * tau + a[k] * (theta @ xii_hat) +
                               (gamma[i * K + k] @ (d - (C @ xii_hat))) <= s[i])
            constraints.append(
                cp.norm((C.T) @ gamma[i * K + k] - a[k] * theta, q) <= lam)
            constraints.append(gamma[i * K + k] >= 0)

    # Solving the problem
    problem = cp.Problem(cp.Minimize(obj), constraints=constraints)
    result = problem.solve()

    if theta.value is None or np.isnan(sum(theta.value)):
        raise ValueError(
            "No solution exists for the Mean-Risk Portfolio problem")

    return theta.value, tau.value, lam.value, result
