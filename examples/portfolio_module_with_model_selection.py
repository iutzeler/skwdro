"""
===================
Mean-Risk Portfolio
===================

An example of resolution of the mean-risk portfolio problem.
"""
import torch as pt
import numpy as np
import matplotlib.pyplot as plt

from skwdro.operations_research import Portfolio

from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV, KFold
from sklearn.metrics import make_scorer

def main():

    N = 100 #Number of samples

    #Create input: 2 assets with only one that gives us good returns
    X = pt.tensor([1.,0.]) 
    X = pt.tile(X,(N,1)) #Duplicate the above line N times

    #Creating the estimator and solving the problem
    estimator = Portfolio(solver="entropic_torch_post", reparam="softmax", n_zeta_samples=10*N)

    print("Estimator params: ", estimator.get_params)

    #Tuning rho using grid search
    param_grid = {"rho": [10**(-i) for i in range(4,-4,-1)]}
    grid_cv = KFold(n_splits=5, shuffle=True)

    grid_estimator= GridSearchCV(estimator=estimator, param_grid=param_grid, cv=grid_cv, refit=True, n_jobs=-1, verbose=3)
    #grid_estimator= HalvingGridSearchCV(estimator=estimator, param_grid=param_grid, cv=grid_cv,n_jobs=-1, refit=True, verbose=3, min_resources="smallest")

    grid_estimator.fit(X) #Fit on the new estimator

    best_params = grid_estimator.best_params_
    best_score = grid_estimator.best_score_

    print("Best params: ", best_params)
    print("Best score: ", best_score)

    best_estimator = grid_estimator.best_estimator_
    best_estimator.solver_reg = best_params['rho']

    theta = best_estimator.coef_
    lam = best_estimator.dual_var_
    tau = best_estimator.problem_.loss.loss.tau.item()

    print("Value of theta: ", theta)
    print("Value of tau:", tau)
    print("Value of lambda: ", lam)

    filename = "test_post.npy" if best_estimator.solver == "entropic_torch_post" else "test_pre.npy"
    with open (filename, 'rb') as f:
        losses = np.load(f)
    f.close()

    indexes = np.array([i for i in range(len(losses))])

    print("Optimal value for the primal problem: ", best_estimator.problem_.loss.loss.value(X=X).mean())
    if best_estimator.solver == "entropic_torch_pre":
        print("Optimal value for the dual problem: ", best_estimator.problem_.loss.forward(xi=X, zeta=X.unsqueeze(0), zeta_labels=None, xi_labels=None))
    elif best_estimator.solver == "entropic_torch_post":
        print("Optimal value for the dual problem: ", best_estimator.problem_.loss.forward(xi=X, xi_labels=None))

    plt.xlabel("Iterations")
    plt.ylabel("Dual loss value")
    plt.title("Evolution of dual loss value throughout the iterations")
    plt.plot(indexes, losses)
    plt.show()


if __name__ == "__main__":
    main()
