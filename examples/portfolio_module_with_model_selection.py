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
from sklearn.model_selection import GridSearchCV, KFold
    
def main():

    N = 100 #Number of samples

    #Create input: 2 assets with only one that gives us good returns
    X = pt.tensor([1.,0.]) 
    X = pt.tile(X,(N,1)) #Duplicate the above line N times

    #Creating the estimator and solving the problem
    estimator = Portfolio(solver="entropic_torch_post", reparam="none", n_zeta_samples=10*N, rho=1e-10, solver_reg=1e-10)
    #estimator.fit(X)

    print("Estimator params: ", estimator.get_params)

    #Tuning rho using grid search
    param_grid = {"rho": [10**(-i) for i in range(4,-4,-1)]}
    #TODO: KFold(n_splits=N) is equiavelnt to LeaveOneOut (needs to be clearer for me). Do I need to shuffle in my case?
    # Shuffle can cause overfitting in some situations 
    grid_cv = KFold(n_splits=N, shuffle=True)

    grid_estimator= GridSearchCV(estimator=estimator, param_grid=param_grid, cv=grid_cv)
    print("Grid estimator params: ", grid_estimator.get_params) #Compare parameters with estimator
    grid_estimator.fit(X) #Fit on the new estimator with tuned parameters?

    theta = grid_estimator.coef_
    lam = grid_estimator.dual_var_
    tau = grid_estimator.problem_.loss.loss.tau.item()

    print("Value of theta: ", theta)
    print("Value of tau:", tau)
    print("Value of lambda: ", lam)

    filename = "test_post.npy" if grid_estimator.solver == "entropic_torch_post" else "test_pre.npy"
    #TODO: Maybe try to get the evolution of the primal loss value throughout the iterations
    with open (filename, 'rb') as f:
        losses = np.load(f)
    f.close()

    indexes = np.array([i for i in range(len(losses))])

    print("Optimal value for the primal problem: ", grid_estimator.problem_.loss.loss.value(X=X).mean())
    if grid_estimator.solver == "entropic_torch_pre":
        print("Optimal value for the dual problem: ", grid_estimator.problem_.loss.forward(xi=X, zeta=X.unsqueeze(0), zeta_labels=None, xi_labels=None))
    elif grid_estimator.solver == "entropic_torch_post":
        print("Optimal value for the dual problem: ", grid_estimator.problem_.loss.forward(xi=X, xi_labels=None))

    plt.xlabel("Iterations")
    plt.ylabel("Dual loss value")
    plt.title("Evolution of dual loss value throughout the iterations")
    plt.plot(indexes, losses)
    plt.show()


if __name__ == "__main__":
    main()
