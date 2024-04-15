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
from skwdro.base.rho_tuners import *

def main():

    N = 10 #Number of samples

    #Create input: 2 assets with only one that gives us good returns
    X = pt.tensor([1.,0.]) 
    X = pt.tile(X,(N,1)) #Duplicate the above line N times

    #Creating the estimator and solving the problem
    estimator = Portfolio(n_zeta_samples=10*N)

    rho_tuner = RhoTunedEstimator(estimator)
    rho_tuner.fit(X=X, y=None)

    best_estimator = rho_tuner.best_estimator_
    print("Best estimator: ", best_estimator)

    theta = best_estimator.coef_
    lam = best_estimator.dual_var_
    tau = best_estimator.tau_

    print("Value of theta: ", theta)
    print("Value of tau:", tau)
    print("Value of lambda: ", lam)

    filename = "test_post.npy" if best_estimator.solver == "entropic_torch_post" else "test_pre.npy"
    with open (filename, 'rb') as f:
        losses = np.load(f)
    f.close()

    indexes = np.array([i for i in range(len(losses))])

    print("Optimal value for the primal problem: ", best_estimator.loss.primal_loss.value(xi=X).mean())
    if best_estimator.solver == "entropic_torch_pre":
        print("Optimal value for the dual problem: ", best_estimator.loss.forward(xi=X, zeta=X.unsqueeze(0), zeta_labels=None, xi_labels=None))
    elif best_estimator.solver == "entropic_torch_post":
        print("Optimal value for the dual problem: ", best_estimator.loss.forward(xi=X, xi_labels=None))

    plt.xlabel("Iterations")
    plt.ylabel("Dual loss value")
    plt.title("Evolution of dual loss value throughout the iterations")
    plt.plot(indexes, losses)
    plt.show()


if __name__ == "__main__":
    main()
