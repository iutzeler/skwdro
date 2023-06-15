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
    
def main():

    N = 100 #Number of samples

    #Create input: 2 assets with only one that gives us good returns
    X = pt.tensor([1,0]) 
    X = pt.tile(X,(N,1)) #Duplicate the above line N times

    #Creating the estimator and solving the problem
    estimator = Portfolio(solver="entropic_torch", reparam="softmax", rho=1e10, solver_reg=1e-3)
    estimator.fit(X)

    theta = estimator.coef_
    lam = estimator.dual_var_
    tau = estimator.problem_.loss.loss.tau.item()

    print("Value of theta: ", theta)
    print("Value of tau:", tau)
    print("Value of lambda: ", lam)

    filename = "test_post.npy"
    with open (filename, 'rb') as f:
        losses = np.load(f)
    f.close()

    indexes = np.array([i for i in range(len(losses))])

    print("Optimal value for the primal problem: ", estimator.problem_.loss.loss.value(X=X).mean())
    print("Optimal value for the dual problem: ", estimator.problem_.loss.forward(xi=X, xi_labels=None))

    plt.xlabel("Iterations")
    plt.ylabel("Loss value")
    plt.title("Evolution of loss value throughout the iterations")
    plt.plot(indexes, losses)
    plt.show()


if __name__ == "__main__":
    main()