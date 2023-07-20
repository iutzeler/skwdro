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
    X = pt.tensor([1.,0.]) 
    X = pt.tile(X,(N,1)) #Duplicate the above line N times

    #Creating the estimator and solving the problem
    estimator = Portfolio(solver="entropic_torch_post", reparam="softmax", n_zeta_samples=10*N, rho=1e10, solver_reg=1e-10)
    estimator.fit(X)

    theta = estimator.coef_
    lam = estimator.dual_var_
    tau = estimator.problem_.loss.primal_loss.tau.item()

    print("Value of theta: ", theta)
    print("Value of tau:", tau)
    print("Value of lambda: ", lam)

    filename = "test_post.npy" if estimator.solver == "entropic_torch_post" else "test_pre.npy"

    with open (filename, 'rb') as f:
        losses = np.load(f)
    f.close()

    indexes = np.array([i for i in range(len(losses))])

    print("Optimal value for the primal problem: ", estimator.problem_.loss.primal_loss.value(xi=X).mean())
    if estimator.solver == "entropic_torch_pre":
        print("Optimal value for the dual problem: ", estimator.problem_.loss.forward(xi=X, zeta=X.unsqueeze(0), zeta_labels=None, xi_labels=None))
    elif estimator.solver == "entropic_torch_post":
        print("Optimal value for the dual problem: ", estimator.problem_.loss.forward(xi=X, xi_labels=None))

    plt.xlabel("Iterations")
    plt.ylabel("Dual loss value")
    plt.title("Evolution of dual loss value throughout the iterations")
    plt.plot(indexes, losses)
    plt.show()


if __name__ == "__main__":
    main()
