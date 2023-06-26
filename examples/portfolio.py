"""
===================
Mean-Risk Portfolio
===================

An example of resolution of the mean-risk portfolio problem.
"""

import matplotlib.pyplot as plt
import numpy as np

from skwdro.operations_research import Portfolio

N = 10 #Number of samples

#Create input: 2 assets with only one that gives us good returns
X = np.array([1,0]) 
X = np.tile(X,(N,1)) #Duplicate the above line N times

print("Value of the samples:", X)

#Creating the estimator and solving the problem
estimator = Portfolio(solver="dedicated", rho=1e-10)
estimator.fit(X)

print("Value of C (after fitting):", estimator.C_)
print("Value of d (after fitting):", estimator.d_)

theta = estimator.coef_
lam = estimator.dual_var_

print("Value of theta: ", theta)
print("Value of lambda: ", lam)

print("Optimal value: ", estimator.problem_.loss.value(theta=theta,xi=X))
