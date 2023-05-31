"""
===================
Mean-Risk Portfolio
===================

An example of resolution of the mean-risk portfolio problem.
"""

import matplotlib.pyplot as plt
import numpy as np

from skwdro.operations_research import Portfolio
from skwdro.base.costs import NormCost

N = 10 #Number of samples

#Create input: 2 assets with only one that gives us good returns
X = np.array([1,0]) 
X = np.tile(X,(N,1)) #Duplicate the above line N times

print("Value of the samples:", X)

#Giving conditions on xi
C = np.array([[0,0],[0,0]])
d = np.array([1,1])

#Creating the estimator and solving the problem
estimator = Portfolio(solver="dedicated", rho=1e-10)
estimator.fit(X)

print("Value of C (after fitting):", estimator.C_)
print("Value of d (after fitting):", estimator.d_)

print("Value of theta: ", estimator.coef_)
print("Value of lambda: ", estimator.dual_var_)