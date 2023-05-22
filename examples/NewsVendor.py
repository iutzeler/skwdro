"""
===========================
NewsVendor
===========================

An example plot of :class:`skwdro.Estimator.NewsVendor`
"""



from skwdro.operations_research import NewsVendor
import numpy as np
X = np.random.exponential(scale=2.0,size=(20,1))


print("Torch")
estimator = NewsVendor(solver="entropic_torch")
estimator.fit(X)

print(estimator.coef_,estimator.dual_var_)


print("Standard (entropic)")
estimator = NewsVendor()
estimator.fit(X)

print(estimator.coef_,estimator.dual_var_)

print("Dedicated")
estimator = NewsVendor(solver="dedicated")
estimator.fit(X)

print(estimator.coef_,estimator.dual_var_)


