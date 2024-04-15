"""
===========================
Weber
===========================

An example of :class:`skwdro.Estimator.Weber`
"""



from skwdro.operations_research import Weber

import numpy as np
import matplotlib.pyplot as plt

m = 3

X = np.random.randn(m,2)*10
w = np.random.exponential(size=(m,),scale=10)


print(X,w)

print("Torch")
estimator = Weber(rho=10.0)
estimator.fit(X,w)

print(estimator.position_)


print("IRLS")

y = np.random.rand(1,2)

T = 100
for t in range(T):
    num = np.zeros(y.shape)
    den = 0.0
    for i in range(m):
        num += w[i]*X[i,:]/np.linalg.norm(X[i,:]-y)
        den += w[i]/np.linalg.norm(X[i,:]-y)
     
    y = num/den

print(y)


plt.scatter(X[:,0],X[:,1],s=w*10,label="Facilities")
plt.scatter(estimator.position_[0],estimator.position_[1],s=np.max(w)*10,label="Classical")
plt.scatter(y[:,0],y[:,1],s=np.max(w)*10,label="Robust")
plt.legend()
plt.show()