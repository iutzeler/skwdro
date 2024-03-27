"""
===================
Linear regression
===================

An example of linear regression.
"""

import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from skwdro.linear_models import LinearRegression as RobustLinearRegression


d = 10
m = 100

x0 = np.random.randn(d)

X = np.random.randn(m,d)

y = X.dot(x0) +  np.random.randn(m)

X_train, X_test, y_train, y_test = train_test_split(X,y)


rob_lin = RobustLinearRegression(rho=0.1,solver="entropic",fit_intercept=True)

try: 
    rob_lin.fit(X_train, y_train)
    y_pred_rob = rob_lin.predict(X_test)
except(DeprecationWarning):
    pass

rob_lin2 = RobustLinearRegression(rho=0.1,solver="entropic_torch",fit_intercept=True)
rob_lin2.fit(X_train, y_train)
y_pred_rob2 = rob_lin2.predict(X_test)

rob_lin3 = RobustLinearRegression(rho=0.1,solver="dedicated",fit_intercept=True)
rob_lin3.fit(X_train, y_train)
y_pred_rob3 = rob_lin3.predict(X_test)


lin = LinearRegression()
lin.fit(X_train, y_train)
y_pred = lin.predict(X_test)




print(f"sklearn error: {np.linalg.norm(y_pred-y_test)} " )
#print(f"skwdro error w/entropic: {np.linalg.norm(y_pred_rob-y_test)}")
print(f"skwdro error w/torch: {np.linalg.norm(y_pred_rob2-y_test)}")
print(f"skwdro error w/dedicated: {np.linalg.norm(y_pred_rob3-y_test)}")