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


rob_lin = RobustLinearRegression(rho=0.1)
rob_lin.fit(X_train, y_train)
y_pred_rob = rob_lin.predict(X_test)

lin = LinearRegression()
lin.fit(X_train, y_train)
y_pred = lin.predict(X_test)


print(f"sklearn error: {np.linalg.norm(y_pred-y_test)} " )
print(f"skwdro error: {np.linalg.norm(y_pred_rob-y_test)}")