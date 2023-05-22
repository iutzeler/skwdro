"""
===================
Linear regression
===================

An example of linear regression.
"""

import numpy as np
from sklearn.model_selection import train_test_split

from skwdro.linear_models import LinearRegression


d = 10
m = 100

x0 = np.random.randn(d)

X = np.random.randn(m,d)

y = X.dot(x0) +  np.random.randn(m)

X_train, X_test, y_train, y_test = train_test_split(X,y)


rob_lin = LinearRegression()

rob_lin.fit(X_train, y_train)

rob_lin.score(X_test,y_test)
