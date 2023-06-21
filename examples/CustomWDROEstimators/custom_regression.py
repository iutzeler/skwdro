"""
===================
Custom regression
===================

Example of a custom WDRO regression.
"""

import numpy as np
from sklearn.model_selection import train_test_split


from my_WDRO_regressor import MyWDRORegressor


d = 10
m = 100

x0 = np.random.randn(d)

X = np.random.randn(m,d)

y = X.dot(x0) +  np.random.randn(m)

X_train, X_test, y_train, y_test = train_test_split(X,y)


custom_est = MyWDRORegressor(rho=0.1,solver="entropic_torch",fit_intercept=True) # Pre-sampled version
# custom_est = MyWDRORegressor(rho=0.1,solver="entropic_torch_post",fit_intercept=True) # Re-sampled version
custom_est.fit(X_train, y_train)
y_pred = custom_est.predict(X_test)

print(f"Score for the custom WDRO estimator: {np.linalg.norm(y_pred-y_test)}")