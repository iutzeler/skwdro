r"""
Linear regression
=====================
This example illustrates the use of the :class:`skwdro.linear_models.LinearRegression` class to perform a simple Wasserstein distributionally robust linear regression.

The samples are of the form :math:`\xi = (x,y) \in \mathbb{R}\times\mathbb{R}` and the sought predictor is linear.

"""
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_regression
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from skwdro.linear_models import LinearRegression

# %%
# Problem setup
# ~~~~~~~~~~~~~

n = 100 # Total number of samples
n_train = int(np.floor(0.8 * n)) # Number of training samples
n_test = n - n_train # Number of test samples

# Generate some data
X, y = make_regression(n_samples=n, n_features=1, noise=50, random_state=0)

# Normalize the data
X = minmax_scale(X, feature_range=(-1, 1))
y = minmax_scale(y, feature_range=(-1, 1))

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=n_train, test_size=n_test, random_state=0)

# %%
# WDRO linear regression
# ~~~~~~~~~~~~~~~~~~~~~~

# Define a range of radii, rho=0 corresponds to the standard linear regression
rhos = [0., 1e-2, 1e-1]

# Fit the model for each radius
estimators = []
for rho in rhos:
    print(f'Fitting model for rho={rho}')
    estimator = LinearRegression(rho=rho, fit_intercept=True)
    estimator.fit(X_train, y_train)
    estimators.append(estimator)

# %%
# Evaluating the models
# ~~~~~~~~~~~~~~~~~~~~~

# Compute the training and test errors
train_errors = []
test_errors = []
for estimator in estimators:
    train_errors.append(mean_squared_error(y_train, estimator.predict(X_train)))
    test_errors.append(mean_squared_error(y_test, estimator.predict(X_test)))

# Print the results
for i, rho in enumerate(rhos):
    print(f'rho={rho}: training error={train_errors[i]:.2e}, test error={test_errors[i]:.2e}')

# %%
# Plotting the results
# ~~~~~~~~~~~~~~~~~~~~~

# Create a figure
plt.figure()

# Plot the training data
plt.scatter(X_train, y_train, color='blue', label='Training data')

# Plot the test data
plt.scatter(X_test, y_test, color='red', label='Test data')

# Plot the estimated models
X_plot = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
for i, estimator in enumerate(estimators):
    y_plot = estimator.predict(X_plot)
    plt.plot(X_plot, y_plot, label=f'rho={rhos[i]}')

# Add labels and legend
plt.xlabel('X')
plt.ylabel('y')
plt.legend()

# Show the plot
plt.show()






