.. title:: User guide : contents

.. _user_guide:

==================================================
User guide
==================================================

The goal of this page is to provide an introduction to the main features of the package: the ``scikit-learn`` and the ``PyTorch`` interfaces. We will demonstrate the main functionalities on a simple Linear Regression example.


Linear Regression
~~~~~~~~~~~~~~~~~

Given some feature vectors :math:`x_1,\dots,x_n \in \mathbb{R}^d` and the corresponding target values :math:`y_1,\dots,y_n \in \mathbb{R}`, the goal is to learn a linear model :math:`w \in \mathbb{R}^d,\ b \in \mathbb{R}` that predicts the target value from the feature vector, i.e., :math:`y_i \approx w^T x_i + b` for all :math:`i=1,\dots,n`.

The most common approach to learn the parameters :math:`w` and :math:`b` is to minimize the empirical risk with the squared loss function:

.. math::

    \min_{w, b} \frac{1}{n} \sum_{i=1}^n (y_i - w^T x_i - b)^2.




Solving the regression problem with ``scikit-learn``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The linear regression problem can now be solved with the ``LinearRegression`` estimator from ``scikit-learn``.
We assume that we are given ``X_train`` of shape ``(n_train, n_features)`` and ``y_train`` of shape ``(n_train,)`` as training data and ``X_test`` of shape ``(n_test, n_features)`` as test data.

::

    from sklearn.linear_model import LinearRegression

    # Fit the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict the target values
    y_pred = model.predict(X_test)


Solving the robust regression problema with ``skwdro``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Robust estimators from ``skwdro`` can be used as drop-in replacements for ``scikit-learn`` estimators (they actually inherit from ``scikit-learn`` estimators and classifier classes.)
``skwdro`` provides robust estimators for standard problems such as linear regression or logistic regression.
``LinearRegression`` from ``skwdro.linear_model`` is a robust version of ``LinearRegression`` from ``scikit-learn`` and be used in the same way. The only difference is that now an uncertainty radius ``rho`` is required.

::

    from skwdro.linear_model import LinearRegression

    # Uncertainty radius
    rho = 0.1

    # Fit the model
    robust_model = LinearRegression(rho=rho)
    robust_model.fit(X_train, y_train)

    # Predict the target values
    y_pred = robust_model.predict(X_test)

As a consequence, robust estimators can be tried and used without much change to existing pipelines!

By default, the ``LinearRegression`` estimator from ``skwdro`` uses will solve the robust optimization problem with entropic regularization and by calling a stochastic first-order solver in ``PyTorch``. A dedicated solver can be used by setting the ``solver`` parameter in the constructor to ``'dedicated'``.

::

    robust_model = LinearRegression(rho=rho, solver='dedicated')

Solving the regression problem with the ``PyTorch`` interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The next setion now describe the ``PyTorch`` interface of ``skwdro``: it allows more flexibility, custom models and optimizers. 

Assume now that the data is given as a dataloader ``train_loader``.

::

    import torch
    import torch.nn as nn
    import torch.optim as optim

    from skwdro.torch import robustify

    # Uncertainty radius
    rho = 0.1

    # Define the model
    model = nn.Linear(n_features, 1)

    # Define the loss function
    loss_fn = nn.MSELoss()

    # Define a sample batch for initialization
    sample_batch_x, sample_batch_y = next(iter(train_loader))
    
    # Robust loss
    robust_loss = robustify(loss_fn, model, rho, sample_batch_x, sample_batch_y)

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(100):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss = robust_loss(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

This is the simplest use of the ``PyTorch`` interface: just wrap the usual loss and model with the ``robustify`` function and use the resulting loss function in the training loop.

To make the optimization of the robust model more efficient, we also provide an learning-rate free optimizer tailored to this problem. 

::

    # Adaptive optimizer
    optimizer = robust_loss.optimizer


