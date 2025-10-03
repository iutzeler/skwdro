.. title:: User guide : contents

.. _user_guide:

==========
User guide
==========

The goal of this page is to point to the parts of the documentation that showcase the main features of the package: the ``scikit-learn`` and the ``PyTorch`` interfaces. We will demonstrate the main functionalities on a simple Linear Regression example.


Linear Regression
~~~~~~~~~~~~~~~~~

Given some feature vectors :math:`x_1,\dots,x_n \in \mathbb{R}^d` and the corresponding target values :math:`y_1,\dots,y_n \in \mathbb{R}`, the goal is to learn a linear model :math:`w \in \mathbb{R}^d,\ b \in \mathbb{R}` that predicts the target value from the feature vector, i.e., :math:`y_i \approx w^T x_i + b` for all :math:`i=1,\dots,n`.

The most common approach to learn the parameters :math:`w` and :math:`b` is to minimize the empirical risk with the squared loss function:

.. math::

    \min_{w, b} \frac{1}{n} \sum_{i=1}^n (y_i - w^T x_i - b)^2.




Solving the regression problem with ``scikit-learn``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The linear regression problem can now be solved with the :py:class:`sklearn.linear_model.LinearRegression` estimator from ``scikit-learn``.
We assume that we are given ``X_train`` of shape ``(n_train, n_features)`` and ``y_train`` of shape ``(n_train,)`` as training data and ``X_test`` of shape ``(n_test, n_features)`` as test data.

.. code-block:: python
   :linenos:
   :caption: Scikit's ``LinearRegression`` interface

   from sklearn.linear_model import LinearRegression

   # Fit the model
   model = LinearRegression()
   model.fit(X_train, y_train)

   # Predict the target values
   y_pred = model.predict(X_test)


Solving the robust regression problem with ``skwdro``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Robust estimators from ``skwdro`` can be used as drop-in replacements for ``scikit-learn`` estimators (they actually inherit from ``scikit-learn`` estimators and classifier classes.)
``skwdro`` provides robust estimators for standard problems such as linear regression or logistic regression.
:py:class:`skwdro.linear_models.LinearRegression` is a robust version of :py:class:`sklearn.linear_model.LinearRegression` and may be used in the same way. The only difference is that now an uncertainty radius ``rho`` is required.

.. code-block:: python
   :linenos:
   :caption: SkWDRO's ``LinearRegression`` interface
   :emphasize-added: 3,12,13,17,19
   :emphasize-removed: 2,16,18

   >>> import numpy as np
   >>> from sklearn.linear_model import LinearRegression as ERMRegression
   >>> from skwdro.linear_models import LinearRegression as DRORegression
   >>> 
   >>> # Some toy linear problem: e.g. additive noise level shift
   >>> rng = np.random.RandomState(666)
   >>> X_train = rng.randn(10, 1)
   >>> X_test = rng.randn(5, 1) + .5
   >>> y_train = 2. * X_train.flatten() + .01 * rng.randn(10)
   >>> y_test = 2. * X_test.flatten() + .1 * rng.randn(5)
   >>> 
   >>> # Uncertainty radius
   >>> rho = 0.1
   >>> 
   >>> # Fit the model
   >>> erm_model = ERMRegression()
   >>> robust_model = DRORegression(rho=rho)
   >>> erm_model.fit(X_train, y_train)
   >>> robust_model.fit(X_train, y_train)
   >>> 
   >>> # Predict the target values
   >>> y_pred = erm_model.predict(X_test)
   >>> print(np.mean((y_pred - y_test)**2))
   0.009900357816198937
   >>> y_pred = robust_model.predict(X_test)
   >>> print(np.mean((y_pred - y_test)**2))
   0.009643423384431925

As a consequence, robust estimators can be tried and used without much change to existing pipelines!

By default, the ``LinearRegression`` estimator from ``skwdro`` uses will solve the robust optimization problem with entropic regularization and by calling a stochastic first-order solver in ``PyTorch``. A dedicated solver from ``CvxPy`` can be used by setting the ``solver`` parameter in the constructor to ``'dedicated'``.

.. code-block:: python

    robust_model = LinearRegression(rho=rho, solver='dedicated')

Solving the regression problem with the ``PyTorch`` interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The next section now describe the ``PyTorch`` interface of ``skwdro``: it allows more flexibility, custom models and optimizers. 

Assume now that the (training) data is given as a dataloader ``train_loader``.

.. code-block:: python
   :linenos:
   :caption: SkWDRO's ``PyTorch``-type interface
   :emphasize-lines: 17,20,26,29,37,40

   import torch as pt
   import torch.nn as nn
   import torch.optim as optim

   from skwdro.torch import robustify

   # Toy data
   n_features = 3
   X = pt.randn(32, n_features)
   y = X @ pt.rand(n_features, 1) + 1.
   train_loader = pt.utils.data.DataLoader(
       pt.utils.data.TensorDataset(X, y),
       batch_size=4
   )

   # Uncertainty radius
   rho = pt.tensor(.1)

   # Define the model
   model = nn.Linear(n_features, 1)

   # Define the loss function
   loss_fn = nn.MSELoss(reduction='none')

   # Define a sample batch for initialization
   sample_batch_x, sample_batch_y = X[:16, :], y[:16, :]

   # Robust loss
   robust_loss = robustify(loss_fn, model, rho, sample_batch_x, sample_batch_y)

   # Define the optimizer
   optimizer = optim.AdamW(model.parameters(), lr=.1)

   # Training loop
   for epoch in range(100):
       avg_loss = 0.
       robust_loss.get_initial_guess_at_dual(X, y)
       for batch_x, batch_y in train_loader:
           optimizer.zero_grad()
           loss = robust_loss(batch_x, batch_y)
           loss.backward()
           optimizer.step()
           avg_loss += loss.detach().item()
       print(f"=== Loss (epoch \t{epoch}): {avg_loss/len(train_loader)}")

This is the simplest use of the ``PyTorch`` interface: just wrap the usual loss and model with the ``robustify`` function and use the resulting loss function in the training loop.

To make the optimization of the robust model more efficient, we also provide an learning-rate free optimizer tailored to this problem, taken from pieces of the literature: [#CDM23]_ and [#MD24]_. 

.. code-block:: python
   :caption: Fetch the optimizer from the robust loss!

    # Adaptive optimizer
    optimizer = robust_loss.optimizer

Next
----

.. card-carousel:: 2

   .. card:: Scikit part of the library
      :link: sklearn.html

      Tutorial on how to use pre-implemented examples with their scikit-learn interface.

   .. card:: PyTorch part of the library
      :link: pytorch.html

      Tutorial on how to robustify your model easily with the pytorch wrappers.

   .. card:: What is WDRO
      :link: wdro.html

      Gentle introduction to the world of Distributionally Robust Optimization, and motivations for its Wasserstein version.

   .. card:: Sinkhorn-WDRO
      :link: why_skwdro.html

      More about why and how to regularize the WDRO formulation with the Sinkhorn divergence.

   .. card:: API
      :link: api_deepdive/submodules.html

      More details about the exposed API.

References
==========

.. [#CDM23] Cutkosky, Defazio and Mehta: **Mechanic: a Learning Rate Tuner**, *NIPS*, 2023
.. [#MD24] Mishchenko and Defazio: **Prodigy: An Expeditiously Adaptive Parameter-Free Learner**, *ICML*, 2024
