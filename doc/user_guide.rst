.. title:: User guide : contents

.. _user_guide:

==================================================
User guide
==================================================

Using SKWDRO estimators
---------

The estimators can be used as thos in sklearn.

Regressor
~~~~~~~~~

Similarly, regressors are scikit-learn estimators which implement a ``predict``
method. The use case is the following:

* at ``fit``, some parameters can be learned from ``X`` and ``y``;
* at ``predict``, predictions will be computed using ``X`` using the parameters
  learned during ``fit``.

In addition, scikit-learn provides a mixin_, i.e.
:class:`sklearn.base.RegressorMixin`, which implements the ``score`` method
which computes the :math:`R^2` score of the predictions.

One can import the mixin as::

    >>> from sklearn.base import RegressorMixin

Therefore, we create a regressor, :class:`MyOwnRegressor` which inherits from
both :class:`sklearn.base.BaseEstimator` and
:class:`sklearn.base.RegressorMixin`. The method ``fit`` gets ``X`` and ``y``
as input and should return ``self``. It should implement the ``predict``
function which should output the predictions of your regressor::

    >>> import numpy as np
    >>> class MyOwnRegressor(BaseEstimator, RegressorMixin):
    ...     def fit(self, X, y):
    ...         return self
    ...     def predict(self, X):
    ...         return np.mean(X, axis=1)

We illustrate that this regressor is working within a scikit-learn pipeline::

    >>> from sklearn.datasets import load_diabetes
    >>> X, y = load_diabetes(return_X_y=True)
    >>> pipe = make_pipeline(MyOwnTransformer(), MyOwnRegressor())
    >>> pipe.fit(X, y)  # doctest: +ELLIPSIS
    Pipeline(...)
    >>> pipe.predict(X)  # doctest: +ELLIPSIS
    array([...])

Since we inherit from the :class:`sklearn.base.RegressorMixin`, we can call
the ``score`` method which will return the :math:`R^2` score::

    >>> pipe.score(X, y)  # doctest: +ELLIPSIS
    -3.9...

Classifier
~~~~~~~~~~

Similarly to regressors, classifiers implement ``predict``. In addition, they
output the probabilities of the prediction using the ``predict_proba`` method:

* at ``fit``, some parameters can be learned from ``X`` and ``y``;
* at ``predict``, predictions will be computed using ``X`` using the parameters
  learned during ``fit``. The output corresponds to the predicted class for each sample;
* ``predict_proba`` will give a 2D matrix where each column corresponds to the
  class and each entry will be the probability of the associated class.

In addition, scikit-learn provides a mixin, i.e.
:class:`sklearn.base.ClassifierMixin`, which implements the ``score`` method
which computes the accuracy score of the predictions.

One can import this mixin as::

    >>> from sklearn.base import ClassifierMixin

Therefore, we create a classifier, :class:`MyOwnClassifier` which inherits
from both :class:`slearn.base.BaseEstimator` and
:class:`sklearn.base.ClassifierMixin`. The method ``fit`` gets ``X`` and ``y``
as input and should return ``self``. It should implement the ``predict``
function which should output the class inferred by the classifier.
``predict_proba`` will output some probabilities instead::

    >>> class MyOwnClassifier(BaseEstimator, ClassifierMixin):
    ...     def fit(self, X, y):
    ...         self.classes_ = np.unique(y)
    ...         return self
    ...     def predict(self, X):
    ...         return np.random.randint(0, self.classes_.size,
    ...                                  size=X.shape[0])
    ...     def predict_proba(self, X):
    ...         pred = np.random.rand(X.shape[0], self.classes_.size)
    ...         return pred / np.sum(pred, axis=1)[:, np.newaxis]

We illustrate that this regressor is working within a scikit-learn pipeline::

    >>> X, y = load_iris(return_X_y=True)
    >>> pipe = make_pipeline(MyOwnTransformer(), MyOwnClassifier())
    >>> pipe.fit(X, y)  # doctest: +ELLIPSIS
    Pipeline(...)

Then, you can call ``predict`` and ``predict_proba``::

    >>> pipe.predict(X)  # doctest: +ELLIPSIS
    array([...])
    >>> pipe.predict_proba(X)  # doctest: +ELLIPSIS
    array([...])

Since our classifier inherits from :class:`sklearn.base.ClassifierMixin`, we
can compute the accuracy by calling the ``score`` method::

    >>> pipe.score(X, y)  # doctest: +ELLIPSIS
    0...



Setting up parameters
-----------


