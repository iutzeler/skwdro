.. title:: User guide : contents

.. _user_guide:

==================================================
User guide
==================================================

Using WDRO estimators
-----------------------

The estimators can be used as those in sklearn.

WDRO Regressors
~~~~~~~~~~~~~~~

Similarly, regressors are scikit-learn estimators which implement a ``predict``
method. The use case is the following:

* at ``fit``, some parameters can be learned from ``X`` and ``y``;
* at ``predict``, predictions will be computed using ``X`` using the parameters
  learned during ``fit``.



Since we inherit from the :class:`sklearn.base.RegressorMixin`, we can call
the ``score`` method which will return the :math:`R^2` score::

    >>> pipe.score(X, y)
    -3.9...

WDRO Classifier
~~~~~~~~~~~~~~~

Similarly to regressors, classifiers implement ``predict``. In addition, they
output the probabilities of the prediction using the ``predict_proba`` method:

* at ``fit``, some parameters can be learned from ``X`` and ``y``;
* at ``predict``, predictions will be computed using ``X`` using the parameters
  learned during ``fit``. The output corresponds to the predicted class for each sample;
* ``predict_proba`` will give a 2D matrix where each column corresponds to the
  class and each entry will be the probability of the associated class.


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
---------------------


