#######################
scikit-learn interface
#######################

The following models are implemented in skwdro

.. currentmodule:: skwdro


Linear Models
=============


Logistic Regression
-------------------

.. currentmodule:: skwdro.linear_models

.. autosummary::
   :toctree: generated/
   :template: class.rst


   LogisticRegression

Linear Regression
-----------------

.. currentmodule:: skwdro.linear_models

.. autosummary::
   :toctree: generated/
   :template: class.rst

   LinearRegression


Operations Research
===================


NewsVendor
----------

.. currentmodule:: skwdro.operations_research

.. autosummary::
   :toctree: generated/
   :template: class.rst

   NewsVendor


Portfolio Selection
-------------------

.. currentmodule:: skwdro.operations_research

.. autosummary::
   :toctree: generated/
   :template: class.rst

   Portfolio

Represents the Mean-Risk Portfolio model.

The stochastic optimisation problem associated with the optimal decision to be taken is

.. math::

   \underset{\theta \in \Theta}{\inf} \mathbb{E}^{\mathbb{P}}(- \langle \theta, \xi \rangle) + \eta \mathbb{P}\text{-CVaR}_{\alpha}(- \langle \theta, \xi \rangle)



where :math:`\eta \geq 0`  and  :math:`\alpha \in (0,1]` are the risk aversion parameters, and :math:`\mathbb{P} \mathrm{-CVaR}_{\alpha}` is the Conditional Value at Risk relative to the unknown probability distribution :math:`\mathbb{P}` and confidence level :math:`\alpha`. CVaR is a measure of risk used in financial mathematics. One of the most common definitions for a random variable X governed by a continuous probability distribution is as follows:

.. math::
   \text{CVaR}_{\alpha}(X) = \mathbb{E}[X | \ X \geq \text{VaR}_{\alpha}(X)]


where :math:`\mathrm{VaR}_{\alpha}` is the Value at Risk of order :math:`\alpha`, which corresponds exactly to the quantile of order :math:`\alpha`. The CVaR is therefore the average expected value for the portfolio following the returns on investments, given that this value is greater than the Value at Risk.


.. Weber problem
.. -------------

.. .. currentmodule:: skwdro.operations_research

.. .. autosummary::
..    :toctree: generated/
..    :template: class.rst

..    Weber



