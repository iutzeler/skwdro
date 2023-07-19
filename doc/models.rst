####################
Models
####################

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

\begin{equation*}
     \underset{theta \in \Theta}{inf} \mathbb{E}^{\mathbb{P}}(- \langle \theta, \xi \rangle) + \eta \mathbb{P}\text{-CVaR}_{\alpha}(- \langle \theta, \xi \rangle)
\end{equation*}

where $\eta \geq 0$ and $\alpha \in (0,1]$ are the risk aversion parameters, and $\mathbb{P}\text{-CVaR}_{\alpha}$ is the Conditional Value at Risk relative to the unknown probability distribution $\mathbb{P}$ and confidence level $\alpha$. CVaR is a measure of risk used in financial mathematics. One of the most common definitions for a random variable $X$ governed by a continuous probability distribution is as follows:

\begin{equation*}
    \text{CVaR}_{\alpha}(X) = \mathbb{E}[X | \ X \geq \text{VaR}_{\alpha}(X)]
\end{equation*}

where $\text{VaR}_{\alpha}$ is the Value at Risk of order $\alpha$, which corresponds exactly to the quantile of order $\alpha$. The CVaR is therefore the average expected value for the portfolio following the returns on investments, given that this value is greater than the Value at Risk.



.. Weber problem
.. -------------

.. .. currentmodule:: skwdro.operations_research

.. .. autosummary::
..    :toctree: generated/
..    :template: class.rst

..    Weber



