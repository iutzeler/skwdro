####################
What is WDRO?
####################

Wasserstein Distributionally Robust Optimization (WDRO) is a mathematical program that can provide robustness to data shifts in machine learning models.


Machine Learning models
=======================


Let us denote the cost :math:`f_\theta(\xi)`  of a prediction parametrized by :math:`\theta` for some uncertain variable :math:`\xi`.
For instance, in linear regression, we have :math:`\xi=(x,y)\in\mathbb{R}^d\times\mathbb{R}` with  :math:`x` the data and  :math:`y` the label. Then, :math:`f_\theta(\xi) = \frac{1}{2} ( \langle \theta , x \rangle - y )^2`.

In machine learning, it is usual to train our model (or fit, ie. optimize on :math:`\theta`) using data samples :math:`(\xi_i)_{i=1}^n`  of the uncertain parameter by minimizing the Empirical Risk, which leads to the problem: 

.. math::    
    \min_{\theta} \frac{1}{n} \sum_{i=1}^n  f_\theta(\xi_i)
    :label: ERM


Equation :eq:`ERM` is usually called Empirical Risk Minimization (ERM) is the literature.


Robustness
==========


.. Robust optimization has a long history in the theory and practice of decision-making, when some robustness is sought against uncertainty. A standard approach is to minimize the worst-case cost when $\sample$ lives in some uncertainty set $\uncertainty$, known or chosen by the user, which leads to the problem
.. \begin{align}
..   \label{eq:wc}
..   \min_{\param \in \params} ~ \sup_{\sample \in \uncertainty} \obj_\param(\sample)\,.
.. \end{align}

.. Both approaches suffer from intrinsic limitations. The robust approach \eqref{eq:wc} relies on an uncertainty set $\uncertainty$ that may be difficult to design and can lead to pessimistic decisions, corresponding to an unlikely uncertainty variable.
.. On the other hand, the sample average approximation problem \eqref{eq:sto} is built over the assumption that the empirical distribution $\empirical$ is close to the true distribution of the uncertain variable met in the target application. This may not be verified in practice, indeed: i) there may be too few samples to approximate correctly their underlying distribution, or ii) the uncertain variable's distribution may change between the optimization and application of the model (see for instance the datasets of \cite{koh2021wilds}).

.. Nevertheless, the empirical distribution still provides partial information about the encountered distribution of $\sample$, so it seems reasonable to assume that the two distributions are close. \Ac{DRO} thus consists in minimizing the \emph{worst expectation} of the cost when the \emph{distribution} lives in a {neighborhood} $\nhd(\empirical)$ of $\empirical$. The resulting problem is then 
.. \begin{align}
..     \min_{\param \in \params} ~ \sup_{\probalt \in \nhd(\empirical)}\ex_{\sample \sim \probalt}[\obj_\param(\sample)]\,
..     %\tag{\ac{DRO}}
..     \label{eq:intro_dro}
.. \end{align}
.. where the inner $\sup$ is taken over probability measures on $\samples$ in the set $ \nhd(\empirical)$. 
.. This approach can be traced back at least to \cite{scarf1958min}, we refer to the review \cite{rahimian2019distributionally} for a general formulation and connections with the literature. 


.. The maximization over probability measures is an infinite dimensional problem and thus a compromise has to be found between the modelling capacity and the computational tractability of the objective. This question will be pervasive in the whole project. 
.. First, we can notice that if $ \nhd(\empirical)$ is reduced to the singleton $\{\empirical\}$, the problem is equivalent to \eqref{eq:sto}. \cite{delage2010distributionally} proposed to use a finitely-parametrized ambiguity set based on moments (mean and covariance) and investigated the tractability of the associated problem for a class of cost functions. 
.. Non-parametric ambiguity sets relying on $\phi$-divergences (such as $\chi^2$ or Kullback-Leibler divergences) were also investigated, for instance in \eg \cite{ben2013robust,namkoong2016stochastic} where the numerical tractability is again put forward. However, the ambiguity sets constructed from $\phi$-divergences only contain distributions with the same support as the empirical distribution.


WDRO
====


.. These sets are of the form $ \nhd(\empirical) =  \{\probalt \in \probs: \wass{\empirical, \probalt} \leq \radius\}$ for some $\radius>0$, where $\probs$ is the set of probability distributions and, for a cost function $\cost : \samples\times\samples \to \R_+$,  the Wasserstein distance between $\empirical$ and $\probalt$ is defined as
.. \begin{align}\label{eq:wass}
..     \wass[\cost]{\empirical, \probalt} = \inf \left\{\ex_{(\sample, \altsample) \sim  \coupling}  \left[ \cost(\sample, \altsample) \right] : \coupling \in \couplings, \coupling_1 = \empirical, \coupling_2 = \probalt\right\}\,,
.. \end{align}
.. with $\coupling_1$ (resp. $\coupling_2$) the first (resp. second) marginal of the transport plan $\coupling$. With such an ambiguity set, \eqref{eq:intro_dro} becomes a \ac{WDRO} problem. 

.. As Wasserstein-based ambiguity sets are appealing in terms of expressiveness and mathematical foundation, \ac{WDRO} has been very popular in the recent years in the robust optimization and machine learning communities \cite{blanchet2019robust,esfahani2018data,gao2016distributionally,kuhn2019wasserstein,ho2022adversarial}. 
.. Another argument supporting this approach in data science is that it provides stronger generalization guarantees, see \eg \cite{esfahani2018data,lee2018minimax,an2021generalization}.

