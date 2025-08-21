#############
What is WDRO?
#############

Wasserstein Distributionally Robust Optimization (WDRO) is a mathematical framework that can provide robustness to data shifts, in machine learning models as well as many settings.
Bellow is a contextualized explanation on its usecases, alternatives, foundations, and core advantages.


Machine Learning models
=======================


Let us denote the cost (or "loss") function :math:`L_\theta(\xi)` associated with a prediction, parametrized by :math:`\theta` for some uncertain data input :math:`\xi\in\Xi`.
For instance, in linear regression, we have :math:`\xi=(x,y)\in\mathbb{R}^d\times\mathbb{R}` with  :math:`x` the data and  :math:`y` the target. Then, one may pick as loss function the "Squared Error", leading in average to the popular MSE estimator.

It is formally written as :math:`L_\theta(\xi) = \frac{1}{2}(y- \langle \theta , x \rangle)^2`.
In general for regression, this will be framed as :math:`L_\theta(\xi):=\ell(y-\langle\theta, x\rangle)` for some scalar function :math:`\ell: \mathbb{R}\to\mathbb{R}` such that *wolog* :math:`\ell(0)=0`.
On the other hand, for classification tasks, we have :math:`\xi=(x,y)\in\mathbb{R}^d\times\{0, 1\}` with  :math:`x` the data and  :math:`y` the label. One may use the soft-margin loss function for logistic regression :math:`L_\theta(\xi):=\log\left(1+e^{-y\langle\theta, x\rangle}\right)`.
In general for linear classification, as can be seen in the logistic regression case, the losses will instead be framed as :math:`L_\theta(\xi):=\ell(-y\langle\theta, x\rangle)` for some scalar function :math:`\ell: \mathbb{R}\to\mathbb{R}` such that *wolog* :math:`\ell(+\infty)=0`.

You can learn more about how you can use the library to build a ``PyTorch`` optimization model by reading through the user guide.

In machine learning, it is usual to train our model (or fit, i.e. optimize on :math:`\theta`) using a finite amount of data samples :math:`(\xi_i)_{i=1}^N`, by minimizing the **Empirical Risk** over this available dataset, which leads to the problem: 

.. math::
    \min_{\theta} \frac{1}{N} \sum_{i=1}^N  L_\theta(\xi_i)
    :label: ERM

Formally, one may view this dataset as an empirical distribution, that may be sampled at will through a query function (e.g. for ``PyTorch``, which is favored by ``SkWDRO``'s internals and interfaces, see `their tutorial <https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html>`__ as well as the `implementation <https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/torch/utils/data/dataloader.py#L481C9-L481C17>`__ to see how this query mechanism is used in practice).
Equation :eq:`ERM` is usually called Empirical Risk Minimization (ERM) in the literature.


Robustness
==========

Robust optimization aims to provide decision-making with some level of resilience against uncertainty. Traditional approaches include:

- defining an uncertainty set where the worst-case scenario is considered in a deterministic way on the whole set of possible outcomes :math:`\Xi`, or subsets thereof,
- or assuming that the empirical distribution represents closely enough the true distribution of the uncertain variable (see :eq:`ERM`), hence trusting that resampling :math:`\hat{\mathbb{P}}^N` multiple times will cover enough uncertainty.

However, these methods can be limiting due to difficulties in setting appropriate uncertainty sets, intractability of the associated optimisation problems, and issues with the representativeness of the available samples.

Distributionally Robust Optimization (DRO) introduces a solution by considering the worst expectation of the cost function when the probability distribution belongs to a neighborhood around the empirical distribution. This approach allows for a more flexible set of distributions, parameterized by a radius (denoted :math:`\rho`) from the empirical measure.

The problem ends up being formulated as follows in the general case:

.. math::
   \min_\theta \sup_{\mathbb{Q}\in \mathcal{U}_\rho((\xi_i)_{i=1}^N)} \mathbb{E}_{\zeta\sim\mathbb{Q}}[L_\theta(\zeta)]
   :label: DRO

This formulation leaves to be chosen the size of the neighborhood :math:`\rho` and the notion of "proximity" between the samples and the robust distribution :math:`\mathbb{Q}`.


Distributional robustness: divergences
--------------------------------------

In order to define the ambiguity sets, various tools have been proposed [#BtHWMR11]_ in the literature alongside WDRO, including but not limited to:

* the :math:`KL`-divergence,
* the :math:`\chi^2` distance,
* the total variation distance,
* the **Wasserstein optimal transport cost**, and variations thereof.

The most popular class of techniques up to recently has been the use of :math:`\phi`-divergences, i.e. distances of the form:

.. math::
   \mathcal{D}_\phi(\mathbb{P} \| \mathbb{Q}) := \mathbb{E}_\mathbb{Q}\left[\phi\left(\frac{\mathrm{d}\mathbb{P}}{\mathrm{d}\mathbb{Q}}\right)\right].
   :label: PhiDiv

They are appealing for their apparent simplicity, and because a lot of commonly used divergences in other fields, such as statistics, admit this representation. E.g., for the :math:`KL`-divergence :math:`\phi(s)=\begin{cases}s\log(s) - s + 1 & \text{if } s\ge 0\\ +\infty & \text{otherwise}\end{cases}`, for total variation :math:`\phi(s)=\begin{cases}(s-1)^2 & \text{if } s\ge 0\\ +\infty & \text{otherwise}\end{cases}`.
See [#KSW24]_ §2.2 for a detailed explanation of the ambiguity sets related to these divergences.

Fortunately, there so happens to exist a recent line of work aiming to engineer those ambiguity sets in a real-world setting, head of which is a python library called `python-dro <https://python-dro.org/>`__.
We encourage curious users to take a look at `their implementation of some problems <https://python-dro.org/tutorials/linear_fdro.html>`__ using this framework.

.. note::
   .. warning:: About :math:`\phi`-divergences.

   These divergences have a common noteworthy property: the "true" (and inaccessible) distribution that the observed samples originate from, denoted :math:`\mathbb{P}`, lies out of the neighborhood around the empirical distribution :math:`\hat{\mathbb{P}}^N` with probability 1 as long as it is absolutely continuous with respect to the Lebesgue measure.
   Depending on the nature of the problem at hand, this may be a real issue, which can be adressed by picking a different notion of neighborhood, as explained bellow.


Distributional robustness: optimal transport
--------------------------------------------

A recent push has been made towards wider neighborhoods notions for DRO than what we explored in our last chapter, lead by the WDRO formulation.
Its ambiguity set relies on the **Wasserstein optimal transport cost**.
Dating back to Monge's *"Mémoire sur la théorie des déblais et des remblais" (1781)*, the optimal transport mathematical framework sparked interest for incorporating geometrical properties into the notion of distance/divergence between distributions.

The transport cost is defined through a so-called "ground cost", denoted :math:`c`, defining how much needs to be paid to transport an input :math:`\xi` to any other valid input :math:`\zeta\in\Xi` (usually such that :math:`c(\xi, \xi)=0`, and :math:`c(\xi, \zeta)>0`, :math:`\forall\zeta\neq\xi`, e.g. a distance function).
Thus, given this geometrical insight on the studied space of uncertainty, the average transport cost :math:`c(\xi, \zeta)` from reference samples :math:`\xi\sim\hat{\mathbb{P}}^N` to their robust counterparts :math:`\zeta\sim\mathbb{Q}` appears as a natural way to prescribe the transport cost from :math:`\hat{\mathbb{P}}^N` to :math:`\mathbb{Q}`.

.. admonition:: Def.: Transport plan.

   A *transport plan* between two measures :math:`\mathbb{X}` and :math:`\mathbb{Y}` is a measure on the product of their supports :math:`\Xi^2` that has as first and second marginals respectively :math:`\mathbb{X}` and :math:`\mathbb{Y}`.

   We denote by :math:`\Pi(\mathbb{X}, \mathbb{Y})` the set of all these possible transport plans:

   .. math::
      \Pi(\mathbb{X}, \mathbb{Y}):=\left\lbrace \pi\in\mathcal{M}(\Xi^2) \mid \int_{\zeta\in\Xi} \mathrm{d}\pi(A, \zeta) = \mathrm{d}\mathbb{X}(A)\text{ and }\int_{\xi\in\Xi} \mathrm{d}\pi(\xi, B) = \mathrm{d}\mathbb{Y}(B)\right\rbrace.

   Later, we will denote the marginals respectively by :math:`[\pi]_1` and :math:`[\pi]_2`.


Now that we have this notion of transport between distribution, we can recall the definition of the *Wasserstein distance* at the core of modern distributionally robust optimisation:

.. admonition:: Def.: Wasserstein transport cost.

   Let :math:`c` a *ground cost*, and two distributions on :math:`\Xi`, :math:`\mathbb{X}` and :math:`\mathbb{Y}`.

   .. math::
      W_c(\mathbb{X}, \mathbb{Y}) := \inf_{\pi\in\Pi(\mathbb{X}, \mathbb{Y})}\mathbb{E}_{(\xi, \zeta)\sim\pi}\left[c(\xi, \zeta)\right]
      :label: Wasserstein_ot

.. note:: An interesting property of the transport cost with respect to its ground cost is that if :math:`c` is a distance risen to some power :math:`1\le p\le\infty`, then :math:`\sqrt[p]{W_c}` becomes a distance on the space of measures :math:`\mathcal{M}(\Xi)`.
   This yields the acclaimed **Wasserstein distance**:

   .. admonition:: Def.: Wasserstein distances of order :math:`p`, or ":math:`p`-Wasserstein-distances".

      If :math:`\Xi` is endowed with a distance function :math:`d_\Xi: \Xi^2\to\mathbb{R}^+`, then we call :math:`p`\ *-Wasserstein-distances* the transport cost associated with the :math:`d_\Xi^p` ground cost:

      .. math::

         \DeclareMathOperator*{\esssup}{ess\,sup}

         \begin{cases}
             W_p(\mathbb{X}, \mathbb{Y}) := \inf_{\pi\in\Pi(\mathbb{X}, \mathbb{Y})}\sqrt[p]{\mathbb{E}_{(\xi, \zeta)\sim\pi}\left[d_\Xi(\xi, \zeta)^p\right]} & \text{if }p\in\mathbb{N}^*\\
             W_\infty(\mathbb{X}, \mathbb{Y}) := \inf_{\pi\in\Pi(\mathbb{X}, \mathbb{Y})}\esssup_{(\xi, \zeta)\sim\pi} d(\xi, \zeta) & \text{otherwise.}
         \end{cases}


WDRO in a nutshell
==================

Considering what we noted about :math:`\phi`-divergences, in that they are limited to rebalancing histograms thus lacking representation power, we may turn to the Wasserstein type of ambiguity sets.
This leaves as main problem the following:

.. math::
   \min_\theta \sup_{W_c(\hat{\mathbb{P}}^N, \mathbb{Q})\le\rho} \mathbb{E}_{\zeta\sim\mathbb{Q}}[L_\theta(\zeta)].
   :label: WDRO

Several parts of the literature focus on providing a dual formula for :eq:`WDRO`, which holds under mild assumptions:

.. math::
   \min_{\theta, \lambda\ge 0} \lambda\rho + \mathbb{E}_{\xi\sim\hat{\mathbb{P}}^N}\left[\sup_{\zeta\in\Xi}\left\lbrace L_\theta(\zeta)-\lambda c(\xi, \zeta)\right\rbrace\right]
   :label: WDRO_dual

Its main advantage is the fact that it switched from a variational infinite-dimentional problem of finding a *worst measure* to a (usually) finite-dimensional problem of finding a *worst input*.

Now one must take note that the inner supremum of :eq:`WDRO_dual` is still to be taken with utmost care: if :math:`\Xi` is not bounded, and :math:`f_\theta(\cdot)-\lambda c(\xi, \cdot)` grows large by any means, then the problem is ill-posed.
Note is also to be taken that even if the supremum is attained, it could be computationally intractable depending on the nature of :math:`\hat{\mathbb{P}}^N`, :math:`\Xi`, :math:`c`, and :math:`f_\theta`.
Hence, even though this problem is easier than its primal counterpart, it needs more structure to be amenable to high-dimensional problems.
See the `next tutorial <why_skwdro.html>`__ for more insights on this.


Some instances of reformulated WDRO problems
--------------------------------------------

In some cases, the WDRO problem may be reformulated into a convex finite-dimensional program, that one can solve with disciplined programming (e.g. the `cvxpy <https://github.com/cvxpy/cvxpy>`__ python library).
Many of those can be found in the seminal work of [#EK17]_.

+----------------------------+----------------------------------------------------------------------------+---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------+
| **Model**                  | **loss structure**                                                         | **Input set** :math:`\Xi` | **WD-Robust formulation**                                                                                                                                                                                                                                                                            | **Source**             |
+============================+============================================================================+===========================+======================================================================================================================================================================================================================================================================================================+========================+
| Logistic regression        | :math:`f_\theta(\xi):=\log(1+e^{-y\left\langle\theta\mid x\right\rangle})` | :math:`\mathbb{R}^d`      | :math:`\min_\theta \rho \text{Lip}(f)\|\theta\|_* + \mathbb{E}_{\xi\sim\hat{\mathbb{P}}^N}\left[f_\theta(\xi)\right]`                                                                                                                                                                                | [#SaEK15]_, [#SaKE19]_ |
+----------------------------+----------------------------------------------------------------------------+---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------+
| SVM                        | :math:`f_\theta` lipschitz, norm constraint on :math:`\theta`              | :math:`\mathbb{R}^d`      | Idem                                                                                                                                                                                                                                                                                                 | [#SaKE19]_             |
+----------------------------+----------------------------------------------------------------------------+---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------+
| Convex functions           | :math:`f_\theta` input-convex with any parametrization :math:`\theta`      | :math:`\mathbb{R}^d`      | :math:`\min_\theta \rho\kappa_\theta + \mathbb{E}_{\xi\sim\hat{\mathbb{P}}^N}\left[f_\theta(\xi)\right]`, (see more about [kappa]_)                                                                                                                                                                  | [#EK17]_               |
+----------------------------+----------------------------------------------------------------------------+---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------+
| Piecewise-affine (convex)  | :math:`f_{A, b}(\xi):=\max_i(A\xi+b)_i`                                    | :math:`\{\xi|C\xi\le d\}` | :math:`\begin{align}\min_\theta\inf_{\lambda, s_i, \Gamma_i\ge 0} & \lambda\rho + \sum_{i=1}^Ns_i\\ \text{s.t.} & A\xi_i+b+\Gamma_i(d-C\xi_i)\le s_i\mathbb{1}\\ & \|C^T\Gamma_i-A\|_{*, \infty}\le\lambda\end{align}`                                                                               | [#EK17]_               |
+----------------------------+----------------------------------------------------------------------------+---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------+
| Piecewise-affine (concave) | :math:`f_{A, b}(\xi):=\min_i(A\xi+b)_i`                                    | :math:`\{\xi|C\xi\le d\}` | :math:`\begin{align}\min_\theta\inf_{\lambda, s_i, g_i\ge 0, t_i\ge 0} & \lambda\rho + \sum_{i=1}^Ns_i\\ \text{s.t.} &\langle t_i | A\xi_i+b\rangle+\langle g_i | d-C\xi_i\rangle\le s_i\\ &\|(C^T\Gamma_i-A^Tt_i)_{i, :}\|_{*, \infty}\le\lambda\\ &\langle t_i | \mathbb{1}\rangle = 1\end{align}` | [#EK17]_               |
+----------------------------+----------------------------------------------------------------------------+---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------+

.. [kappa] Here the authors define a notion of growth rate :math:`\kappa_\theta` reminiscent of [#GCK24]_, defined here as :math:`\sup_{\theta | f_\theta^*(\upsilon)<\infty}\|\upsilon\|_*`

Note that the convex case is the most general, but it requires a good knowledge of the loss function through the :math:`\kappa` constant.


Conclusion
==========

We saw that a lot of models are already well-studied through the lense of WDRO when it comes to robustness, but we lack techniques to robustify **efficiently** losses on which we lack knowledge (e.g. big neural nets).
While it remains very relevant to some of the problems mentionned above, this questions its applicability in real world scenarii: in this library we propose to turn to a regularization technique to make the bigger and tougher models amenable to robustness.

See the next tutorial on `Sinkhorn-WDRO <why_skwdro.html>`_ to understand how we make it happen.

==========
References
==========
.. [#SaKE19] Shafieezadeh-Abadeh, Kuhn and Esfahani: **Regularization via Mass Transportation**, *JMLR*, 2019
.. [#SaEK15] Shafieezadeh-Abadeh, Esfahani and Kuhn: **Distributionally Robust Logistic Regression**, *NIPS*, 2015
.. [#BtHWMR11] Ben-Tal, Hertog, DeWaegenaere, Melenberg and Rennen: **Robust solutions of optimization problems affected by uncertain probabilities**, *Management Sciences*, 2011
.. [#EK17] Esfahani and Kuhn: **Data-Driven Distributionally Robust Optimization Using the Wasserstein Metric: Performance Guarentees and Tractable Reformulations**, *Mathematical Programming*, 2017
.. [#KSW24] Kuhn, Shafiee and Wiesemann: **Distributionally Robust Optimization**, *Acta Numerica*, 2024
.. [#GCK24] Gao, Chen and Kleywegt: **Wasserstein Distributionally Robust Optimization and Variation Regularization**, *Operations Research*, 2024
