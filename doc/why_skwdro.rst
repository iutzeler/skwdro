===========
Why SkWDRO?
===========

Let us now present the ideas in this library.

.. tip:: Read the previous `tutorial on WDRO <wdro.html>`__ to understand better this part.

Pronouncing its name
====================

The idea of ``SkWDRO`` comes from a modification of WDRO (see the previous `tutorial on the topic <wdro.html>`_).
The prefix ``Sk`` is added as a pun mixing *"Sinkhorn"*, the name of the entropic regularization term we use, and *"scikit"*\ (-learn) which is the python library for some of the out-of-the-box (Sk-)WDRO examples we coded, as a layer of abstraction.
In the dev team, we ended up pronouncing it by spelling the six letters, but we still believe it sounds great phonetically as well with a slight french accent (``s-qu-ou-drô``/``secoue-drô``, pun intended).

What is wrong with WDRO?
========================

Recall the general "tractable" formula for WDRO:

.. math::
    :label: WDRO_dual

    \min_{\theta, \lambda\ge 0} \lambda\rho + \mathbb{E}_{\xi\sim\hat{\mathbb{P}}^N}\left[\sup_{\zeta\in\Xi}\left\lbrace L_\theta(\zeta)-\lambda c(\xi, \zeta)\right\rbrace\right]

While in small dimension :math:`d` and with a concave loss :math:`L_\theta` (or concave `math:`c`-transformed loss) this supremum expression in :eq:`WDRO_dual` might seem tractable at first, it becomes prohibtively complicated in any other setting: the optimality conditions may be non-obvious, spurious maxima are to be expected in the non-concave cases, and algorithms to solve the optimality conditions are not guarenteed to reach a good convergence.
One will also need to leverage some fashion of the implicit function theorem to sort out the dependency between the optimal :math:`\zeta^*` and the parameters :math:`\theta` in order to perform descent algorithms on the latter.

Then what can we do about it?
=============================

A great solution to this problem is offered in [#AIM23]_, following [#WGX23]_.
It relies on a regularization of the optimal transport problem, by adding an entropic term to the Wasserstein-DRO problem.
As [#AIM23]_ explains, the term can be added either to the constraint term :math:`W_c(\hat{\mathbb{P}}^N, \mathbb{Q})` to soften the neighborhood's contours, close to what is proposed by [#WGX23]_, or one may add it to the objective function directly :math:`\sup_{\pi\in\Pi(\hat{\mathbb{P}}^N, \cdot)}\mathbb{E}_{(\xi, \zeta)\sim\pi}\left[L_\theta(\zeta)\right] - \varepsilon\mathcal{D}_{KL}(\pi\|\pi_0)`.

There is no *a priori* theoretical reason to prefer one over the other, and [#AIM23]_ studies the combination of both.
So in this library, for purely computational reasons, we prefer to perform the regularization directly in the objective because the dual formula that then emerges has less dependency on the dual parameter:

.. math::
    :label: dual_loss

    L_\theta^\texttt{robust}(\xi) := \lambda\rho + \varepsilon\log\mathbb{E}_{\zeta\sim\nu_\xi}\left[e^{\frac{L_\theta(\zeta)-\lambda c(\xi, \zeta)}{\varepsilon}}\right]

Advantages and drawebacks of skwdro
-----------------------------------

Here is sumarized what we won and lost through the regularization process.

+-------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------+--------------------------------------------+
| /                                                                                   | Pros                                                                                          | Cons                                       |
+=====================================================================================+===============================================================================================+============================================+
| :math:`\sup_{\zeta}\, L_{\theta}(\zeta)\;-\;\lambda\, c(\zeta,\xi)`                 | - No hyperparameter                                                                           | - No closed form (in general)              |
+-------------------------------------------------------------------------------------+------------------------------+----------------------------------------------------------------+--------------------------------------------+
| :math:`\varepsilon \log \mathbb{E}_{\zeta \sim \mathcal{N}(\xi,\sigma^2I)}\!\left[  | - :math:`\mathbb{E}_{\zeta \sim \mathcal{N}(\xi,\sigma^{2})}` is tractable by sampling        | - Pick :math:`\varepsilon` and             |
| e^{\left(L_{\theta}(\zeta)-\lambda\, c(\zeta,\xi)\right)/\varepsilon}               |   (e.g., MC)                                                                                  |   :math:`\sigma^{2}`                       |
| \right]`                                                                            | - **Smooth** in :math:`(\lambda, \theta)`                                                     |   (see e.g. [#AIM24]_ for some heuristics) |
+-------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------+--------------------------------------------+

If the problem at hand benefits most from WDRO, a lot of good technical solutions should be found in e.g. `the python-dro library <python-dro.org>`__.
But in most cases, its application will not be directly possible: you shoud then turn to our library to leverage :eq:`dual_loss`.

The smoothness of the *"log-average-exponential"* (i.e. log-sum-exp) expression in :eq:`dual_loss` is its main selling point: you can now plug it in you favorite SGD algorithm to get a solution, skipping theoretical work.
One of the main goals of the library is to offer the estimation of :eq:`dual_loss` on a plater, battery-included: the loss is differentiable by autodiff capabilities in order to plug it in your usual descent algotithm and some freedom is left for you to tune it through the ``PyTorch`` library.
Thus we advise readers to take a good look at the `PyTorch interface tutorial <pytorch.html>`_ to learn how to use the interfaces.

References
==========

.. [#AIM23] Azizian, Iutzeler and Malick: **Regularization for Wasserstein Distributionally Robust Optimization**, *COCV*, 2023
.. [#AIM24] Azizian, Iutzeler and Malick: **Exact Generalization Guarantees for (Regularized) Wasserstein Distributionally Robust Models**, *NIPS*, 2024
.. [#WGX23] Wang, Gao, Xie: **Sinkhorn Distributionally Robust Optimization**, *arXiv (2109.11926)*, 2023
