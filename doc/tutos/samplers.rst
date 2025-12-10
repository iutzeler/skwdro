==============================================
Recipe for a good sampling strategy for SkWDRO
==============================================

:bdg-primary:`Practice`

.. tip:: Read the `tutorial on SkWDRO <../why_skwdro.html>`__ to understand better this part.


Recall the formula for SkWDRO:

.. math::

    L_\theta^\texttt{robust}(\xi) := \lambda\rho + \varepsilon\log\mathbb{E}_{\zeta\sim\nu_\xi}\left[e^{\frac{L_\theta(\zeta)-\lambda c(\xi, \zeta)}{\varepsilon}}\right].

Its inner expectation :math:`\nu_\xi` is cruxial to the estimation, as it sheds light on the *a priori* knowledge you will incorporate into the estimation of the true risk measure.
We will answer the question of **how to pick this crucial hyperparameter**.

Use pre-built options
=====================

The library relies on various pre-built combinations of distributions, covering modelisation with and without targets (more details on the `wdro presentation <../wdro.html>`_).

Problem-specific examples
-------------------------

Some samplers for specific robust models have been implemented to provide guidelines for inspiration of astute readers of their source code, as well as ready out-of-the-box options.

Almost all of the pre-implemented samplers rely on a normal distribution for the input data, with varying distributions for the targets (which, recall, are called `labels` in the code's nomenclature in order to catter to machine learning applications).
- For a Bernoulli law on targets :math:`\{-1, 1\}`, see :py:class:`skwdro.base.samplers.torch.ClassificationNormalBernouilliSampler`.
- To sample a dirac distribution on :math:`\xi_\texttt{labels}`, see :py:class:`skwdro.base.samplers.torch.ClassificationNormalIdSampler`.
- If the target's space has less structure, use a normal distribution (e.g. for regression tasks, even though the name suggests otherwise), see :py:class:`skwdro.base.samplers.torch.ClassificationNormalNormalSampler`.
- If you have problems that do not have targets, you can draw inspiration from either :py:class:`skwdro.base.samplers.torch.PortfolioNormalSampler` or :py:class:`skwdro.base.samplers.torch.NewsVendorNormalSampler`. For an example without a normal distribution on the inputs, see :py:class:`skwdro.base.samplers.torch.PortfolioLaplaceSampler`.

Cost-based sampling
-------------------

If your cost functional has been crafted with care, it might include a default sampling option defined as singleton of pair o:py:class:torch.distributions.Distribution`\ (s).
In this case, defining a sampler is more natural and can be done from the cost's methods through two helper classes.

.. autoclass:: skwdro.base.samplers.torch.LabeledCostSampler
   :members:
   :inherited-members:
   :no-index:

.. autoclass:: skwdro.base.samplers.torch.NoLabelsCostSampler
   :members:
   :inherited-members:
   :no-index:




Build your own sampler
======================

The library exposes three main interfaces to make your own sampling strategy.
What they will implement is a conditional measure of probability :math:`\nu_\xi` that, given a reference :math:`\xi` value (or batch thereof), will sample a batch of :math:`\zeta` realisations.

Some background to make educated guesses
----------------------------------------

This measure stems theoretically from the `entropic regularization <../why_skwdro.html>`__ of the WDRO problem: the addition of the term :math:`\mathcal{D}_\text{KL}(\pi, \pi_0)` in the objective function of the primal problem will, after some lagrangian duality induced manipulations (see [#AIM23]_), produce the logsumexp structure of :eq:`dual_loss`.
In this procedure, a disintegration lemma must be used to split the reference generator :math:`\pi_0(\xi, \zeta)` into two parts. Under heavy abuse of notation, it would write as follows.

.. math::

   P_{\pi_0}(\xi\in X\cap\zeta\in Y) = \underbrace{P_{\pi_0}(\xi\in X)}_{\hat{\mathbb{P}}^N(X)}\underbrace{P_{\pi_0}(\zeta\in Y|\xi\in X)}_{\nu_\xi(Y)}.

.. note:: A common wisdom is that usually, if :math:`\nu_\xi` does depend on :math:`\xi`, it should better average to it i.e.

   .. math::

      \mathbb{E}_{\zeta\sim\nu_\xi}[\zeta|\xi]\approx\xi.

   This property is by no means necessary, but proves useful in verifying one of the assumptions from [#AIM23]_ which is strict feasability of the reference distribution (i.e :math:`\mathbb{E}_{(\xi, \zeta)\sim\pi_0}[c]<\rho`).
   Otherwise, choosing carfully :math:`\nu_\xi` such that it is independant of :math:`\xi` may be relevant for some applications.

In general, one must consider well-posed problems in which :math:`\nu_\xi` has good properties: it is not too far from :math:`\hat{\mathbb{P}}^N` but it also explores far enough of it to provide robustification, i.e. gets closer to the true problem's distribution :math:`\mathbb{P}`.

The common interface for samplers
---------------------------------

To build your own sampler, you can write a class that inherits :py:class:`skwdro.base.sampler.torch.BaseSampler`, or to be more precise inherit one of the following classes:

.. autoclass:: skwdro.base.samplers.torch.LabeledSampler
   :members:
   :inherited-members:
   :no-index:

.. autoclass:: skwdro.base.samplers.torch.NoLabelsSampler
   :members:
   :inherited-members:
   :no-index:

These two templates leave for you to define **only two methods**: the constructor and a special :py:meth:`skwdro.base.samplers.torch.BaseSampler.reset_sampler` method that defines how to change the parameters of :math:`\nu_\xi` if :math:`\xi` changes (resetting dynamically the :py:class:`torch.distributions.Distribution` objects if needs be.
But while it is not mendatory, one may rewrite to their liking some custom methods, including the central :py:meth:`skwdro.base.samplers.torch.BaseSampler.sample` method used by the library.

Learning by examples: a case study on mixed-features WDRO
---------------------------------------------------------

Say now you want to implement a logistic regression model based on a mixture of continuous and discrete features as described in [#BSW23]_ which  proposes, between other tools, a cutting-plane algorithm to solve WDRO formulation with these features. As a case study, we explain here how to use SkWDRO to approximate the solution of this problem.

Let's consider that the problem is formulated such that the `n_continuous_features`\ +\ `n_discrete_features` features are concatenated in an input variable `xi`, and target labels are in `xi_labels`. Consider the discrete features to be first one-hot encoded and then recentered to :math:`\{-1, 1\}`, just like in the usual logistic regression from the documentation. Here is how we would build the sampler for such a problem in pytorch.

.. code-block:: python
   :linenos:
   :caption: Constructor method for mixed features sampler
   :emphasize-lines: 44, 69-80

   import torch
   from skwdro.base.samplers.torch import LabeledSampler, IsOptionalCovarianceSampler
   import torch.distributions as dst


   class MixedFeaturesSampler(LabeledSampler, IsOptionalCovarianceSampler):
       data_s: dst.MultivariateNormal
       labels_s: dst.TransformedDistribution
       discrete_features_s: dst.TransformedDistribution
       """
       This class samples both continuous and discrete features of the design space.
       The inputs ``xi`` are assumed to follow a layout in which the continuous
       features are arranged at the beginning of the features vector and the
       discrete ones follow, with the labels treated separately as the
       ``xi_labels`` variable.
       The concatenation in ``xi`` of the features must be done on the last axis,
       the categorical variables are encoded in {-1, +1} with -1 representing the negation
       of the class at a given index while +1 represents a realisation of the class.
       Any value between -1 and 1 may also be used to represent an unsure class. To allow
       this, all the tensor must be encoded as one common floating type (e.g. float32).
       """
       def __init__(
           self,
           xi, xi_labels,
           n_continuous_features,
           n_discrete_features,
           # Probability of switching a class
           p,
           *,
           # reusing the same trick for covariance matrices as in the logreg
           sigma = None,
           tril = None,
           prec = None,
           cov = None,
           seed = None,
       ):
           assert 0. <= p <= 1.
           self.p = p

           self.n_continuous_features = n_continuous_features
           self.n_discrete_features = n_discrete_features
           assert xi.size(-1) == n_continuous_features + n_discrete_features

           xi_cont, xi_discr = torch.split(xi, [n_continuous_features, n_discrete_features], dim=-1)

           # See the source code of IsOptionalCovarianceSampler to see how
           # to specify covariance matrices
           covar = self.init_covar(n_continuous_features, sigma, tril, prec, cov)

           # Recycle code from usual logreg samplers
           super().__init__(
               # Continuous part of the input sampler
               dst.MultivariateNormal(
                   loc=xi_cont,
                   **covar  # type: ignore
               ),
               dst.TransformedDistribution(
                  dst.Bernoulli(
                      p
                  ),
                  dst.transforms.AffineTransform(
                      loc=-xi_labels,
                      scale=2 * xi_labels
                  )
               ),
               seed
           )

           # Discrete part of the input sampler
           # Implements a random switch of the class indicator for each class of
           # each discrete feature.
           self.discrete_features_s = dst.TransformedDistribution(
               dst.Bernoulli(
                   p
               ),
               dst.transforms.AffineTransform(
                   loc=-xi_discr,
                   scale=2 * xi_discr
               )
           )

The method that we should add to this class is the sampling procedure.
It is composed of the sampling of the continuous input variable, its categorical part, and the perturbation of the labels.

.. code-block:: python
   :linenos:
   :caption: Sampling strategy for mixed-features regression
   :emphasize-lines: 29

       def sample_labels(self, n_sample):
           """
           Samples the target labels through Bernoulli swaps.
           Overrides w/ ``sample`` to prevent ``rsample`` from crashing since bernoulli
           isn't reparametrizeable.
           """
           zeta_labels = self.labels_s.sample(torch.Size((n_sample,)))
           assert isinstance(zeta_labels, torch.Tensor)
           return zeta_labels

       def sample_discrete_features(self, n_sample):
           """
           Samples the categorical (discrete) input features through Bernoulli swaps.
           Overrides w/ ``sample`` to prevent ``rsample`` from crashing since bernoulli
           isn't reparametrizeable.
           """
           zeta_discrete = self.discrete_features_s.sample(torch.Size((n_sample,)))
           assert isinstance(zeta_discrete, torch.Tensor)
           return zeta_discrete

       def sample(self, n_samples):
           """
           Overwrite of LabeledSampler's method.
           This is the function that will be called by the library internally to sample
           the conditional distribution of inputs ``(zeta, zeta_labels)``.
           """
           zeta_cont = self.sample_data(n_samples)
           zeta_discr = self.sample_discrete_features(n_samples)
           zeta = torch.cat([zeta_cont, zeta_discr.to(zeta_cont)], dim=-1)
           zeta_labels = self.sample_labels(n_samples)
           return zeta, zeta_labels

Finally, we tackle the mendatory part linked to the reset of the sampler's moments ``xi`` and ``xi_labels``.

.. code-block:: python
   :linenos:
   :caption: Sampling strategy for mixed-features regression

       def reset_mean(self, xi, xi_labels):
           self.__init__(
               xi, xi_labels,
               self.n_continuous_features, self.n_discrete_features,
               self.p,
               tril=self.data_s._unbroadcasted_scale_tril
           )


Testing the snippets above on some fake data

.. code-block:: python

   >>> # Gen some fake data
   >>> xi_c = torch.randn((100, 3))
   >>> xi_d = torch.randint(-1, 1, (100, 10))
   >>> xi_d[xi_d== 0.] = -1
   >>> xi_l = torch.randint(-1, 1, (100, 1))
   >>> xi_l[xi_l== 0.] = -1
   >>> xi = torch.cat((xi_c, xi_d), dim = -1)
   >>> #
   >>> # Test the sampler
   >>> s = MixedFeaturesSampler(xi, xi_l, 3, 10, 0.1, sigma=0.1)
   >>> print(s.sample(10)[0].shape)
   torch.Size([10, 100, 13])
   >>> s.reset_mean(torch.cat((xi_c, xi_d), dim = -1)*0.1, xi_l)
   >>> print(s.sample(1)[0])
   tensor([[[-0.0778,  0.2271, -0.3122,  ..., -0.1000, -0.1000, -0.1000],
            [-0.1395,  0.1225, -0.1504,  ..., -0.1000,  0.1000, -0.1000],
            [ 0.0092,  0.0816, -0.0500,  ..., -0.1000, -0.1000, -0.1000],
            ...,
            [-0.0248,  0.0565,  0.1841,  ..., -0.1000, -0.1000, -0.1000],
            [ 0.1627, -0.0517,  0.0814,  ..., -0.1000,  0.1000, -0.1000],
            [ 0.1598, -0.1213, -0.1780,  ..., -0.1000, -0.1000,  0.1000]]])

References
==========
.. [#AIM23] Azizian, Iutzeler, and Malick: **Regularization for Wasserstein Distributionally Robust Optimization**, *COCV*, 2023
.. [#BSW23] Belbasi, Selvi, and Wiesemann: **It’s All in the Mix: Wasserstein Classification and Regression with Mixed Features**, *ArXiV* (https://arxiv.org/abs/2312.12230), 2023
