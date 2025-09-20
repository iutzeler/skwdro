skwdro.base.samplers.torch module
---------------------------------

This module exposes a base class :py:class:`BaseSampler` that you can subclass to build your own :math:`\pi_0` reference transport plan.
It is defined through its right-conditional :math:`\nu_\xi(\zeta)=\pi_0(\zeta|\xi)` (by slight abuse of notation), as its first marginal is fixed to be :math:`\hat{\mathbb{P}}^N` the dataset.
Formally it is defined through the disintegration lemma and its marginal property is required in order to achieve some technical feasability conditions, cf. [#AIM23]_ for clear explanations of these purposes.

Those classes have a :py:meth:`BaseSampler.reset_mean` method to dynamicaly change the mean(s) of the generating distributions.

.. automodule:: skwdro.base.samplers.torch
   :members:
   :show-inheritance:
   :undoc-members:


References
~~~~~~~~~~

.. [#AIM23] Azizian, Iutzeler and Malick: **Regularization for Wasserstein Distributionally Robust Optimization**, *COCV*, 2023
