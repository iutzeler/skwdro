=================================================
Recipe for a good ground-cost for Wasserstein-DRO
=================================================

.. tip:: Read the `tutorial on SkWDRO <why_skwdro.html>`__ to understand better this part.


Recall the formula for SkWDRO:

.. math::
    :label: dual_loss

    L_\theta^\texttt{robust}(\xi) := \lambda\rho + \varepsilon\log\mathbb{E}_{\zeta\sim\nu_\xi}\left[e^{\frac{L_\theta(\zeta)-\lambda c(\xi, \zeta)}{\varepsilon}}\right].


It includes a cost function :math:`c` that imposes a notion of **geometry** in the design space of the samples :math:`\xi,\zeta\in\Xi`.
We will answer the question of **how to pick this crucial hyperparameter**.

Distance structure
==================

Many of the examples of robustness treated in the litterature showcase costs of the form

.. math::
   c(\xi, \zeta) = d(\xi, \zeta)^p

for some power :math:`p\ge 1`.
These are especially well treated in the litterature for their nice distance properties that are transmitted to the transport cost, making it the so-called **Wasserstein distance** [#V09]_ (see last section for implementation details).

Simple cases: turn to euclidean geometry
----------------------------------------

If your problem is formulated in simple cases in which no structure is prescribed on the space of samples :math:`\Xi`, your cost should look like the euclidean norm:

.. math::
   :label: euclidean_cost

   c(\xi, \zeta) := \|zeta-\xi\|_2

Then, this opens a wide range of Wasserstein **distances**, called the :math:`W_p` distances in the litterature, for which you raise this norm to some power and change the allowed radius :math:`\rho` accordingly.

.. math::
   :label: wp_cost

   c(\xi, \zeta) := \|zeta-\xi\|_2^p

The choice of `p` can be made in accordance with the behaviour of the loss function :math:`L_\theta` in its "worst regions".
To learn more about the growth criteria that make most sense regarding this remark, take a look at [#GCK24]_.

The interface of SkWDRO with regard to this matter is very simple to use: in the `robustification interface <api_deepdive/robustification.html>`__ you may directly specify such a structure by using a `decoded string following our simple grammar <api_deepdive/cost_decoding.html>`__.

Here, this is done with the following cost specification:

.. code-block:: python
   :caption: specification of the cost power

   p: float = 1.  # pick to your liking
   cost_spec = f"t-NC-2-{p}"

.. tip:: Notice that if you choose :math:`p=2`, you gain access to an efficient importance-sampling algorithm. More on this in another tutorial...

Other norm-based costs
----------------------

Then, if you want to impose more geometry to interpret your results in a different way (e.g. comparing to the robustness induced by FGSA, or anything like this), you can change the norm to a different one by specifying a :math:`k` parameter.

.. code-block:: python
   :caption: specification of the norm type

   p: float = 1.
   k: float = 1.  # pick to your liking
   cost_spec = f"t-NC-{k}-{p}"

What about targets?
-------------------

If the loss function relates to a classification or regression task, then one may allow for targets to be subject to uncertainty as well.
The cost specification interface lets you account for that as well, by switching from the usual "Norm-Cost" (``NC``) to a "Norm-with-Label-Cost" (``NLC``):

.. code-block:: python
   :caption: specification of the norm type

   p: float = 1.
   k: float = 1.  # pick to your liking
   cost_spec = f"t-NLC-{k}-{p}"

.. tip:: Again, picking the ``2-2`` combination unlocks the importance-sampling algorithm.

Building you own cost function
==============================

Many applications of WDRO do not fall into the setting above.
From optics to discrete optimisation, they need to impose other kinds of structure via the cost function.
So instead of trying to cover every case by hand, we allow users to subclass :py:class:`skwdro.base.costs_torch.Cost` in order to implement their own.

The documentation of this useful abstract class will guide you through the methods you need to implement:

.. autofunction:: skwdro.base.costs_torch.TorchCost
    :no-index:

The methods you should override are the following:

- the :py:meth:`skwdro.base.costs_torch.Cost.value` forwarding method that should take couples :math:`(\xi, \zeta)`, specified as four arguments ``(xi, xi_labels, zeta, zeta_labels)`` (with ``_label`` arguments allowed to be set to ``None``). It should return the cost incurred for transporting one unit of mass from :math:`\xi` to :math:`\zeta`.
- the :py:meth:`skwdro.base.costs_torch.Cost._sampler_data` method is useful if you wish to build a :py:class:`skwdro.base.samplers.torch.NoLabelsCostSampler` or :py:class:`skwdro.base.samplers.torch.LabeledCostSampler`. Let it return the :py:class:`torch.distributions.Distribution` instance from which to sample data points, given :math:`\xi`.
- same goes for the :py:meth:`skwdro.base.costs_torch.Cost._sampler_labels` method if your model has targets. It can return a ``None`` value instead of a distribution if your whole setup handles ``None`` as labels.
- the :py:meth:`skwdro.base.costs_torch.Cost.solve_max_series_exp` method lets you use importance sampling if you believe that it is well defined for your cost function and the structure of the problem studied. If not, make it ``raise`` and set all importance sampling flags to ``False`` at the creation of the loss function.

Illustration on some new example
--------------------------------

Consider a problem stemming from some gaussian curvature prescription model, or unbalanced WDRO [#GGV21]_:

- the space of available samples is the sphere :math:`\Xi = \mathcal{S}^{d-1}`,
- the cost function is log-bilinear for samples pointing in the same half-space

   .. math::
      :label: cost-GCP

      c(\xi, \zeta) = -\log(\texttt{ReLU}[\left\langle\zeta, \xi\right\rangle]) + \chi_{\{(x, y)|\langle x, y\rangle > 0\}}(\xi, \zeta).


If we want to use this structure to build some WDRO model, you may implement this cost functional as is done bellow.

.. code-block:: python
   :linenos:
   :caption: Cost function for gaussian curvature prescription

   import torch as pt
   from skwdro.distributions import Distribution
   from skwdro.base.costs_torch import Cost


   class HalfSphereUniform(Distribution):
      """
      Proposition of a torch ``Distribution`` that samples uniformly on the half-sphere
      pointing in the same direction as the "center" ``xi``. The expectation of this distribution is
      thus ``xi``.
      """
      def __init__(self, xi):
         super().__init__(pt.Size(), xi.shape, False)
         self.center = xi

      def rsample(self, sample_shape = pt.Size()) -> pt.Tensor:
         """
         Generates a sample_shape shaped reparameterized sample or sample_shape
         shaped batch of reparameterized samples if the distribution parameters
         are batched.

         Samples an isotropic gaussian, then projects the samples to the right half-space of
         positive scalar product with the "center" ``xi``, and finally projects them on the
         sphere.
         """
         noise = pt.randn(sample_shape + self.center.size())
         dim_range = tuple(range(len(sample_shape), noise.dim()))
         projected_noise = noise * pt.sign(-pt.sum(self.center * noise, dim=dim_range, keepdim=True))
         return projected_noise / pt.linalg.norm(projected_noise, dim=dim_range, keepdim=True)


   class GaussianPrescriptionCost(Cost):
      def __init__(self):
         # This initialization procedure is not important
         super().__init__(
            "Gaussian-curvature-prescription cost functional",
            "pt"
         )
         # Here is the important line: the homogeneity for the radius
         # is set bellow (see next section if you are curious).
         self.power: float = 1.

      def value(self, xi, xi_labels, zeta, zeta_labels):
         r"""
         This value function computes the cost of a pair (``xi``, ``zeta``).

         .. math::
             c(\xi, \zeta):=-\log\left([\langle\zeta,\xi\rangle]_+\right)
         """
         assert xi_labels is None
         assert zeta_labels is None
         # Write the cost function here, using only pytorch functions to
         #    allow all the internal machinery to go smoothly. Here e.g.
         #    we leverage the relu function to compute max(0, <x,y>)
         scalar_prod = (xi * zeta).sum(dim=-1)
         if scalar_prod <= 0.:
            return pt.zeros_like(scalar_prod)
         return pt.log(pt.nn.functional.relu(scalar_prod))

      def _sampler_data(self, xi, epsilon):
         del epsilon
         return HalfSphereUniform(xi)

      def _sampler_labels(self, xi_labels, epsilon) -> None:
         assert xi_labels is None
         return None

      def solve_max_series_exp(self, xi, xi_labels, rhs, rhs_labels):
         assert xi_labels is None
         assert rhs_labels is None
         return xi, xi_labels


The definition of the half-sphere sampler is not mendatory! You may use any placeholder you want for the ``_sampler_data`` method overload, and follow the `user guide <user_guide.html>`_ to see how to implement you own custom sampler and integrate it to the dual-loss formulation.

This cost function is not associated trivially to any learning problems, but it showcases the way you can impose geometrical structure on the ``SkWDRO`` framework in general.

To go further
-------------

Getting back to the first `WDRO tutorial <wdro.html>`_, you may recall that the transport cost constraint was cast as follows

.. math::

   W(\hat{\mathbb{P}}^N, \mathbb{Q}) \le \rho.

But in order to get a true **distance**, one may not use any cost function!
So e.g. in the case of distances put to some power :math:`p` as in :py:class:`~skwdro.base.costs_torch.NormCost`, one must acknowledge that the power :math:`p` must be taken into account in the radius.
For example, in the litterature, the distance is defined in a straightforward way for any distance :math:`d`

.. math::

   W_p(\hat{\mathbb{P}}^N, \mathbb{Q}) := \sqrt[p]{\inf_{\pi\in\Pi(\hat{\mathbb{P}}^N, \mathbb{Q})} \int_{\Xi^2}d(\xi, \zeta)^p d\pi(\xi, \zeta)} \le \rho

To avoid changing all the theoretical derivations related to the duality results in the `SkWDRO framework <why_skwdro.html>`_, we just raise both sides of the equation to the power :math:`p`, which translates to appropriate tricks inside of the libraries solvers.

.. tip:: In fact the interface is flexible enough to specify this behavior for any loss! Set the :py:attr:`skwdro.base.costs_torch.TorchCost.p` attribute to any positive floating point number to allow for a power parameter, defining to what power you want to raise both sides of the equation so that the cost is of the same order of magnitude in average as :math:`\rho`. See the example above.

References
==========
.. [#V09] Villani: **Optimal transport: old and new**, 2009
.. [#GCK24] Gao, Chen and Kleywegt: **Wasserstein Distributionally Robust Optimization and Variation Regularization**, *Operations Research*, 2024
.. [#GGV21] Gallouët, Ghezzi, and Vialard: **Regularity theory and geometry of unbalanced optimal transport**, 2021
