#################
PyTorch interface
#################

.. currentmodule:: skwdro


In a general machine-learning setting, any practitionners turn to deep-learning techniques that require the use of very specific tools that cater to the "big-data" setting with massively parallel operations.
This begs for other computational architectures (e.g. GPUs, TPUs, etc), and adapted codebases.
Popular among the deep-learning comunity are three main python libraries: ``PyTorch``, ``Keras/Tensorflow``, and ``Jax``.
They offer state-of-the-art performances and a lot of utilities to build and manipulate both models as well as the training data.

In this tutorial, we will understand how to use the ``PyTorch`` interfaces in :py:mod:`skwdro` in order to robustify a simple model.
We aim at transforming any ``Pytorch``\ -parametrized loss function :math:`L_\theta` into its robust counterpart:

.. math::
    :label: dual_loss

    L_\theta^\texttt{robust}(\xi) := \lambda\rho + \varepsilon\log\mathbb{E}_{\zeta\sim\nu_\xi}\left[e^{\frac{L_\theta(\zeta)-\lambda c(\xi, \zeta)}{\varepsilon}}\right]


Model presentation
==================

Start from a simple network with one hidden layer, aiming at classifying 2-dimensional samples:

.. code-block:: python
   :linenos:

   # Specify the model
   class SimpleNN(nn.Module):
       def __init__(self, hidden_units):
           super().__init__()
           # Two hidden layers and logit output
           self.linear_relu_stack = nn.Sequential(
               nn.Linear(2, hidden_units),
               nn.ReLU(),
               nn.Linear(hidden_units, hidden_units),
               nn.ReLU(),
               nn.Linear(hidden_units, 1),
           )

       def forward(self, x):
           logits = self.linear_relu_stack(x)
           return logits

   # Instanciate it
   model = SimpleNN(32)

Usually, one would train it with a simple training procedure, looking vaguely as follows:

.. code-block:: python
   :linenos:
   :caption: Training procedure: default ERM.

   for sample, target in my_dataloader:
       # Clean the kitchen
       my_optimizer.zero_grad()

       # Forward pass
       inference = model(sample)
       loss = my_loss_function(inference, target)

       # Backward pass
       loss.backward()
       my_optimizer.step()

       # Testing
       if my_condition():
           with pt.no_grad():
               model.eval()
               print(my_loss_function(test_sample, test_target))
               model.train()

This is very simple thanks to the ease-of-use of ``PyTorch``, and now we wish to see how to use the interface to robustify this procedure.

SkWDRO's interface for robustification
======================================

The main idea of the interface comes from the fact that machine-learning models can be split into two kinds:

* Some that attempt to link an input to a target, i.e. learn a parametrized function :math:`f_\theta` such that in average on the available data :math:`y\approx f_\theta(x)`. For the sake of clarity, we distinguish two subcases:
    * Classification where ``y`` is categorical.
    * Regression where ``y`` can be anything.

* Others that prescribe some cost to available samples to be minimized (or reward, or likelihood, etc, conversly to be maximized - wolog), i.e. learn :math:`c_\theta` such that on average :math:`c_\theta` is small.

This means that the loss functions :math:`L_\theta` that we study will always look like follows for some function :math:`\ell`:

.. math::
    :label: losses_types

    L_\theta(\xi):=\begin{cases}
       \ell(y-f_\theta(x)) & \text{in the regression case}\\
       \ell(-y.f_\theta(x)) & \text{in the classification case}\\
       \ell(c_\theta(x)) & \text{in the estimation case.}
    \end{cases}

With that distinction in mind, the assumed structure of models treated by ``SkWDRO`` is as :math:`L_\theta` in its most general case: a function that for each input :math:`\xi\in\Xi` (possibly batched), outputs a scalar (batched accordingly).
This is general enough to cover the three cases mentioned, but without further assumption it leaves for ``Python`` a choice to make at the function call: do we input the ``2-uple`` :math:`\xi=(x, y)` for regression/classification and input a ``2-uple`` :math:`\xi=(x,)` in other cases, or is it better to assume some structure on :math:`\xi` at the function signature level?
Turning to the way most ``PyTorch`` models are specified, especially with regard to the native :math:`\ell` functionals (see the `torch doc on the topic <https://docs.pytorch.org/docs/stable/nn.functional.html#loss-functions>`__ e.g. :py:func:`torch.nn.binary_cross_entropy`), it seems like the second option is more idiomatic, so this ``input``\ &\ ``target`` structure is assumed.

.. admonition:: **Assumption**: Target model structure.

    We aim for a ``PyTorch`` model specified as a :py:class:`torch.nn.Module` subclass, containing the parameters :math:`\theta` as attributes.
    It should expose a overridden ``forward()`` method with the following signature with respect to the studied ``Model`` type (as ``self`` instance in python):

    .. code-block:: haskell
       :caption: Forward call specification for SkWDRO models.

       forward :: Model -> x -> Maybe y -> Float

    The inputs (``x`` and ``y``) should be batchable, in which case the output will be so.
    The ``Maybe`` on the ``y`` variable should wrap the batches (i.e. no array of ``Maybe``\ s, instead use a ``Maybe [y]``).
    Translated in ``Python``: one should be able to call:

    .. code-block:: python

       model.forward(inputs, targets)

    (whether or not ``inputs`` contains batch dimensions) in the first two cases of :eq:`losses_types`, and

    .. code-block:: python

       model.forward(inputs, None)

    in the third one.

Getting back to the training procedure described before, we see that it is not close to the default setting at all!
Thus in the ``SkWDRO`` team we attempted to provide meaningful but easy-to-use interfaces to guide you through the transformation process.

* The :py:func:`skwdro.torch.robustify` function is the main, best, and most easy to use one. Read the explanations bellow to understand how to use it in priority and leave the rest for situations that are not covered by this function.
* :py:class:`skwdro.solvers.DualLoss` and variations thereof are meant for people who already have a model complying with the assumptions above, and lets you have a lot more control on the precise pieces of the model.


``robustify``: the simplest method
----------------------------------

Here comes the documentation of the main dish, the :py:func:`~skwdro.torch.robustify` function.

.. autofunction:: skwdro.torch.robustify
   :no-index:

This may seem daunting at first glance, so let's dive in while focusing on the most important parts.

Diving into the ``robustify`` function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here are the arguments and how to use them in the example training procedure:


.. It should be given in its :py:class:`~torch.nn.Module` form! This is due to some internals of the interfaces that concatenate parameters together in some cases, e.g. to let your :math:`\ell` functional contain a subset of the parameters :math:`\theta` as well if you feel like it.
..             If your model was specified as a :py:mod:`torch.nn.functional` instead, you will need to translate it: take a look at the `torch 'nn' doc <https://docs.pytorch.org/docs/stable/nn.html#loss-functions>`__ or wrap it by hand as follows:

* The important ones:
    * **loss_**: this is the function that takes the output of your inference model :math:`f_\theta(x)` and computes the mismatch to the target :math:`y`, whether in the sense of classification or regression.
        .. warning:: The functional interface of pytorch is available for this argument specificaly, but it is recent and less tested.
            Use at your own risk. If you believe that the interface is the reason behind some bug you have, we encourage you to wrap it in a :py:class:`~torch.nn.Module` instance, as follows:

            .. code-block:: python
                :caption: Translate the functional api to object-oriented
                :linenos:
                :emphasize-lines: 6

                class MyOopLoss(torch.nn.Module):
                    def forward(self, input, target, *args, **kwargs):
                        # Here goes any reshaping necessary
                        ...
                        # Call directly the function
                        return my_functional_loss(
                            input, target,
                            reduction='none',
                            *args, **kwargs
                        )

    * **transform_** is the inference mechanism :math:`f_\theta`. In most cases, it will contain all of the parameters :math:`\theta`, e.g. it can be your linear model :py:class:`~torch.nn.Linear` or any kind of :py:class:`~torch.nn.Module` (neural nets, etc). No functional interface ios available there.
    * **rho** is the most important hyperparameter of the WDRO framework: it represents the radius of uncertainty defining the (regularized) Wasserstein ambiguity set. You can use simple cross-validation to find a suitable one for your particular problem if you do not have any idea about it, or you can turn to some tricks from the litterature, e.g. [#BMN21]_.
    * **xi_batchinit** and **xi_labels_batchinit** must be taken as a subset of the dataset to help the optimizers with a good starting point :math:`\lambda_0` for the dual parameter :math:`\lambda\ge 0` of ``SkWDRO``\ 's magic formula :eq:`dual_loss`.
    * **epsilon** is the regularization parameter determining how much we smooth the Wasserstein ambiguity region with the entropic regularization :math:`\varepsilon\mathcal{KL}(\pi\|\pi_0)`.
        One may use cross-validation to select a good one, but the beware the importance of numerical stability in this choice: even though the out-of-sample performance remain the main goal, very small values of epsilon may lead to difficulties in the optimization process.
    * **sigma** is the amount of noise of the (non-truncated) gaussian distribution that is used as :math:`\pi_0` *"reference transport plan"*. More precisely, :math:`\pi_0(\xi, \zeta):=\delta_\xi\otimes\mathcal{N}(\xi, \sigma^2I)`, so that given a sample :math:`\xi` (or batch thereof) the "adversarial" samples are sampled from :math:`\mathcal{N}(\xi, \sigma^2I)` in the "log-avg-exp" expression.
    * **n_samples** defines the number of :math:`\zeta` samples drawn for **each** :math:`\xi`. Recall that the computational efforts for the gradient step thus goes from :math:`\mathcal{O}(B.d)` for ERM with batchsize :math:`B` and dimension :math:`d` to :math:`\mathcal{O}(B.d.\texttt{n_samples})`, which can be significant if you want precise gradients.

* The not so important ones:
    * **post_sample** can be set to false to sample the :math:`\zeta` adversarial samples only once at the beginning of the optimization procedure, if you are doing fullbatch optimization in the first place (i.e. GD, not SGD). This opens the door to more performant algorithms such as :py:class:`torch.optim.BFGS`.t the expense of statistical soundness of the estimation of the logsumexp expression.
    * **cost_spec** is string-like specification defining the ``2-uple`` ``(k, p)``, in order to specify the cost functional for the Wasserstein distance as a ``p``\ -th power distance :math:`\|\zeta-\xi\|_k^p` in some :math:`\|\cdot\|_k`\ -Banach space. See the :py:func:`skwdro.base.cost_decoder` function's sources for more details.
    * **imp_samp** can be set to false to disable importance sampling on the inner expectation of the *"log-avg-exp"* expression when it would otherwise be enabled (when :math:`p=k=2`).
    * **adapt** can be set to either ``"mechanic"`` or ``"prodigy"`` to set up an automatic learning rate tuner based on the adam optimizer for the builtin optimizer of the ``SkWDRO`` loss function. Otherwise, set it to ``None`` to get the regular :py:class:`~torch.optim.AdamW` implementation. **learning_rate** can be set to any positive floating point number in order to specify the stepsize of ``Adam`` in this case.


This function is meant to perform two tasks as one: merge your loss function with your inference model, if need be, and build the dual loss displayed in :eq:`dual_loss`.
The output is a :py:class:`skwdro.oracle_torch.DualLoss` object/module that represents :math:`L_\theta^\texttt{robust}` from :eq:`dual_loss`.

Before/after comparison for ``robustify``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
    :linenos:
    :caption: Training procedure: SkWDRO with ``robustify``.
    :emphasize-lines: 1-9
    :emphasize-added: 18,29,30
    :emphasize-removed: 16,17,28

    robust_model = robustify(
        loss,
        model,
        pt.tensor(0.01),  # the radius you picked for the Wasserstein ball
        *next(iter(my_dataloader)),
        # Optionally set those keyword-only HPs:
        epsilon = 1e-3,
        sigma = 0.1
    )

    for sample, target in my_dataloader:
        # Clean the kitchen
        my_optimizer.zero_grad()

        # Forward pass
        inference = model(sample)
        loss = my_loss_function(inference, target)
        loss = robust_model(inference, target)

        # Backward pass
        loss.backward()
        my_optimizer.step()

        # Testing
        if my_condition():
            with pt.no_grad():
                model.eval()
                print(my_loss_function(test_sample, test_target))
                # Note: to perform forward inference, use robust_model.primal_loss.transform
                print(robust_model.primal_loss(test_sample, test_target))
                model.train()

Dual losses: tune everything you want
-------------------------------------

Conceptually, :eq:`dual_loss` is specified in its most general form by the following building blocks:


+-----------------------------+-------------------------------------------------------------------+----------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| /                           | Math notation                                                     | Notation in the codebase (with link if relevant)                                                   | Examples                                                                                                                                                                                                     |
+=============================+===================================================================+====================================================================================================+==============================================================================================================================================================================================================+
| A loss function             | :math:`L_\theta(\zeta)`                                           | ``primal_loss``/``loss``                                                                           | :math:`(\zeta_y - \left\langle\theta|\zeta_x\right\rangle)^2`, :math:`\log\left(1+e^{\zeta_y\left\langle\theta|\zeta_x\right\rangle}\right)`, etc                                                            |
+-----------------------------+-------------------------------------------------------------------+----------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| A cost functional           | :math:`c(a, b)`                                                   | :py:class:`~skwdro.base.costs_torch.Cost` (`tuto <tutos/costs.html>`__)                            | :math:`\|b-a\|_k^p`, :math:`\begin{cases}0&\text{if }a=b\\ 1&\text{otherwise}\end{cases}`, etc                                                                                                               |
+-----------------------------+-------------------------------------------------------------------+----------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| A reference transport plan  | :math:`\nu_\xi(\zeta):=\pi_0(\xi, \zeta)`                         | :py:class:`~skwdro.base.samplers.torch.base_samplers.BaseSampler` (`tuto <tutos/samplers.html>`__) | :math:`\mathcal{N}(\zeta | \xi, \sigma^2I)`, :math:`\mathcal{U}_{\left[\xi-\frac{\sigma}2, \xi+\frac{sigma}2\right]}(\zeta)`, :math:`\mathcal{U}_{\{0, \dots, 255\}}(\zeta)`, :math:`\delta_\xi(\zeta)`, etc |
+-----------------------------+-------------------------------------------------------------------+----------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

:py:mod:`skwdro` lets you build your custom robust loss function representing the dual formula :eq:`dual_loss` through its second main interface: :py:class:`skwdro.solvers.DualLoss`.

Diving into the ``DualLoss`` class(es)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The main one anyone will want to try, that is aliased by :py:class:`~skwdro.solvers.DualLoss`, is the following.

.. autofunction:: skwdro.solvers.DualPostSampledLoss
    :no-index:

It has a sister-class :py:class:`~skwdro.solvers.DualPreSampledLoss`, that will keep the same sampled :math:`\zeta` values for the inner expectation, needing only to sample it once at the expense of statistical soundness, coming from an idea of [#WGX23]_.

In order to get the building blocks for the first two arguments of the constructor described above, here is a simple receipe:

* build your own :math:`L_\theta` loss, either,
    * by combining the :math:`\ell` functional with your inference model :math:`f_\theta`/:math:`c_\theta`, using the :py:class:`skwdro.base.losses_torch.WrappedPrimalLoss` helper class,
    * or by reusing any loss already available in :py:mod:`skwdro.base.losses_torch`,

* then get yourself a cost functional tailored to the geometry and properties of your space of interest,
    * by subclassing :py:class:`skwdro.base.costs_torch.Cost`,
    * or by using any of the already available ones in :py:mod:`skwdro.base.costs_torch`,

* get a good sampling strategy to explore adversarial samples according to the association ``cost``-``space``-``prior knowledge``,
    * build your own sampler :math:`\nu_\xi` by subclassing :py:class:`skwdro.base.samplers.torch.BaseSampler`,
    * use the sampler generated by the geometry of your space, i.e. linked to the cost you chose previously, using the :py:mod:`skwdro.base.samplers.torch.cost_samplers` module's helpers,
    * fetch the sampler best suited (according to the chef's menu, if available, see the :py:module:`skwdro.base.losses_torch` module) to the problem you are solving by calling your losses :py:meth:`skwdro.base.losses_torch.Loss.default_sampler` method.

.. note::

    To build your loss function, you will need to have chosen a sampler, and to pick it you may want to use the :py:meth:`skwdro.base.losses_torch.Loss.default_sampler` utility.
    While this may sound a bit circular, one may set to starting sampler of any loss to ``None`` and then overwrite it dynamically with the setter method:

    .. code-block:: python
        :caption: Set a sampler after initialization (Logistic regression).
        :linenos:

        sigma = torch.tensor(.01)
        loss = skwdro.base.losses_torch.LogisticLoss(None, d=10)
        loss.sampler = loss.default_sampler(xi, xi_labels, sigma)

Before/after comparison for ``DualLoss``\ es
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
    :linenos:
    :caption: Training procedure: SkWDRO with ``DualLoss``.
    :emphasize-lines: 1-35
    :emphasize-added: 55,56,44
    :emphasize-removed: 42,43,54

    SEED = 42
    xi_warmup, xi_labels_warmup = next(iter(my_dataloader)),

    # Ingredient 2: the cost functional
    # Pick \|x-y\|_2^2
    cost = NormLabelCost(
        2., 2., 1000.
    )

    # Ingredient 3: the sampling distribution
    sampler = LabeledCostSampler(
        cost,
        xi_warmup, xi_labels_warmup,
        sigma = 0.1,
        seed = SEED
    )

    # Ingredient 1: the loss function
    loss_functional = WrappedPrimalLoss(
        my_loss_function,  # as a torch.Module!
        model,
        sampler,
        has_labels=True
    )

    # Mix together, and let it rest
    robust_model = DualLoss(
        loss,
        cost,
        10,
        pt.tensor(1e-3),  # the regularization parameter
        pt.tensor(0.01),  # the radius you picked for the Wasserstein ball
        # Optionally set those keyword-only HPs:
        epsilon = 1e-3,
        sigma = 0.1
    )

    for sample, target in my_dataloader:
        # Clean the kitchen
        my_optimizer.zero_grad()

        # Forward pass
        inference = model(sample)
        loss = my_loss_function(inference, target)
        loss = robust_model(inference, target)

        # Backward pass
        loss.backward()
        my_optimizer.step()

        # Testing
        if my_condition():
            with pt.no_grad():
                model.eval()
                print(my_loss_function(test_sample, test_target))
                # Note: to perform forward inference, use robust_model.primal_loss.transform
                print(loss_functional(test_sample, test_target))
                model.train()

Conclusion
==========

The take away message of this small tutorial is that you can either use the simple interface of :py:func:`skwdro.torch.robustify` to change only a few lines in your codebase, or you can get more control over the details of the algorithm by using the :py:class:`skwdro.solvers.DualLoss` interfaces.
We advise new users to turn to the first, while users who try to study the behavior of ``SkWDRO`` in more details may take a look at various strategies proposed by the latter through the various aforementioned modules (for costs, samplers, losses, etc).

References
==========

.. [#BMN21] Blanchet, Murthy and Nguyen: **Statistical Analysis of Wasserstein Distributionally Robust Estimators**, *TutORials in Operations Research*, 2021
.. [#WGX23] Wang, Gao, Xie: **Sinkhorn Distributionally Robust Optimization**, *arXiv (2109.11926)*, 2023

.. Operations Research
.. ===================
.. 
.. Portfolio Selection
.. -------------------
.. 
.. Two losses are implemented in PyTorch: RiskPortfolioLoss_torch et MeanRisk_torch.
.. 
.. The general loss function is given by
.. 
.. .. math::
.. 
..    \underset{\theta \in \Theta}{\inf} \mathbb{E}^{\mathbb{P}}(- \langle \theta, \xi \rangle) + \eta \mathbb{P}\text{-CVaR}_{\alpha}(- \langle \theta, \xi \rangle)
.. 
.. 
.. RiskPortfolioLoss_torch thus implements the first term of the loss function
.. which is purely based on the dot product :math:`- \langle \theta, \xi \rangle`
.. representing the risks associated with return on investment. The idea of implementing this term
.. in a separate class stems from the fact that there are several functions which exist
.. to model these financial risks. Thus, the MeanRisk_torch
.. class implementing the function to be minimized is defined by composition with an instance of the
.. RiskPortfolioLoss_torch class as an attribute to define the first term.
.. This maintains the overall structure of the general class
.. since the definition of a new risk modeling function
.. requires the creation of its own class,
.. and the rest of the loss function is defined by composition with an instance of this class.
