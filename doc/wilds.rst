==============================
Some results on a real dataset
==============================

The library can be applied to a wide range of difficult problems in machine
learning, and since it has been mostly showcased on more simple cases (either
linear problems or low dimensional ones), this short page aims at describing
some results related to a more realistic example of difficult model.

The dataset
===========

The `iWildsCam dataset <https://wilds.stanford.edu/>`_ is composed of images
of animals from various places on earth. They are labeled with their specie
title among 60 possible labels, as well as a location on earth.
The dataset is then split in such a way that the training/validation set contains
images from a fixed (non-exhaustive) set of locations, and the test set contains
images from other locations, absent from the training set.

This specific split of the dataset presents a practical example of a
**distribution shift** in the dataset. Indeed the testing set contains visual
features that are absent of the training set for a given animal, and hence cannot
be seen by the machine learning model used. So it must accomodate for this shift
during its training in order to obtain good test results.

Methodology
===========

The data receives a pre-treatement as described in [#MRPH24]_, using their trained
neural network to provide a fix set of pretrained features that must be classified.
As described in their paper, those pretrained features come from a Resnet50 network
pretrained on Imagenet, reusing their rich intermediate representations at late
layers.

Both a multiclass logistic regression classifier and a shallow (two-layers)
neural network are tested.
We report those results bellow, showing how the optimisation procedure manages to
achieve good accuracies in multiple hyperparameters setting.

Results
=======

We show the results obtained from run scripts one may find in our supplementary
`experiments repository <https://github.com/floffy-f/skwdro-experiments>`_,
running the following command:

.. code-block:: console
   :caption: Launch this command in your terminal to run the Wilds experiments

   $ uv run optim_script.v2.py -s 0.001 -is on -l -m train
   $ uv run optim_script.v2.py -s 0.001 -is on -l -m plot_acc

.. warning:: As a disclaimer: this part of the code is not per se part of the
   ``SkWDRO`` library, hence it does not abide to its quality standards. It is
   meant as a research script to investigate this specific dataset, and did not
   receive the same care as the library itself.

You may of course play with the hyperparameters available. By default, the
options let you train a shallow network with one hidden layer (of 64 neurons).
You can change the training hyperparameters, but you will need to dive into the
code to change more subtle settings like the architecture. Still, a linear model
(with the ``-c`` flag) is available.

Please refer to the output of the help section to get more details:

.. code-block:: console
   :caption: Help section

   $ uv run optim_script.v2.py --help

The neural net example
----------------------

The training outcomes for the neural network is as follows:

.. image:: assets/wilds/train_wilds.png

Notice how the overfitting behaviour changes substantially with the robustness radius.

* For small values of :math:`\rho`, the accuracy raises in the first hundred iterations,
  and then goes down as the training procedure overfits the training set in the long run.
* In contrast for higher values of the radius, the accuracy raises steadily. The training
  loss (measured as the WDRO dual objective described in `this tutorial <why_skwdro>`__)
  is higher, displaying its more pessimistic nature.

As a followup on
`the explanations we give on the lambda optimisation landscape <examples/Study/plot_lambda_landscape.html>`__,
one may study the results of the :math:`\lambda` optimisation depending on the
chosen radius, and how much it changes. This way we may deduce how much importance
we give to its optimisation.

.. code-block:: console
   :caption: See how much lambda varies depending on the problem studied.

   $ uv run optim_script.v2.py -s 0.001 -is on -l -m tb_lam
   Rho      | Lmin  | Lmax  | ratio
   -       -       -       -
   ρ=1.0e-06        | 109672        | 109752 | 1.00
   ρ=1.0e-05        | 10975         | 11054 | 1.01
   ρ=1.0e-04        | 1098  | 1180 | 1.08
   ρ=1.0e-03        | 110   | 188 | 1.71
   ρ=1.0e-02        | 11    | 102 | 9.32
   -       -       -       -

A linear separator
------------------

Then one may try a linear model yielding vastly different results, and prompting
other interpretation as of the linear separability of the frozen features:

.. code-block:: console
   :caption: Here is a slightly different setting for the linear case

   $ uv run optim_script.v2.py -s 0.0001 -is on -l -c -m train
   $ uv run optim_script.v2.py -s 0.0001 -is on -l -c -m plot_acc_train

.. image:: assets/wilds/train_wilds_logreg.png

.. note::

   The plot above illustrate the training of the linear model on a smaller number of iterations.
   On 10000 iterations, we would observe the same overfitting behaviour as above in the long run.
   We zoom on the first iterations to show the difference between radii that yield monotonous
   accuracy increases and others that start to decrease after 800 iterations.

The behaviour of the optimisation procedure provides some cues on how to find a suitable radius
:math:`\rho`: we want it to yield a fast and stable optimisation procedure (unlike
:math:`\rho=10^{-2}` above), but still avoiding overfitting (unlike :math:`\rho\le 10^{-4}`
above). So here a radius around ``1e-3`` seems fitting.

References
==========

.. [#MRPH24] Mehta, Roulet, Pillutla, and Harchaoui: **Distributionally Robust Optimization with Bias and Variance Reduction**, *ICLR*, 2024
