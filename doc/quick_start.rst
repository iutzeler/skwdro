#######################
Quick Start with skwdro
#######################

Here is a setup guide to the ``SkWDRO`` library, in order for readers to follow along the multiple tutorials in this documentations with a working environment.

In order to test the installation, we will proceed with a trivial example of robust logistic regression.

.. code-block:: python
   :emphasize-lines: 3, 14
   :caption: Training snippet: copy-paste it in your python interactive session

   import numpy as np
   from sklearn.datasets import make_blobs
   from skwdro.linear_models import LogisticRegression

   SEED = 666

   # Train a simple classifier with a small robustness radius.
   def train():
      X, y, *maybe_centers = make_blobs(return_centers=True, random_state=SEED)
      # Unpack safely for testing phase
      assert len(maybe_centers) == 1
      centers, = maybe_centers

      robust_estimator = LogisticRegression(1e-2)
      robust_estimator = robust_estimator.fit(X, y)
      return robust_estimator, centers


.. code-block:: python
   :caption: Testing snippet: copy-paste it in your python interactive session

   # Test the model in a simple case without weird distribution shifts: just
   # adding some noise to inputs.
   def test(estimator, centers):
       centers[0] += np.random.randn(2)
       centers[1] += np.random.randn(2)
       X, y, *_ = make_blobs(centers=centers, random_state=SEED)
       return estimator.score(X, y)


Setting up the package
======================

To start imediately and play with the various interfaces of ``SkWDRO``, you can install it via the usual python pipeline with ``pip``:

.. code-block:: bash
   :caption: Installation

   $ pip install skwdro

Then immediately launch a python console with the interpreter linked to your ``pip`` installer, and test the code snippet above as follows:

.. code-block:: python

   >>> print(test(*train()))
   0.9

Now we will present some advices for more careful installation of ``SkWDRO`` to integrate it in a bigger workflow, as a dependency of your project.

Using ``.venv``\ s
------------------

In order to work in teams on projects, and for reproducibility, it is advised to put your python packages in a precise list of versions.
This can be specified lazily either through a ``requirements.txt`` file (most common historically), or through a ``pyproject.toml`` file.

A good way on your own machine to orchestrate the local installation of these working dependencies is to store them all in one place.
For python projects, this is called a virtual environment, and is implemented in ``venv`` (standard library).

To create one for your own project, the procedure would look as follows **for a bash shell** (you may need to install your distribution's version of ``python-venv``):

.. code-block:: bash
   :caption: Setup and activate your own venv, with the name you like!
   :emphasize-text: .my_robust_project_venv

   $ python -m venv .my_robust_project_venv

   $ source .my_robust_project_venv/bin/activate

Then the activation of the environment will always be done via the source command above.
It **should** (but might not) put aliases on important commands (add them yourself if needed, via a bash script that you can ``source`` as well).

.. code-block:: bash
   :caption: Don't forget to check you aliases (or set them up to make your life easier)
   :emphasize-text: .my_robust_project_venv

   $ which python
   .../.my_robust_project_venv/bin/python

   $ which pip
   .../.my_robust_project_venv/bin/pip

Now you should be able to test the codes snippets:

.. code-block:: bash
   :caption: Installation

   $ pip install skwdro

.. code-block:: python

   >>> print(test(*train()))
   0.9


Using ``uv``
------------

`uv <https://docs.astral.sh/uv/pip/environments/#using-a-virtual-environment>`_ is a recent tool gaining traction in the python developement community, that aims at simplifying the process above.

.. code-block:: bash
   :caption: Setup your own venv

   $ uv venv
   $ uv pip install skwdro

Only two simple commands, that will create a `.venv` directory and store everything needed inside, as well as making your ``pyproject.toml`` file and update it automatically.

.. code-block:: python
   :caption: That's it.

   >>> print(test(*train()))
   0.9


Using ``conda``/``mamba``/etc
-----------------------------

We propose a conda distribution of this package, but the developement team does not guarentee that it will be as up-to-date as the Pypy version.

You can find it at the `anaconda.org repository <https://anaconda.org/flvincen/skwdro>`__.

.. code-block:: bash
   :emphasize-text: conda

   $ conda install flvincen::skwdro

Then you can test it with the interpreter that your conda distribution uses.

.. code-block:: python
   :caption: That is still simpler than using .venv... Or is it?

   >>> print(test(*train()))
   0.9


Developing SkWDRO with the online repository
============================================

If you wish to contribute to the project, you are welcome to!
Please follow basic rules described in the `Contributor Covenant <https://github.com/iutzeler/skwdro?tab=coc-ov-file>`_, and fork the repository to make pull requests through the available utilities in GitHub™.

Local installation with pip
---------------------------

If you like to use ``pip`` to do global installation of local projects, its editable mode should work out-of-the-box if all dependencies in ``pyproject.toml`` are satisfied.

.. code-block:: bash
   :caption: Editable installation

   $ pip install -e .

Using the project as intended: ``hatch``
----------------------------------------

The ``hatch`` utility is used in this project to maintain a sound developement environment without relying on ``pip``.
We made a ``Makefile`` catered to the use of this precise tool.
You can use it to instantiate a new shell instance running as a thin layer, similarly to a ``venv``.

.. code-block:: bash
   :caption: New shell instance

   $ make shell

Then, you can escape this shell instance with ``Ctrl-D`` (or equivalent).
A full test suite is available to verify non-regression of the codebase.

.. code-block:: bash
   :caption: Lauch the testsuite

   $ make test
   $ make test_gen
   $ make test_sk
   $ make test_misc

You can also recompile the documentation locally.

.. code-block:: bash
   :caption: Look up the documentation locally

   $ make shell
   $ cd doc/
   $ make html

Using ``uv``
------------

This option is untested and probably not stable, but ``uv`` may be used to run the test suite and the code with ``uv run``.

.. warning:: the project must already be set up with ``uv`` in the first place, see some pointers above.

.. code-block::
   :caption: Example: run one of the available files

   $ uv run python examples/builtin_models/linear_regression.py

Next
====

.. card-carousel:: 2

   .. card:: User guide
      :link: user_guide.html

      Learn about the most basic usecases of the library.

   .. card:: What is WDRO?
      :link: wdro.html

      Gentle introduction to the world of Distributionally Robust Optimization, and motivations for its Wasserstein version.

   .. card:: PyTorch part of the library
      :link: pytorch.html

      Tutorial on how to robustify your model easily with the pytorch wrappers.

   .. card:: API
      :link: api_deepdive/submodules.html

      More details about the exposed API.
