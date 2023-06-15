#####################################
Quick Start with the project-template
#####################################

This package serves as a skeleton package aiding at developing compatible
scikit-learn contribution.

Creating your own scikit-learn contribution package
===================================================

1. Download and setup your repository
-------------------------------------

To create your package, you need to clone the ``project-template`` repository::

    $ git clone https://github.com/scikit-learn-contrib/project-template.git

Before to reinitialize your git repository, you need to make the following
changes. Replace all occurrences of ``skwdro`` and ``skwdro``
with the name of you own contribution. You can find all the occurrences using
the following command::

    $ git grep skwdro
    $ git grep skwdro

To remove the history of the template package, you need to remove the `.git`
directory::

    $ cd project-template
    $ rm -rf .git

Then, you need to initialize your new git repository::

    $ git init
    $ git add .
    $ git commit -m 'Initial commit'

Finally, you create an online repository on GitHub and push your code online::

    $ git remote add origin https://github.com/your_remote/your_contribution.git
    $ git push origin master

2. Develop your own scikit-learn estimators
-------------------------------------------

.. _check_estimator: http://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.check_estimator.html#sklearn.utils.estimator_checks.check_estimator
.. _`Contributor's Guide`: http://scikit-learn.org/stable/developers/
.. _PEP8: https://www.python.org/dev/peps/pep-0008/
.. _PEP257: https://www.python.org/dev/peps/pep-0257/
.. _NumPyDoc: https://github.com/numpy/numpydoc
.. _doctests: https://docs.python.org/3/library/doctest.html

You can modify the source files as you want. However, your custom estimators
need to pass the check_estimator_ test to be scikit-learn compatible. You can
refer to the :ref:`User Guide <user_guide>` to help you create a compatible
scikit-learn estimator.

In any case, developers should endeavor to adhere to scikit-learn's
`Contributor's Guide`_ which promotes the use of:

* algorithm-specific unit tests, in addition to ``check_estimator``'s common
  tests;
* PEP8_-compliant code;
* a clearly documented API using NumpyDoc_ and PEP257_-compliant docstrings;
* references to relevant scientific literature in standard citation formats;
* doctests_ to provide succinct usage examples;
* standalone examples to illustrate the usage, model visualisation, and
  benefits/benchmarks of particular algorithms;
* efficient code when the need for optimization is supported by benchmarks.

3. Edit the documentation
-------------------------

.. _Sphinx: http://www.sphinx-doc.org/en/stable/

The documentation is created using Sphinx_. In addition, the examples are
created using ``sphinx-gallery``. Therefore, to generate locally the
documentation, you are required to install the following packages::

    $ pip install sphinx sphinx-gallery sphinx_rtd_theme matplotlib numpydoc pillow

The documentation is made of:

* a home page, ``doc/index.rst``;
* an API documentation, ``doc/api.rst`` in which you should add all public
  objects for which the docstring should be exposed publicly.
* a User Guide documentation, ``doc/user_guide.rst``, containing the narrative
  documentation of your package, to give as much intuition as possible to your
  users.
* examples which are created in the `examples/` folder. Each example
  illustrates some usage of the package. the example file name should start by
  `plot_*.py`.

The documentation is built with the following commands::

    $ cd doc
    $ make html


