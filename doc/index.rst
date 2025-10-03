.. raw:: html

    <div align="center">
      <h1>SkWDRO - Tractable Wasserstein Distributionally Robust Optimization</h1>
      <a href="https://github.com/iutzeler/skwdro"><img src="https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white"></a>
    </div>

    <p style="display:inline"><nobr>
        <table>
            <td><a href="https://github.com/iutzeler/skwdro/actions/workflows/doc.yml" alt="Doc tests"><img alt="Workflow Test" src="https://img.shields.io/github/actions/workflow/status/iutzeler/skwdro/test.yml?style=for-the-badge&label=Tests"></a></td>
            <td><a href="https://github.com/iutzeler/skwdro/actions/workflows/doc.yml" alt="Doc tests"><img alt="Workflow Style" src="https://img.shields.io/github/actions/workflow/status/iutzeler/skwdro/style.yml?style=for-the-badge&label=Style"></a></td>
            <td><a href="https://github.com/iutzeler/skwdro/actions/workflows/doc.yml" alt="Doc tests"><img alt="Workflow Doc" src="https://img.shields.io/github/actions/workflow/status/iutzeler/skwdro/doc.yml?style=for-the-badge&label=Doc build"></a></td>
        </table>
        <table>
            <td><a href="https://pypi.org/project/skwdro/"><img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/skwdro?style=for-the-badge"></a></td>
            <td><a href="https://anaconda.org/flvincen/skwdro"> <img src="https://anaconda.org/flvincen/skwdro/badges/version.svg" /></a></td>
            <td><a href="https://arxiv.org/abs/2410.21231"><img src="https://img.shields.io/badge/arXiv-2410.21231-b31b1b.svg?style=for-the-badge&logo=arXiv&logoColor=b31b1b"></a></td>
        </table>
    </nobr><p>

Welcome to SkWDRO's doc
=======================

``skwdro`` is a Python package that offers **WDRO versions** for a large range of estimators, either by extending **scikit-learn estimators** or by providing a wrapper for **pytorch modules**.

Here is a quick reading order that we advise:

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Bird's eye view

   Welcome to the docs <self>
   Getting started <quick_start>
   Quick tour of WDRO <wdro>
   Sinkhorn regularisation of WDRO: SkWDRO <why_skwdro>
   User guide <user_guide>


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: PyTorch part of the library

   Introduction to the PyTorch interface <pytorch>
   Some torch examples for practice <examples/Custom/index>
   examples/Study/index

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Scikit part of the library

   How to use the available scikit-learn estimators <sklearn>
   Some visual illustrations for scikit-learn estimators <examples/Builtin/index>

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Advanced topics

   Samplers tutorial <tutos/samplers>
   Cost functionals tutorial <tutos/costs>
   About the tuning of the uncertainty radius <tuning>
   More on specific solvers available through scikit-learn estimators <solvers>
..   #More about optimizers <optim>

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API

   API/modules
   api_deepdive/submodules


.. mermaid::
   :align: center

   flowchart TD
      A(((Getting started))):::important
      A -->|"Dive in quickly"| B["User guide"]:::important
      A -->|"If you are a bit new to DRO"| C["What is WDRO?"]:::normal
      C --> D["Why SkWDRO?"]:::important
      D --> B

      B -->|"Implement your own problem"| F["Pytorch interface"]:::important
      F --> G["Pytorch examples"]:::normal

      B -->|"For pre-implemented examples"| H["Sklearn interfaces"]:::normal
      H --> I["Sklearn examples"]:::normal

      G --> J["API"]:::important
      I --> J

      G -->|"In depth explanations"| K>"Samplers"]
      K --> L>"Cost functionals"]

      %% Styles
      classDef important color:#EE8888,fill:#f9f2d0,stroke:#e6b800,stroke-width:2px,color:black,rx:10px,ry:10px;
      classDef normal color:#888888,fill:#f2f2f2,stroke:#999999,stroke-width:1.5px,color:black,stroke-dasharray:3 3,rx:20px,ry:20px;

      %% Hyperlinks
      click A "quick_start.html"
      click B "user_guide.html"
      click C "wdro.html"
      click D "why_skwdro.html"
      click F "pytorch.html"
      click G "examples/Custom/index.html"
      click H "sklearn.html"
      click I "examples/Builtin/index.html"
      click J "api_deepdive/submodules.html"
      click K "tutos/samplers.html"
      click L "tutos/costs.html"


`Getting started <quick_start.html>`_
-------------------------------------

.. code-block:: bash
   :caption: Install it now!

   $ pip3 install skwdro


See the getting-started guide to see how to install the package and get to learn its basic usage.
Then you can take a look at `some of the theory <why_skwdro.html>`__ that goes behind the duality result we use to make Sinkhorn-WDRO tractable.

.. admonition:: SkWDRO main formula

   If you are performing "Empirical Risk Minimization", and wish to robustify it, we can do it through the following transformation:

   .. math::

      \frac{1}{N}\sum_{i=1}^N L(\xi_i) ~ \longmapsto ~ \min_{\lambda\ge 0}\rho\lambda+\varepsilon\frac{1}{N}\sum_{i=1}^N \log\mathbb{E}_{\zeta\sim\nu_{\xi_i}}\left[\frac{L(\zeta)-\lambda c(\xi_i, \zeta)}{\varepsilon}\right]

   This transformation introduces computational difficulties, so let us handle it for you!

`User Guide <user_guide.html>`_
-------------------------------

Quick hitchhiker's guide to the interfaces available to guide you through the process of robustification in "simple" cases.
 


`In depth guide to the PyTorch customization functions <pytorch.html>`_
-----------------------------------------------------------------------

Learn more about the way you can robustify your own model with ``SkWDRO`` and how to specify it to make it compatible with the library.

Next
----

.. card-carousel:: 2

   .. card:: Getting started
      :link: quick_start.html

      Install everything needed.

   .. card:: User guide
      :link: user_guide.html

      Learn about the most basic usecases of the library.

   .. card:: What is WDRO?
      :link: why_skwdro.html

      Gentle introduction to the world of Distributionally Robust Optimization, and motivations for its Wasserstein version.

   .. card:: API
      :link: api_deepdive/submodules.html

      More details about the exposed API.
