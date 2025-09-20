Welcome to skwdro's documentation!
==================================

This project aims at providing easy-to-use Wasserstein Distributionally Robust (WDRO) versions of popular machine learning (and operations research) problems.

.. raw:: html

    <table>
        <tr>
            <td rowspan=3>
                <b> CI </b>
            </td>
            <td>
                Test
            </td>
            <td>
                <a href="https://github.com/iutzeler/skwdro/actions/workflows/doc.yml" alt="Doc tests"><img alt="Workflow Test" src="https://img.shields.io/github/actions/workflow/status/iutzeler/skwdro/test.yml?style=for-the-badge&label=Tests"></a>
            </td>
        </tr>
        <tr>
            <td>
                Style
            </td>
            <td>
                <a href="https://github.com/iutzeler/skwdro/actions/workflows/doc.yml" alt="Doc tests"><img alt="Workflow Style" src="https://img.shields.io/github/actions/workflow/status/iutzeler/skwdro/style.yml?style=for-the-badge&label=Style"></a>
            </td>
        </tr>
        <tr>
            <td>
                Doc
            </td>
            <td>
                <a href="https://github.com/iutzeler/skwdro/actions/workflows/doc.yml" alt="Doc tests"><img alt="Workflow Doc" src="https://img.shields.io/github/actions/workflow/status/iutzeler/skwdro/doc.yml?style=for-the-badge&label=Doc build"></a>
            </td>
        </tr>
        <tr>
            <td>
                <b> Doc </b>
            </td>
            <td>
                Readthedocs
            </td>
            <td>
                <a href="https://skwdro.readthedocs.io/latest/" alt="Read the Docs"><img src="https://img.shields.io/badge/ReadTheDocs-blue?style=for-the-badge&logo=sphinx"></a>
            </td>
        </tr>
        <tr>
            <td rowspan=3>
                <b> Checks </b>
            </td>
            <td>
                Code style
            </td>
            <td>
                <a href="https://github.com/astral-sh/ruff" alt="Ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json&style=for-the-badge"></a>
            </td>
        </tr>
        <tr>
            <td>
                Types
            </td>
            <td>
                <a href="https://github.com/python/mypy" alt="MyPY"><img src="https://img.shields.io/badge/mypy-checked-blue?style=for-the-badge&logo=python"></a>
            </td>
        </tr>
        <tr>
            <td>
                Build
            </td>
            <td>
                <a href="https://github.com/prefix-dev/rattler-build" alt="Rattlebuild-badge"><img src="https://img.shields.io/badge/Built_by-rattle--build-yellow?logo=anaconda&style=for-the-badge&logoColor=black"></a>
            </td>
        </tr>
        <tr>
            <td rowspan=3>
                <b> Install </b>
            </td>
            <td>
                Pip
            </td>
            <td>
                <a href="https://pypi.org/project/skwdro/"><img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/skwdro?style=for-the-badge"></a>
            </td>
        </tr>
        <tr>
            <td>
                Conda
            </td>
            <td>
                <a href="https://anaconda.org/flvincen/skwdro"> <img src="https://anaconda.org/flvincen/skwdro/badges/version.svg" /> </a>
            </td>
        </tr>
        <tr>
            <td>
                Github
            </td>
            <td>
                <a href="https://github.com/iutzeler/skwdro"><img src="https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white"></a>
            </td>
        </tr>
        <tr>
        <td colspan=2>
           <b> Cite </b>
        </td>
        <td>
            <a href="https://arxiv.org/abs/2410.21231"><img src="https://img.shields.io/badge/arXiv-2410.21231-b31b1b.svg?style=for-the-badge&logo=arXiv&logoColor=b31b1b"></a>
        </td>
    </tr>
    </table>


    <div align="center">
      <h1>SkWDRO - Wasserstein Distributionaly Robust Optimization</h1>
      <h4>Model robustification with thin interface</h4>
      <h6><q cite="https://adversarial-ml-tutorial.org/introduction">You can make pigs fly</q>, <a href="https://adversarial-ml-tutorial.org/introduction">[Kolter&Madry, 2018]</a></h6>
    </div>

    <p align="center">
      <a href="https://www.python.org">
        <img alt="Python" src="https://img.shields.io/badge/Python-blue?logo=python&logoColor=yellow&style=for-the-badge">
      </a>
      <a href="https://pytorch.org/">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-purple?logo=PyTorch&style=for-the-badge">
      </a>
      <a href="https://scikit-learn.org">
        <img alt="Scikit Learn" src="https://img.shields.io/badge/ScikitLearn-red?logo=scikit-learn&style=for-the-badge">
      </a>
      <img alt="License" src="https://img.shields.io/badge/License-BSD_3--Clause-blue.svg?style=for-the-badge">
    </p>


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
   :caption: PyTorch interface

   pytorch
   examples/Custom/index
   examples/Study/index

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Scikit interface

   sklearn
   examples/Builtin/index

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Advanced topics

   tutos/costs
   tutos/samplers
   optim
   solvers
   tuning

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API

   API/modules
   api_deepdive/submodules


.. mermaid::

   flowchart TD
      A["Getting started"]:::important
      A -->|"Dive in quickly"| B["User guide"]:::important
      A -->|"Learn theory about internals"| C["What is WDRO?"]:::normal
      C --> D["Why SkWDRO?"]:::important
      A --> D
      D --> B

      B -->|"Implement your own problem"| F["Pytorch interface"]:::important
      F --> G["Pytorch examples"]:::normal

      B -->|"For pre-implemented examples"| H["Sklearn interfaces"]:::normal
      H --> I["Sklearn examples"]:::normal

      G --> J["API"]:::important
      I --> J

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


`Getting started <quick_start.html>`_
-------------------------------------

See the getting-started guide to see how to install the package and get to learn its basic usage.
Then you can take a look at `some of the theory <why_skwdro.html>`__ that goes behind the duality result we use to make Sinkhorn-WDRO tractable.

.. admonition:: SkWDRO magic formula

   If you are performing "Empirical Risk Minimization", and wish to robustify it, we can do it through the following transformation:

   .. math::
      :label: ERMtoSkWDRO

      \frac{1}{N}\sum_{i=1}^N L(\xi_i) \to \min_{\lambda\ge 0}\rho\lambda+\varepsilon\frac{1}{N}\sum_{i=1}^N \log\mathbb{E}_{\zeta\sim\nu_{\xi_i}}\left[\frac{L(\zeta)-\lambda c(\xi_i, \zeta)}{\varepsilon}\right]

   This transformation introduces computational difficulties, so let us handle it for you!

`User Guide <user_guide.html>`_
-------------------------------

Quick hitchhiker's guide to the interfaces available to guide you through the process of robustification in "simple" cases.
 


`In depth guide to the PyToch customization functions <pytorch.html>`_
----------------------------------------------------------------------

Learn more about the way you can robustify your own model with :eq:`ERMtoSkWDRO` and how to specify it to make it compatible with the library.
