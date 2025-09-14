Welcome to skwdro's documentation!
==================================

This project provides easy-to-use Wasserstein Distributionally Robust (WDRO) versions of popular machine learning and operations research problems.

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Documentation

   quick_start
   wdro
   why_skwdro
   user_guide
   sklearn
   pytorch


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Examples

   examples/Builtin/index
   examples/Custom/index
   examples/Study/index

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Advanced topics

   optim
   solvers
   tuning

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API

   API/modules
   api_deepdive/submodules

.. API/skwdro.solvers
.. API/skwdro.linear_models
.. API/skwdro.operations_research
.. API/skwdro.neural_network
.. API/skwdro.distributions
.. API/skwdro.base.samplers.torch
.. API/skwdro.base.losses_torch


.. mermaid::

   flowchart TD
      A["Getting started"]:::important
      A --> B["User guide"]:::important
      A --> C["What is WDRO?"]:::normal
      C --> D["Why SkWDRO?"]:::important
      D --> B

      B --> F["Pytorch interface"]:::important
      F --> G["Pytorch examples"]:::normal

      B --> H["Sklearn interfaces"]:::normal
      H --> I["Sklearn examples"]:::normal

      G --> J["API"]:::important
      I --> J

      %% Styles
      classDef important fill:#f9f2d0,stroke:#e6b800,stroke-width:2px,color:black,rx:20px,ry:20px;
      classDef normal fill:#f2f2f2,stroke:#999999,stroke-width:1.5px,color:black,stroke-dasharray:3 3,rx:20px,ry:20px;

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See the getting-started guide to see how to install the package and get to learn its basic usage

`User Guide <user_guide.html>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Documentation of the package, available models, and their numerical resolution.

`About WDRO <wdro.html>`_
~~~~~~~~~~~~~~~~~~~~~~~~~

Introduction to the WDRO framework and ideas predating this library.

`Why SkWDRO? <why_skwdro.html>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
