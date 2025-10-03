skwdro.base.losses_torch module
-------------------------------

:bdg-secondary:`API`

Here you can fin the API for all available losses in the library.
They all subclass the base :py:class:`Loss` class, and so should your custom implementations if you wish to build one from scratch.

In order to make your own loss, you must overload :py:meth:`Loss.value` and probably :py:meth:`Loss.default_sampler`

.. automodule:: skwdro.base.losses_torch
   :members:
   :show-inheritance:
   :undoc-members:
   :no-index:
