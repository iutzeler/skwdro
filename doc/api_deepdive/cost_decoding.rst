skwdro.base.cost_decoder module
-------------------------------

:bdg-secondary:`API`

This module exposes mainly the :py:func:`cost_from_str` function, which translates strings with the following grammar to a cost class inheriting :py:class:`~skwdro.base.costs_torch.Cost`.

.. code-block:: antlr
    :caption: Grammar for the cost-specification strings.
    :linenos:

    // Entry point
    spec: engine DASH type DASH FLOAT DASH FLOAT kappa? ;

    DASH: '-' ;

    FLOAT: .* ; // Python-parseable positive floating point number

    // NC for simple p-powered k-norm cost
    // NLC for same with a penalization of label switches with weight kappa
    type: 'NC' | 'NLC' ;

    kappa: DASH FLOAT ; // Python-parseable positive floating point number


.. automodule:: skwdro.base.losses_torch
   :members:
   :show-inheritance:
   :undoc-members:
   :no-index:
