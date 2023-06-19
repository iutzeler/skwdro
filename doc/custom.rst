.. title:: Custom WDRO Estimators

.. _user_guide:

==================================================
Training custom WDRO Estimators
==================================================


Overview
-----------------------



Examples
-----------------------

Regressor
~~~~~~~~~

A template regressor is available in the folder ``examples/CustomWDROEstimators/``

The template inherits from the :class:`sklearn.base.RegressorMixin`. 



Custom losses
-----------------------

The package accepts custom losses in its optimizers provided that are compliant with pytorch's automatic differentiation and follow some basic formatting.

>>> class MyLoss(Loss):
>>>    def __init__(
>>>            self,
>>>            sampler: Optional[LabeledSampler]=None,
>>>            *,
>>>            d: int=0,
>>>            fit_intercept: bool=False) -> None:
>>>        
>>>        super(MyLoss, self).__init__(sampler)
>>>        assert d > 0, "Please provide a valid data dimension d>0"
>>>
>>>        self.d = d
>>>        self.fit_intercept = fit_intercept
>>>
>>>        # Internal structure
>>>        self.linear = nn.Linear(d, 1, bias=fit_intercept) # [CUSTOMIZE] Dummy linear regression
>>>
>>>
>>>    def value(self, xi: pt.Tensor, xi_labels: pt.Tensor):
>>>
>>>        # Loss value
>>>        prediction = self.linear(xi)                # [CUSTOMIZE] Dummy linear regression
>>>        error = nn.MSELoss(reduction='none')        # [CUSTOMIZE] Dummy linear regression
>>>        loss_value = error( prediction, xi_labels)  # [CUSTOMIZE] Dummy linear regression
>>>
>>>        return  loss_value
>>>
>>>    @classmethod
>>>    def default_sampler(cls, xi, xi_labels, epsilon):
>>>        return ClassificationNormalNormalSampler(xi, xi_labels, sigma=epsilon, l_sigma=epsilon)
>>>
>>>    @property
>>>    def theta(self) -> pt.Tensor:
>>>        return self.linear.weight       # [CUSTOMIZE] Optimized parameters
>>>
>>>    @property
>>>    def intercept(self) -> pt.Tensor:
>>>        return self.linear.bias         # [CUSTOMIZE] Intercept



Custom costs
-----------------------

