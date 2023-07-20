Post-sampling
^^^^^^^^^^^^^
=======
=======

####################
Losses
####################

The following solvers are implemented in skwdro.

.. currentmodule:: skwdro

Operations Research
===================

Portfolio Selection
-------------------

Two losses are implemented in PyTorch: RiskPortfolioLoss_torch et MeanRisk_torch.

The general loss function is given by

.. math::

   \underset{\theta \in \Theta}{\inf} \mathbb{E}^{\mathbb{P}}(- \langle \theta, \xi \rangle) + \eta \mathbb{P}\text{-CVaR}_{\alpha}(- \langle \theta, \xi \rangle)


RiskPortfolioLoss_torch thus implements the first term of the loss function
which is purely based on the scalar product :math:`- \langle \theta, \xi \rangle`
representing the risks associated with return on investment. The idea of implementing this term
in a separate class stems from the fact that there are several functions which exist
to model these financial risks. Thus, the MeanRisk_torch
class implementing the function to be minimized is defined by composition with an instance of the
RiskPortfolioLoss_torch class as an attribute to define the first term.
This maintains the overall structure of the general class
since the definition of a new risk modeling function
requires the creation of its own class,
and the rest of the loss function is defined by composition with an instance of this class.
