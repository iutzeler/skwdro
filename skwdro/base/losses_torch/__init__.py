from .logistic import LogisticLoss
from .newsvendor import NewsVendorLoss_torch
from .quadratic import QuadraticLoss
from .weber import SimpleWeber
from .base_loss import Loss
from .wrapper import WrappedPrimalLoss

from . import base_loss

NewsVendorLoss = NewsVendorLoss_torch

__all__ = ["LogisticLoss", "NewsVendorLoss", "NewsVendorLoss_torch",
           "QuadraticLoss", "SimpleWeber", "Loss", "base_loss", "WrappedPrimalLoss"]
