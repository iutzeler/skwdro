from .base_cost import TorchCost
from .normcost import NormCost
from .normlabelcost import NormLabelCost


"""
Alias for :py:class:`TorchCost`.
"""
Cost = TorchCost


__all__ = ["NormCost", "NormLabelCost", "Cost", "TorchCost"]
