from typing import Union

from .costs import Cost, NormCost as npNormCost, NormLabelCost as npNormLabelCost
from .costs_torch import Cost as TorchCost, NormCost as ptNormCost, NormLabelCost as ptNormLabelCost

def cost_from_str(code: str) -> Union[Cost, TorchCost]:
    engine_, id_, power_, type_ = code.split('-')
    if engine_ == 't':
        if id_ == "NC":
            return ptNormCost(p=float(type_), power=float(power_), name=code)
        elif id_ == "NLC":
            return ptNormLabelCost(p=float(type_), power=float(power_), kappa=1e8, name=code)
        else: raise ValueError("Cost code invalid: "+code)
    elif engine_ == 'n':
        if id_ == "NC":
            return npNormCost(p=float(type_), power=float(power_), name=code)
        elif id_ == "NLC":
            return npNormLabelCost(p=float(type_), power=float(power_), kappa=1e8, name=code)
        else: raise ValueError("Cost code invalid: "+code)
    else: raise ValueError("Cost code invalid: "+code)
