from typing import Optional
from dataclasses import dataclass


from .costs_torch import (
    Cost as TorchCost,
    NormCost as ptNormCost,
    NormLabelCost as ptNormLabelCost
)

DEFAULT_KAPPA: float = 1e8


def cost_from_str(code: str) -> TorchCost:
    splitted = code.split('-')
    engine_, id_, power_, type_ = splitted[:4]
    kappa = float(splitted[4]) if len(splitted) >= 5 else DEFAULT_KAPPA
    if engine_ == 't':
        if id_ == "NC":
            return ptNormCost(
                p=float(type_),
                power=float(power_),
                name=code
            )
        elif id_ == "NLC":
            return ptNormLabelCost(
                p=float(type_),
                power=float(power_),
                kappa=kappa,
                name=code
            )
        else:
            raise ValueError("Cost code invalid: " + code)
    else:
        raise ValueError("Cost code invalid: " + code)


@dataclass
class ParsedCost:
    engine: str
    id: str
    power: float
    type: float
    kappa: float

    def can_imp_samp(self):
        return self.power == 2 and self.type == 2

    def __iter__(self):
        yield self.engine
        yield self.id
        yield self.power
        yield self.type
        yield self.kappa


def parse_code_torch(
    code: Optional[str],
    has_labels: bool = False
) -> ParsedCost:
    if code is None:
        return ParsedCost(
            't',
            'NLC' if has_labels else 'NC',
            2.,
            2.,
            DEFAULT_KAPPA
        )
    else:
        splitted = code.split('-')
        if len(splitted) == 4:
            engine_, id_, power_, type_ = splitted
            assert engine_ == 't', "Torch engine not recognized"
            kappa = DEFAULT_KAPPA
        elif len(splitted) == 3:
            id_, power_, type_ = splitted
            engine_ = 't'
            kappa = DEFAULT_KAPPA
        elif len(splitted) == 5:
            engine_, id_, power_, type_, kappa_ = splitted
            kappa = float(kappa_)
        else:
            raise ValueError("Cost code invalid: " + code)
        return ParsedCost(engine_, id_, float(power_), float(type_), kappa)


def cost_from_parse_torch(parsed: ParsedCost) -> TorchCost:
    _, id_, power_, type_, kappa = parsed

    if id_ == "NC":
        return ptNormCost(p=float(type_), power=float(power_))
    elif id_ == "NLC":
        return ptNormLabelCost(
            p=float(type_),
            power=float(power_),
            kappa=float(kappa)
        )
    else:
        raise ValueError("Cost code invalid (ID): " + str(id_))
