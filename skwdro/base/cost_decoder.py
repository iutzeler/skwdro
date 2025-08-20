from typing import Optional
from dataclasses import dataclass


from .costs_torch import (
    Cost as TorchCost,
    NormCost as ptNormCost,
    NormLabelCost as ptNormLabelCost
)

DEFAULT_KAPPA: float = 1e8


def cost_from_str(code: str) -> TorchCost:
    r"""Converts a string representation of a cost into a TorchCost object representing the mathematical object bellow:

    .. math::

        c(\xi, \zeta):=\|\zeta-\xi\|_k^p

    This function takes a string ``code`` that represents the cost and parses it to create a specific type of :py:class:`TorchCost`. The string is expected to be in a particular format, which includes several components separated by hyphens (`-`). These components are:

    1. A single character indicating the engine type (always 't' for this context).
    2. An identifier ("NC" for input-norm cost without additional parameters, or "NLC" for norm cost with label switch penalty with an additional parameter `kappa`).
    3. The power value :math:`p`, as a float.
    4. The ground distance type :math:`k` asociated to the k-norm :math:`\|\cdot\|_k`, as a float.
    5. (Optional) A float value representing `kappa`, which is only present if the identifier is "NLC".

    The function will raise a :class:`ValueError` if the cost code is invalid or if the engine type is not recognized.

    Parameters
    ----------

    :param code: The string representation of the cost, in the format specified above.
    :type code: str
    :raises ValueError: If the cost code is invalid or the engine type is not 't'.
    :return: An instance of :py:class:`TorchCost` based on the parsed components.
    :rtype: TorchCost
    """
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
    r"""Parses a cost string into a :py:class:`TorchCost` object.
    Returns by default (i.e. for ``None`` input) a ``2-2`` norm cost.

    :param code: The cost string to be converted or parsed, defaults to None
    :type code: str, optional
    :param has_labels: Indicates whether the cost string includes labels, defaults to False
    :type has_labels: bool, optional
    :raises ValueError: If the cost code is invalid or if the engine is not recognized.
    :return: A ParsedCost object representing the parsed cost information.
    :rtype: ParsedCost
    """
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
    r"""Converts a ParsedCost object into a TorchCost object.

    Takes a :py:class:`ParsedCost` instance, which contains specific information about the cost type and parameters, and returns an appropriate :py:class:`TorchCost` subclass based on the parsed ID.

    Parameters
    ----------

    :param parsed: An instance of ParsedCost containing the cost information.
    :type parsed: ParsedCost
    :raises ValueError: If the ID in the parsed cost is not recognized as valid (either "NC" or "NLC"), raises a ValueError 
                         with an error message indicating that the cost code is invalid.
    :return: A TorchCost subclass instance corresponding to the parsed cost type and parameters.
    :rtype: TorchCost
    """
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
