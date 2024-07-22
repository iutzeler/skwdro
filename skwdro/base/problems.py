from skwdro.solvers.oracle_torch import _DualLoss as LossTorch
from typing import List, Optional
import warnings
import numpy as np


def deprecated(message):
    def deprecated_decorator(func):
        def deprecated_func(*args, **kwargs):
            warnings.warn(
                "{} is a deprecated function. {}".format(
                    func.__name__,
                    message
                ),
                category=DeprecationWarning,
                stacklevel=2
            )
            warnings.simplefilter('default', DeprecationWarning)
            return func(*args, **kwargs)
        return deprecated_func
    return deprecated_decorator


Bounds = Optional[List[float]]
LossType = LossTorch


class Distribution:
    empirical: bool
    with_labels: bool

    def __init__(self, m: int, name: str) -> None:
        self.m = m
        self.name = name
        self._samples: Optional[np.ndarray] = None
        self._samples_x: Optional[np.ndarray] = None
        self._samples_y: Optional[np.ndarray] = None

    @property
    def samples(self):
        if self.with_labels:
            raise AttributeError()
        else:
            return self._samples

    @samples.setter
    def samples(self, data):
        if isinstance(data, np.ndarray):
            self._samples = data
        else:
            raise TypeError()

    @property
    def samples_x(self):
        if self.with_labels:
            return self._samples_x
        else:
            raise AttributeError()

    @samples_x.setter
    def samples_x(self, data):
        if isinstance(data, np.ndarray):
            self._samples_x = data
        else:
            raise TypeError()

    @property
    def samples_y(self):
        if self.with_labels:
            return self._samples_y
        else:
            raise AttributeError()

    @samples_y.setter
    def samples_y(self, labels):
        if isinstance(labels, np.ndarray):
            self._samples_y = labels
        else:
            raise TypeError()


class EmpiricalDistributionWithoutLabels(Distribution):
    """ Empirical Probability distribution """

    empirical = True
    with_labels = False

    def __init__(
        self,
        m: int,
        samples: np.ndarray,
        name="Empirical distribution"
    ):
        super(EmpiricalDistributionWithoutLabels, self).__init__(m, name)
        self._samples = samples


class EmpiricalDistributionWithLabels(Distribution):
    """ Empirical Probability distribution with Labels """

    empirical = True
    with_labels = True

    def __init__(
        self,
        m: int,
        samples_x: np.ndarray,
        samples_y: np.ndarray,
        name="Empirical distribution"
    ):
        super(EmpiricalDistributionWithLabels, self).__init__(m, name)
        self._samples_x = samples_x.copy('K')
        self._samples_y = samples_y.copy('K')
