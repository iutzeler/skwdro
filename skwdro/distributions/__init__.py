from torch.distributions import *  # noqa: F403, # pyright: ignore
from .dirac_distribution import Dirac

import torch.distributions as dst
transforms = dst.transforms

__all__ = dst.__all__ + ["Dirac", "transforms"]  # pyright: ignore
