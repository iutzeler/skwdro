from ._dual_interfaces import _DualLoss as BaseDualLoss
from .oracle_torch import *
from .utils import detach_tensor, diff_opt_tensor, diff_tensor, maybe_unsqueeze, normalize_just_vects, normalize_maybe_vects, maybe_flatten_grad_else_raise, NoneGradError, Steps


__all__ = [
        "BaseDualLoss",
        "DualLoss",
        "DualPreSampledLoss",
        "DualPostSampledLoss",
        "dualize_primal_loss",
        "detach_tensor",
        "diff_opt_tensor",
        "diff_tensor",
        "maybe_unsqueeze",
        "normalize_just_vects",
        "normalize_maybe_vects",
        "maybe_flatten_grad_else_raise",
        "NoneGradError",
        "Steps",
    ]
