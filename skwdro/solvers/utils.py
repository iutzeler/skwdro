from typing import Union, Optional, Tuple
import numpy as np
import torch as pt


Steps = Union[int, Tuple[int, int]]


def detach_tensor(tensor: pt.Tensor) -> np.ndarray:
    out = tensor.detach().cpu().numpy().flatten()
    assert isinstance(out, np.ndarray)
    return out  # float(out) if len(out) == 1 else out


def maybe_detach_tensor(tensor: Optional[pt.Tensor]) -> Optional[np.ndarray]:
    return None if tensor is None else detach_tensor(tensor)


def diff_opt_tensor(
    tensor: Optional[pt.Tensor],
    us_dim: Optional[int] = 0
) -> Optional[pt.Tensor]:
    if tensor is None:
        return None
    else:
        return diff_tensor(tensor, us_dim)


def diff_tensor(tensor: pt.Tensor, us_dim: Optional[int] = 0) -> pt.Tensor:
    if us_dim is not None:
        return tensor.clone().unsqueeze(us_dim).requires_grad_(True)
    else:
        return tensor.clone().requires_grad_(True)


def maybe_unsqueeze(
    tensor: Optional[pt.Tensor],
    dim: int = 0
) -> Optional[pt.Tensor]:
    return None if tensor is None else tensor.unsqueeze(dim)


def normalize_maybe_vects(
    tensor: Optional[pt.Tensor],
    threshold: float = 1.,
    scaling: float = 1.,
    dim: int = 0
) -> Optional[pt.Tensor]:
    return None if tensor is None else normalize_just_vects(
        tensor,
        threshold,
        scaling,
        dim
    )


def normalize_just_vects(
    tensor: pt.Tensor,
    threshold: float = 1.,
    scaling: float = 1.,
    dim: int = 0
) -> pt.Tensor:
    n = pt.linalg.norm(tensor, dim=dim, keepdims=True)
    assert isinstance(n, pt.Tensor)
    return tensor / n * pt.min(pt.tensor(threshold), n) / scaling


class NoneGradError(ValueError):
    pass


def maybe_flatten_grad_else_raise(tensor: pt.Tensor) -> pt.Tensor:
    if tensor.grad is None:
        raise NoneGradError(tensor.shape)
    else:
        return tensor.grad.flatten()


def check_tensor_validity(tensor: pt.Tensor) -> bool:
    return bool(tensor.isfinite().logical_not().any().item())


def interpret_steps_struct(
    steps_spec: Steps,
    default_split: float = .3
) -> Tuple[int, int]:
    if isinstance(steps_spec, int):
        assert 0 <= default_split <= 1.
        pretrain_iters = int(steps_spec * default_split)
        return pretrain_iters, steps_spec
    else:  # already tuple
        return steps_spec
