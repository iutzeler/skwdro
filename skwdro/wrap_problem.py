from typing import Tuple, Optional

import torch as pt
import torch.nn as nn

from skwdro.base.costs_torch import Cost
from skwdro.base.cost_decoder import ParsedCost, cost_from_parse_torch, parse_code_torch
from skwdro.base.losses_torch import WrappedPrimalLoss
from skwdro.base.samplers.torch.base_samplers import BaseSampler
from skwdro.base.samplers.torch.cost_samplers import LabeledCostSampler, NoLabelsCostSampler
from skwdro.solvers._dual_interfaces import _DualLoss
from skwdro.solvers.oracle_torch import DualPostSampledLoss, DualPreSampledLoss

SIGMA_FACTOR: float = .5
EPSILON_SIGMA_FACTOR: float = 1e-2
DEFAULT_COST_SPEC: Tuple[float, float] = (2, 2)


def expert_hyperparams(
    rho: pt.Tensor,
    p: float,
    epsilon: Optional[float],
    epsilon_sigma_factor: float,
    sigma: Optional[float],
    sigma_factor: float,
) -> Tuple[pt.Tensor, pt.Tensor]:
    r"""
    Tuning of the hyperparameters for the dual loss.

    Parameters
    ----------
    rho: Tensor, shape (n_samples,)
        Wasserstein radius
    p: float
        power of norm
    epsilon: float
        Epsilon if hard coded, ``None`` to let the algo find it.
    epsilon_sigma_factor: float
        Estimated ratio :math:`\frac{\epsilon}{\sigma}`
    sigma: float
        Sigma if hard coded, ``None`` to let the algo find it.
    sigma_factor: float
        Estimated ratio :math:`\frac{\sigma}{\rho}`
    """
    expert_sigma: pt.Tensor
    expert_epsilon: pt.Tensor

    # Sigma init
    if sigma is None:
        if rho > 0.:
            expert_sigma = rho * sigma_factor
        else:
            expert_sigma = pt.tensor(sigma_factor)
    else:
        expert_sigma = pt.tensor(sigma)

    # Epsilon init
    if epsilon is None:
        epsilon_factor = epsilon_sigma_factor * sigma_factor**p
        expert_epsilon = pt.max(
            epsilon_factor * rho.pow(p - 1),  # epsilon ^ (p/q)
            pt.tensor(1e-7)
        )
    else:
        expert_epsilon = pt.tensor(epsilon)

    return expert_sigma, expert_epsilon


def power_from_parsed_spec(parsed_spec: Optional[ParsedCost]) -> float:
    if parsed_spec is None:
        return 2.
    else:
        return parsed_spec.power


def dualize_primal_loss(
    loss_: nn.Module,
    transform_: Optional[nn.Module],
    rho: pt.Tensor,
    xi_batchinit: pt.Tensor,
    xi_labels_batchinit: Optional[pt.Tensor],
    post_sample: bool = True,
    cost_spec: Optional[str] = None,
    n_samples: int = 10,
    seed: int = 42,
    *,
    epsilon: Optional[float] = None,
    sigma: Optional[float] = None,
    l2reg: Optional[float] = None,
    adapt: Optional[str] = "prodigy",
    imp_samp: bool = True
) -> _DualLoss:
    r"""
    Provide the wrapped version of the primal loss.

    Parameters
    ----------
    loss_: nn.Module
        the primal loss
    transform_: nn.Module
        the transformation to apply to the data before feeding it to the loss
    rho: Tensor, shape (n_samples,)
        Wasserstein radius
    xi_batchinit: Tensor, shape (n_samples, n_features)
        Data points to initialize the samplers and :math:`\lambda_0`
    xi_labels_batchinit: Optional[Tensor], shape (n_samples, n_features)
        Labels to initialize the samplers and :math:`\lambda_0`
    post_sample: bool
        whether to use a post-sampled dual loss
    cost_spec: str|None
        the cost specification in the format ``(k, p)`` for a sample k-norm
        and p-power. ``None`` to use the default ``(2, 2)``.
    n_samples: int
        number of :math:`\zeta` samples to draw before the gradient
        descent begins (can be changed if needed between inferences)
    seed: int
        the seed for the samplers
    epsilon: float|None
        Epsilon if hard coded, ``None`` to let the algo find it.
    sigma: float|None
        Sigma if hard coded, ``None`` to let the algo find it.
    l2reg: float|None
        L2 regularization if needed
    adapt: str|None
        the adaptative step to use between `"prodigy"` and `"mechanic"`.
    imp_samp: bool
        whether to use importance sampling
        (will work only for ``(2, 2)`` costs).
    """
    sampler: BaseSampler
    cost: Cost

    has_labels = xi_labels_batchinit is not None
    if has_labels:
        assert isinstance(xi_labels_batchinit, pt.Tensor), ' '.join([
            "Please provide a starting",
            "(mini/full)batch of labels",
            "to initialize the samplers"
        ])

    parsed_cost = parse_code_torch(cost_spec, has_labels)
    expert_sigma, expert_epsilon = expert_hyperparams(
        rho,
        power_from_parsed_spec(parsed_cost),
        epsilon,
        EPSILON_SIGMA_FACTOR,
        sigma,
        SIGMA_FACTOR
    )
    expert_sigma = expert_sigma.to(xi_batchinit)
    expert_epsilon = expert_epsilon.to(xi_batchinit)

    cost = cost_from_parse_torch(parsed_cost)

    if has_labels:
        assert xi_labels_batchinit is not None
        sampler = LabeledCostSampler(
            cost,
            xi_batchinit,
            xi_labels_batchinit,
            expert_sigma,
            seed
        )
    else:
        sampler = NoLabelsCostSampler(cost, xi_batchinit, expert_sigma, seed)

    loss = WrappedPrimalLoss(
        loss_, transform_, sampler, has_labels, l2reg=l2reg
    )

    # kwargs = {
    #     "rho_0": rho,
    #     "n_samples": n_samples,
    #     "epsilon_0": expert_epsilon,
    #     "adapt": adapt,
    #     "imp_samp": imp_samp and parsed_cost.can_imp_samp(),
    # }
    loss_constructor = (
        DualPostSampledLoss if post_sample
        else DualPreSampledLoss
    )
    return loss_constructor(
        loss,
        cost,
        n_iter=((200, 2800) if post_sample else (100, 10)),
        rho_0=rho,
        n_samples=n_samples,
        epsilon_0=expert_epsilon,
        adapt=adapt,
        imp_samp=(imp_samp and parsed_cost.can_imp_samp())
    )
    # if post_sample:
    #     return DualPostSampledLoss(
    #         loss,
    #         cost,
    #         n_iter=(200, 2800),
    #         **kwargs
    #     )
    # else:
    #     return DualPreSampledLoss(
    #         loss,
    #         cost,
    #         n_iter=(100, 10),
    #         **kwargs
    #     )
