from typing import Tuple, Optional

import torch as pt
import torch.nn as nn

from skwdro.base.costs_torch.normcost import NormCost
from skwdro.base.costs_torch.normlabelcost import NormLabelCost
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
            epsilon_factor * rho.pow(p - 1), # epsilon ^ (p/q)
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
        has_labels: bool,
        xi_batchinit: pt.Tensor,
        xi_labels_batchinit: Optional[pt.Tensor],
        post_sample: bool=True,
        cost_spec: Optional[str]=None,
        n_samples: int=10,
        seed: int=42,
        *,
        epsilon: Optional[float]=None,
        sigma: Optional[float]=None,
        l2reg: Optional[float]=None,
        adapt: str="prodigy",
        imp_samp: bool=True
        ) -> _DualLoss:
    sampler: BaseSampler
    cost: Cost

    parsed_cost = parse_code_torch(cost_spec, has_labels)
    expert_sigma, expert_epsilon = expert_hyperparams(rho, power_from_parsed_spec(parsed_cost), epsilon, EPSILON_SIGMA_FACTOR, sigma, SIGMA_FACTOR)

    cost = cost_from_parse_torch(parsed_cost)

    if has_labels:
        assert xi_labels_batchinit is not None, "Please provide a starting (mini/full)batch of labels to initialize the samplers"
        sampler = LabeledCostSampler(cost, xi_batchinit, xi_labels_batchinit, expert_sigma, seed)
    else:
        sampler = NoLabelsCostSampler(cost, xi_batchinit, expert_sigma, seed)

    loss = WrappedPrimalLoss(loss_, transform_, sampler, has_labels, l2reg=l2reg)

    kwargs = {
        "rho_0": rho,
        "n_samples": n_samples,
        "epsilon_0": expert_epsilon,
        "adapt": adapt,
        "imp_samp": imp_samp and parsed_cost.can_imp_samp(),
    }
    if post_sample:
        return DualPostSampledLoss(
                loss,
                cost,
                n_iter=(200, 800),
                **kwargs
            )
    else:
        return DualPreSampledLoss(
                loss,
                cost,
                n_iter=(100, 10),
                **kwargs
            )
