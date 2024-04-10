from typing import Optional, Tuple

import torch as pt
from torch._functorch.apis import vmap, grad
from torch._functorch import functional_call

from ._misc_dual_interfaces import _SampledDualLoss
from ..utils import diff_opt_tensor, diff_tensor, normalize_just_vects, normalize_maybe_vects, maybe_unsqueeze


class _SampleDisplacer(_SampledDualLoss):
    r'''
    Interfaces for importance sampling, and gradients computations factored for initial lambda
    computation down the abstraction.
    '''
    def get_optimal_displacement(
            self,
            xi: pt.Tensor,
            xi_labels: Optional[pt.Tensor],
            threshold: float=1e10
            ) -> Tuple[pt.Tensor, Optional[pt.Tensor]]:
        r"""
        Thresholds the displacement of the samples to avoid numerical instabilities.
        See :py:method:`~_SampleDisplacer.get_optimal_displacement`.
        Yields :math:`\frac{\nabla_\xi L(\xi)}{\lambda}` with backprop algorithm.

        Parameters
        ----------
        xi : pt.Tensor
            original samples observed
        xi_labels : Optional[pt.Tensor]
            associated labels, if any

        Returns
        -------
        disps: Tuple[pt.Tensor, Optional[pt.Tensor]]
            displacements to maximize the series expansion

        Shapes
        ------
        xi : (m, d)
        xi_labels : (m, d')
        disps: (1, m, d), (1, m, d')
        """
        grad_xi, grad_xi_l = self.get_displacement_direction(xi, xi_labels)
        # Assert type of results and get them returned
        return normalize_just_vects(grad_xi / self._lam, threshold, dim=-1), normalize_maybe_vects(grad_xi_l / self._lam, threshold, dim=-1)

    def compute_functional_grads(
            self,
            xi: pt.Tensor,
            xi_labels: Optional[pt.Tensor]
        ) -> Tuple[pt.Tensor, Optional[pt.Tensor]]:
        model = self.primal_loss
        thetas = {k: v.detach() for k, v in model.named_parameters()}
        _internal_states = {k: v.detach() for k, v in model.named_buffers()}
        def call_loss(data, target, params, states):
            loss = functional_call.functional_call(
                model, # apply primal loss
                (params, states,), # pass in current params and state buffers
                (data, target,) # give as inputs xi and xi_labels
            )
            return loss.squeeze()
        grad_func = grad(
            call_loss, # diff the loss
            (0,) if xi_labels is None else (0, 1,) # wrt data, and labels if relevant
        )
        per_sample_grad_func = vmap(
            vmap(
                grad_func, # apply gradient to data
                in_dims=(
                    0, # always on zeta-batch dims for data
                    None if xi_labels is None else 0, # if labels, on zeta-batch dims, else share the null
                    None, # share parameters
                    None # share the state
                )
            ),
            in_dims=(
                0, # always on xi-batch dims for data
                None if xi_labels is None else 0, # if labels, on xi-batch dims, else share the null
                None, # share parameters
                None # share the state
            )
        )
        if xi_labels is None:
            return per_sample_grad_func(xi, xi_labels, thetas, _internal_states)[0], None
        else:
            return per_sample_grad_func(xi, xi_labels, thetas, _internal_states)

    def get_displacement_direction(
            self,
            xi: pt.Tensor,
            xi_labels: Optional[pt.Tensor]
        ) -> Tuple[pt.Tensor, Optional[pt.Tensor]]:
        r""" Optimal displacement to maximize the adversity of the samples.
        Yields :math:`\nabla_\xi L(\xi)` with backprop algorithm.

        Parameters
        ----------
        xi : pt.Tensor
            original samples observed
        xi_labels : Optional[pt.Tensor]
            associated labels, if any

        Returns
        -------
        disps: Tuple[pt.Tensor, Optional[pt.Tensor]]
            displacements to maximize the series expansion

        Shapes
        ------
        xi : (m, d)
        xi_labels : (m, d')
        disps: (1, m, d), (1, m, d')
        """

        # Freeze the parameters before differentiation wrt xi
        self.freeze()

        # "Dual" samples for forward pass, to get the gradients wrt them
        # diff_xi = diff_tensor(xi) # (1, m, d)
        # diff_xi_l = diff_opt_tensor(xi_labels) # (1, m, d')

        # Forward pass for xi
        # out: pt.Tensor = self.primal_loss.value(
        #         diff_xi,
        #         diff_xi_l if diff_xi_l is not None else None
        #     ).squeeze((0, 2)) # (m,)

        # Backward pass, at all output loss per sample,
        # i.e. one gradient per xi sample, m total
        # out.backward(pt.ones(xi.size(0))) # xi.size = m
        grad_xi, grad_xi_labels = self.compute_functional_grads(
            xi.unsqueeze(0),
            maybe_unsqueeze(xi_labels)
        )
        assert grad_xi.shape == (1, xi.size(0), xi.size(1))
        if grad_xi_labels is not None:
            assert grad_xi_labels.shape == (1, xi_labels.size(0), xi_labels.size(1))

        # Unfreeze the parameters to allow training
        self.freeze(rg=True)
        return grad_xi, grad_xi_labels

        # assert diff_xi.grad is not None
        # if diff_xi_l is not None:
        #     assert diff_xi_l.grad is not None
        #     return diff_xi.grad, diff_xi_l.grad
        # else:
        #     return diff_xi.grad, None

    def displace_samples(
            self,
            xi: pt.Tensor,
            xi_labels: Optional[pt.Tensor],
            zeta: pt.Tensor,
            zeta_labels: Optional[pt.Tensor]
        ) -> Tuple[
                pt.Tensor,
                Optional[pt.Tensor],
                pt.Tensor,
                Optional[pt.Tensor]
            ]:
        r""" Optimal displacement to maximize the adversity of the samples.
        Yields :math:`\frac{\nabla_\xi L(\xi)}{\lambda}` with backprop algorithm.

        Parameters
        ----------
        xi : pt.Tensor
            original samples observed
        xi_labels : Optional[pt.Tensor]
            associated labels, if any
        zeta : pt.Tensor
            adversarial samples
        zeta_labels : Optional[pt.Tensor]
            associated adversarial labels, if any

        Returns
        -------
        disps: Tuple[pt.Tensor, Optional[pt.Tensor]]
            displacements to maximize the series expansion

        Shapes
        ------
        xi : (m, d)
        xi_labels : (m, d')
        zeta : (n_s, m, d)
        zeta_labels : (n_s, m, d')
        """
        disp, disp_labels = self.get_optimal_displacement(
                xi,
                xi_labels
            )
        if disp.isfinite().logical_not().any() or (disp_labels is not None and disp_labels.isfinite().logical_not().any()):
            # Safeguard against NaNs mainly, as well as divergences
            return xi.unsqueeze(0), maybe_unsqueeze(xi_labels, dim=0), zeta, zeta_labels
        else:
            displaced_xi, displaced_xi_labels = self.cost.solve_max_series_exp(
                    xi.unsqueeze(0),
                    maybe_unsqueeze(xi_labels, dim=0),
                    disp,
                    disp_labels
                )
            displaced_zeta, displaced_zeta_labels = self.cost.solve_max_series_exp(zeta, zeta_labels, disp, disp_labels)
            return displaced_xi, displaced_xi_labels, displaced_zeta, displaced_zeta_labels
