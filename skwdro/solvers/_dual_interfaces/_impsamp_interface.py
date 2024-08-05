from typing import Optional, Tuple
# import subprocess as sp

import torch as pt
from torch._functorch.apis import vmap, grad
from torch._functorch import functional_call

from ._misc_dual_interfaces import _SampledDualLoss
from ..utils import (
    normalize_just_vects,
    normalize_maybe_vects,
    maybe_unsqueeze,
    check_tensor_validity,
)


class _SampleDisplacer(_SampledDualLoss):
    r'''
    Interfaces for importance sampling, and gradients computations factored
    for initial lambda computation down the abstraction.
    '''
    def get_optimal_displacement(
            self,
            xi: pt.Tensor,
            xi_labels: Optional[pt.Tensor],
            threshold: float = 1e10
    ) -> Tuple[pt.Tensor, Optional[pt.Tensor]]:
        r"""
        Thresholds the displacement of the samples to avoid numerical
        instabilities.
        See :py:method:`~_SampleDisplacer.get_optimal_displacement`.
        Yields :math:`\frac{\nabla_\xi L(\xi)}{\lambda}` with backprop
        algorithm.

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
        # Compute the inverse scaling in the displacement
        # Warnings:
        # - .item() is important
        # - the inverse of inv_scale must not be computed alone,
        #   it seems that it is crucial it is computed along with
        #   the grads below
        # - the point above takes care of numerical issues,
        #   adding a floor to lambda causes other issues
        inv_scale = self._lam.item()
        # Assert type of results and get them returned
        return (
            normalize_just_vects(
                grad_xi,
                threshold,
                inv_scale,
                dim=-1
            ),
            normalize_maybe_vects(
                grad_xi_l,
                threshold,
                inv_scale,
                dim=-1
            )
        )

    def compute_functional_grads(
        self,
        xi: pt.Tensor,
        xi_labels: Optional[pt.Tensor]
    ) -> Tuple[pt.Tensor, Optional[pt.Tensor]]:
        """
        The new functional API from PyTorch helps us compute the gradient of
        the dual loss with respect to each xi sample. This function outputs
        it via a vectorized routine with the ``vmap`` utility.

        Warning: this function uses a trailing unary dimension ``(1, m, d)``.

        Parameters
        ----------
        xi : pt.Tensor
            original samples observed
        xi_labels : Optional[pt.Tensor]
            associated labels, if any

        Returns
        -------
        grads: Tuple[pt.Tensor, Optional[pt.Tensor]]
            gradients drawn

        Shapes
        ------
        xi : (1, m, d)
        xi_labels : (1, m, d')
        disps: (1, m, d), (1, m, d')
        """
        model = self.primal_loss
        model.eval()  # Deactivate batch-norms & dropouts
        thetas = {k: v.detach() for k, v in model.named_parameters()}
        _internal_states = {k: v.detach() for k, v in model.named_buffers()}

        def call_loss(data, target, params, states):
            loss = functional_call.functional_call(
                # apply primal loss
                model,
                # pass in current params and state buffers
                (params, states,),
                # give as inputs xi and xi_labels
                (data.unsqueeze(0), maybe_unsqueeze(target, dim=0),)
            )
            return loss.squeeze()

        grad_func = grad(
            # diff the loss
            call_loss,
            # wrt data / labels if relevant
            (0,) if xi_labels is None else (0, 1,)
        )
        per_sample_grad_func = vmap(
            vmap(
                # apply gradient to data
                grad_func,
                in_dims=(
                    # always on zeta-batch dims for data
                    0,
                    # if labels on zeta-batch dims, else share the null
                    None if xi_labels is None else 0,
                    # share parameters
                    None,
                    # share the state
                    None
                )
            ),
            in_dims=(
                # always on xi-batch dims for data
                0,
                # if labels, on xi-batch dims, else share the null
                None if xi_labels is None else 0,
                # share parameters
                None,
                # share the state
                None
            )
        )
        psg = per_sample_grad_func(xi, xi_labels, thetas, _internal_states)
        model.train()
        assert isinstance(psg[0], pt.Tensor)
        if xi_labels is None:
            return psg[0], None
        else:
            assert len(psg) == 2
            assert psg[1] is None or isinstance(psg[1], pt.Tensor)
            return psg[0], psg[1]

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
        grad_xi, grad_xi_labels = self.compute_functional_grads(
            xi.unsqueeze(0),
            maybe_unsqueeze(xi_labels)
        )

        # Unfreeze the parameters to allow training
        self.freeze(rg=True)
        return grad_xi, grad_xi_labels

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
        Yields :math:`\frac{\nabla_\xi L(\xi)}{\lambda}` with backprop
        algorithm.

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

        _check: bool = check_tensor_validity(disp)
        _check = _check or (
            (disp_labels is not None) and check_tensor_validity(disp_labels)
        )

        if _check:
            # Safeguard against NaNs mainly, as well as divergences
            # try:
            #     sp.run(["notify-send", "=== ! NAN ALERT ! ====", f"Nans: {disp.isnan().sum()}", "-u", "critcal"], check=False)
            # except FileNotFoundError:
            #     print(f"┌┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈")
            #     print(f"┊SKWDRO: === ! NAN ALERT ! ====\t")
            #     print(f"┊SKWDRO Nans: {disp.isnan().sum()}\t")
            #     print(f"└┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈")
            return (
                xi.unsqueeze(0),
                maybe_unsqueeze(xi_labels, dim=0),
                zeta,
                zeta_labels
            )
        else:
            _solve_for_xi = self.cost.solve_max_series_exp
            displaced_xi, displaced_xi_labels = _solve_for_xi(
                xi.unsqueeze(0),
                maybe_unsqueeze(xi_labels, dim=0),
                disp,
                disp_labels
            )
            displaced_zeta, displaced_zeta_labels = _solve_for_xi(
                zeta, zeta_labels, disp, disp_labels
            )
            return (
                displaced_xi,
                displaced_xi_labels,
                displaced_zeta,
                displaced_zeta_labels
            )
