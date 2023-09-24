from typing import Optional, Tuple

import torch as pt

from ._misc_dual_interfaces import _SampledDualLoss
from ..utils import diff_opt_tensor, diff_tensor, normalize_just_vects, normalize_maybe_vects, maybe_unsqueeze


class _SampleDisplacer(_SampledDualLoss):
    def get_displacement_direction(
            self,
            xi: pt.Tensor,
            xi_labels: Optional[pt.Tensor],
            threshold: float=1.
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
        diff_xi, diff_xi_l = self.get_optimal_displacement(xi, xi_labels)
        # Assert type of results and get them returned
        assert diff_xi.grad is not None
        if diff_xi_l is not None:
            assert diff_xi_l.grad is not None
            return normalize_just_vects(diff_xi.grad, threshold, dim=-1), normalize_maybe_vects(diff_xi_l.grad, threshold, dim=-1)
        else:
            return normalize_just_vects(diff_xi.grad, threshold, dim=-1), None

    def get_optimal_displacement(
            self,
            xi: pt.Tensor,
            xi_labels: Optional[pt.Tensor]
            ) -> Tuple[pt.Tensor, Optional[pt.Tensor]]:
        r""" Optimal displacement to maximize the adversity of the samples.
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

        # Freeze the parameters before differentiation wrt xi
        self.freeze()

        # "Dual" samples for forward pass, to get the gradients wrt them
        diff_xi = diff_tensor(xi) # (1, m, d)
        diff_xi_l = diff_opt_tensor(xi_labels) # (1, m, d')

        # Forward pass for xi
        out: pt.Tensor = self.primal_loss.value(
                diff_xi,
                diff_xi_l if diff_xi_l is not None else None
            ).squeeze((0, 2)) / self._lam # (m,)

        # Backward pass, at all output loss per sample,
        # i.e. one gradient per xi sample, m total
        out.backward(pt.ones(xi.size(0))) # xi.size = m

        # Unfreeze the parameters to allow training
        self.freeze(rg=True)

        return diff_xi, diff_xi_l

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
        disp, disp_labels = self.get_displacement_direction(
                xi,
                xi_labels
            )
        displaced_xi, displaced_xi_labels = self.cost.solve_max_series_exp(
                xi.unsqueeze(0),
                maybe_unsqueeze(xi_labels, dim=0),
                disp,
                disp_labels
            )
        displaced_zeta, displaced_zeta_labels = self.cost.solve_max_series_exp(zeta, zeta_labels, disp, disp_labels)
        return displaced_xi, displaced_xi_labels, displaced_zeta, displaced_zeta_labels
