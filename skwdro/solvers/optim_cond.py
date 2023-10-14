L_AND_T = {"both", "theta_and_lambda", "t&l", "lambda_and_theta", "l&t"}
L_OR_T  = {"one", "theta_or_lambda", "tUl", "lambda_or_theta", "lUt"}
JUST_T  = {"theta", "t"}
JUST_L  = {"lambda", "l"}

# ########### NUMPY code #############
import numpy as np

class OptCond:
    def __init__(self, order, tol_theta: float=1e-4, tol_lambda: float=1e-6, max_iter: int=int(1e5), mode: str="both"):
        assert order > 0
        self.p = order
        self.tol_t = tol_theta
        self.tol_l = tol_lambda
        self.maxiter = max_iter

        self.mode = True
        if mode in L_AND_T:
            self.mode = True
        elif mode in L_OR_T:
            self.mode = False
        else:
            raise ValueError("Please provide a valid string value for parameter 'mode' in OptCond: '" + str(mode) + "' is invalid.")

    def __call__(self, grads, it: int) -> bool:
        grad_theta, grad_lambda = grads
        grad_t_stop = float(np.linalg.norm(grad_theta, ord=self.p)) < self.tol_t and self.tol_t  > 0
        grad_l_stop = float(grad_lambda) < self.tol_l and self.tol_l  > 0
        it_stop = it >= self.maxiter and self.maxiter > 0
        if it_stop: # Warn the user that the optimality conditions may not be verified (loss of accuracy)
            Warning("Maximum number of iterations reached")
        return self.grad_stop(grad_t_stop, grad_l_stop) or it_stop

    def grad_stop(self, grad_t_stop: bool, grad_l_stop: bool) -> bool:
        if self.mode:
            return grad_t_stop and grad_l_stop
        else:
            return grad_t_stop or grad_l_stop

# ########### TORCH code #############
from typing import Callable, Optional, Union

import torch as pt

from skwdro.solvers._dual_interfaces import _DualLoss as BaseDualLoss
from skwdro.solvers.utils import maybe_flatten_grad_else_raise, NoneGradError

LazyTensor = Callable[[], pt.Tensor]

class OptCondTorch:
    r""" Callable object representing some optimality conditions

    May track two different expression of the error:
    * the relative error: :math:`\|u-u_{ref}\| < tol \|u_{ref}\|`
    * the absolute error: :math:`\|u-u_{ref}\| < tol`

    Those equations are evaluated for three possible metrics:
    * the progress in the gradient of the dual loss with respect to the parameter of interest :math:`\nabla_{\theta ,\lambda} J(\zeta)`
    * the progress of the parameters themselves :math:`(\theta , \lambda)`
    * the raw value of the gradient (when it goes to :math:`0` the algorithm must have converged), which is the above formula applied to :math:`u_{ref}=0` in absolute tolerance

    To evaluate the above metrics, one may chose to monitor the convergence in:
    * only :math:`\theta`
    * only :math:`\lambda`
    * both
    * or either

    Parameters
    ----------

    order: int|str
        norm type to use
    tol_theta: float
        if positive, the tolerance (relative or absolute) to allow for the parameters, if <=0 ignores it
    tol_lambda: float
        if positive, the tolerance (relative or absolute) to allow for the dual parameter, if <=0 ignores it
    monitoring: str
        see the global variables :py:data:`L_OR_T` (for either convergence to allow stop), :py:data:`L_AND_T` (for joint convergence to allow stop), :py:data:`JUST_L` (for only :math:`\lambda`), :py:data:`JUST_T` (for only :math:`\theta`) to have the allowed options
    mode: str
        either ``"rel"`` for relative progress or ``"abs"`` for absolute progress. Not checked if the metric is the gradient value
    metric:
        either ``"grad"`` for gradient improvement/change over time, ``"param"`` for parameter-space improvement/change over time, or ``"grad_value"`` for raw absolute parameter-gradient value.
    """
    def __init__(
            self,
            order: Union[int, str],
            tol_theta: float=1e-8,
            tol_lambda: float=1e-8,
            *,
            monitoring: str="theta",
            mode: str="rel",
            metric: str="grad"
            ):
        if isinstance(order, str): assert order == 'inf'
        self.order: float = float(order)
        assert self.order > 0., "Please provide an UINT order for the parameters grad norm, or the 'inf' string"
        self.tol_theta = tol_theta
        self.tol_lambda = tol_lambda
        self.monitoring = monitoring
        self.mode = mode
        self.metric = set(metric.split('&'))
        self.max_iter: int = 0

        self.l_mem: Optional[pt.Tensor] = None
        self.t_mem: Optional[pt.Tensor] = None
        self.l_grad_mem: Optional[pt.Tensor] = None
        self.t_grad_mem: Optional[pt.Tensor] = None

    def __call__(self, dual_loss: BaseDualLoss, it_number: int) -> bool:
        r"""
        This object can be called on a dual loss object to check its current convergence with respect to its allowed max number of iterations.

        Parameters
        ----------
        dual_loss: BaseDualLoss
            the loss to monitor
        it_number: int
            the looping iteration

        Returns
        -------
        cond: bool
            green light to stop algorithm
        """
        self.max_iter: int = dual_loss.n_iter if isinstance(dual_loss.n_iter, int) else dual_loss.n_iter[1]
        flattheta: LazyTensor = lambda: self.get_flat_param(dual_loss.primal_loss)
        flatgrad: LazyTensor = lambda: self.get_flat_grad(dual_loss.primal_loss)
        lam: LazyTensor = lambda: dual_loss.lam
        lamgrad: LazyTensor = lambda: maybe_flatten_grad_else_raise(dual_loss._lam)

        return self.check_iter(it_number) or self.check_all_params(lam, lamgrad, flattheta, flatgrad)

    def check_all_params(
            self,
            lam: LazyTensor,
            lamgrad: LazyTensor,
            flattheta: LazyTensor,
            flatgrad: LazyTensor) -> bool:
        r"""
        Checks the dual and primal parameters for convergence by using functional monads on the tensors,
        see :py:func:`~OptCondTorch.check_t` and :py:func:`~OptCondTorch.check_l`

        Parameter
        ---------
        lam: LazyTensor
            the dual multiplier
        lam_grad: LazyTensor
            its scalar gradient
        flat_theta: LazyTensor
            the flattened concatenation of all the optimizeable parameters of the primal model
        flat_theta_grad: LazyTensor
            the flattened concatenation of the gradients of those parameters

        Returns
        -------
        cond: bool
            green light to stop algorithm
        """
        if self.monitoring in L_AND_T:
            return self.check_l(lam, lamgrad) and self.check_t(flattheta, flatgrad)
        elif self.monitoring in L_OR_T:
            return self.check_l(lam, lamgrad) or self.check_t(flattheta, flatgrad)
        elif self.monitoring in JUST_L:
            return self.check_l(lam, lamgrad)
        elif self.monitoring in JUST_T:
            return self.check_t(flattheta, flatgrad)
        else:
            raise ValueError("Please provide a valid value for the monitoring")

    def check_t(self, flat_theta: LazyTensor, flat_theta_grad: LazyTensor) -> bool:
        r"""
        Check the convergence of the theta parameter, either in gradient or in parameter value.
        The parameters are ``LazyTensor``s which means that they must be called as functions to be evaluated

        Parameter
        ---------
        flat_theta: LazyTensor
            the flattened concatenation of all the optimizeable parameters of the primal model
        flat_theta_grad: LazyTensor
            the flattened concatenation of the gradients of those parameters

        Returns
        -------
        cond: bool
            green light to stop algorithm
        """
        cond = False
        if self.tol_theta <= 0.:
            return cond
        else:
            if "grad" in self.metric:
                mem = self.t_grad_mem
                self.t_grad_mem = new = pt.linalg.norm(flat_theta_grad(), self.order)
                if mem is None:
                    cond = cond or False
                else:
                    cond = cond or self.check_metric(new, mem, self.tol_theta)
            if "param" in self.metric:
                mem = self.t_mem
                self.t_mem = new = pt.linalg.norm(flat_theta(), self.order)
                if mem is None:
                    cond = cond or False
                else:
                    cond = cond or self.check_metric(new, mem, self.tol_theta)
            if "grad_value" in self.metric:
                cond = cond or pt.linalg.norm(flat_theta_grad(), self.order) < self.tol_theta
            return cond

    def check_l(self, lam: LazyTensor, lam_grad: LazyTensor) -> bool:
        r"""
        Check the convergence of the theta parameter, either in gradient or in parameter value.
        The parameters are ``LazyTensor``s which means that they must be called as functions to be evaluated

        Parameter
        ---------
        lam: LazyTensor
            the dual multiplier
        lam_grad: LazyTensor
            its scalar gradient

        Returns
        -------
        cond: bool
            green light to stop algorithm
        """
        if self.tol_theta <= 0.:
            return False
        else:
            if self.metric == "grad":
                mem = self.l_grad_mem
                self.l_grad_mem = new = lam_grad().sum()
                if mem is None:
                    return False
                else:
                    return self.check_metric(new, mem, self.tol_lambda)
            elif self.metric == "param":
                mem = self.l_mem
                self.l_mem = new = lam().sum()
                if mem is None:
                    return False
                else:
                    return self.check_metric(new, mem, self.tol_lambda)
            elif self.metric == "grad_value":
                return lam_grad().item() < self.tol_lambda
            else:
                raise ValueError("Please provide a valid value for the metric monitored")

    def check_metric(self, new_obs: pt.Tensor, memory: pt.Tensor, tol: float) -> bool:
        r"""
        Helper function to get the tolerance check in both the relative and absolute error cases

        Parameters
        ----------
        new_obs: pt.Tensor
            current step metric
        memory: pt.Tensor
            same metric at last step -- initialized at None, so a check must be performed before call to this function
        tol: float
            the positive tolerance rate allowed (same for absolute and relative tolerance)

        Returns
        -------
        cond: bool
            green light to stop algorithm
        """
        assert tol > 0.
        if self.mode == "rel":
            return pt.abs(memory - new_obs).sum().item() < tol * memory.sum().item()
        elif self.mode == "abs":
            return pt.abs(memory - new_obs).sum().item() < tol
        else:
            raise ValueError("Please set the optcond mode to either 'rel' for relative tolerance or 'abs'")

    def check_iter(self, it_number: int) -> bool:
        r"""
        Checks if the maximum number of iterations has been crossed

        Returns
        -------
        cond: bool
            green light to stop algorithm
        """
        if self.max_iter <= 0: return False
        else: return it_number >= self.max_iter

    @classmethod
    def get_flat_param(cls, module: pt.nn.Module) -> pt.Tensor:
        """
        Helper function to get a flat vector containing all the primal parameters
        """
        return pt.concat([*map(
                lambda t: t.flatten(),
                module.parameters()
            )])

    @classmethod
    def get_flat_grad(cls, module: pt.nn.Module) -> pt.Tensor:
        """
        Helper function to get a flat vector containing all the gradients of the primal model
        """
        try:
            return pt.concat([*map(
                    maybe_flatten_grad_else_raise,
                    module.parameters()
                )])
        except NoneGradError as e:
            raise ValueError("The module provided as the primal loss yields None grads for some of its parameters, "
                             "preventing the solver from computing optimality conditions.\n"
                             f"Shape of original tensor: {e.args[0]}"
                             "Please investigate.")
