from skwdro.solvers.utils import maybe_flatten_grad_else_raise, NoneGradError
from skwdro.solvers._dual_interfaces import _DualLoss as BaseDualLoss
import torch as pt
from typing import Callable, Optional, Union, Tuple


LazyTensor = Callable[[], pt.Tensor]
ValidAndError = Tuple[bool, float]

L_AND_T = {"both", "theta_and_lambda", "t&l", "lambda_and_theta", "l&t"}
L_OR_T = {"one", "theta_or_lambda", "tUl", "lambda_or_theta", "lUt"}
JUST_T = {"theta", "t"}
JUST_L = {"lambda", "l"}


def combine_intersect(a: ValidAndError, b: ValidAndError) -> ValidAndError:
    return a[0] and b[0], a[1] + b[1]


def combine_union(a: ValidAndError, b: ValidAndError) -> ValidAndError:
    return a[0] or b[0], a[1] + b[1]


def wrap(b: bool) -> ValidAndError:
    return b, 0.


class OptCondTorch:
    r""" Callable object representing some optimality conditions

    May track two different expression of the error:
    * the relative error: :math:`\|u_n\| < tol \|u_0\|`
    * the absolute error: :math:`\|u_n\| < tol`

    Those equations are evaluated for three possible metrics :math:`u_n`:

    * the progress in the gradient of the dual loss with respect to the
    parameter of interest
    :math:`\nabla_{\theta ,\lambda} J_{\theta_n}(\zeta_n)`
    * the progress of the parameters themselves
    :math:`(\theta_n-\theta_{n-1} , \lambda_n-\lambda_{n-1})`

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
        if positive, the tolerance (relative or absolute) to allow for the
        parameters, if <=0 ignores it
    tol_lambda: float
        if positive, the tolerance (relative or absolute) to allow for the
        dual parameter, if <=0 ignores it
    monitoring: str
        see the global variables :py:data:`L_OR_T` (for either convergence
        to allow stop), :py:data:`L_AND_T` (for joint convergence to
        allow stop), :py:data:`JUST_L` (for only :math:`\lambda`),
        :py:data:`JUST_T` (for only :math:`\theta`) to have the allowed
        options
    mode: str
        either ``"rel"`` for relative progress or ``"abs"`` for absolute
        progress. Not checked if the metric is the gradient value
    metric:
        either ``"grad"`` for gradient improvement/change over time, or
        ``"param"`` for parameter-space improvement/change over time
    """

    def __init__(
        self,
        order: Union[int, str],
        tol_theta: float = 1e-8,
        tol_lambda: float = 1e-8,
        *,
        monitoring: str = "theta",
        mode: str = "rel",
        metric: str = "grad",
        verbose: bool = False
    ):
        """
        """
        if isinstance(order, str):
            assert order == 'inf'
        self.order: float = float(order)
        assert self.order > 0., ' '.join([
            "Please provide a UINT",
            "order for the parameters",
            "grad norm, or the 'inf' string"
        ])
        self.tol_theta = tol_theta
        self.tol_lambda = tol_lambda
        self.monitoring = monitoring
        self.mode = mode
        self.metric = metric
        self.max_iter: int = 0

        self.verbose = verbose

        self.l_mem: Optional[pt.Tensor] = None
        self.t_mem: Optional[pt.Tensor] = None
        self.l_0: Optional[pt.Tensor] = None
        self.t_0: Optional[pt.Tensor] = None
        self.l_grad_0: Optional[pt.Tensor] = None
        self.t_grad_0: Optional[pt.Tensor] = None
        self.delta_l_1: Optional[pt.Tensor] = None
        self.delta_t_1: Optional[pt.Tensor] = None

    def __call__(self, dual_loss: BaseDualLoss, it_number: int) -> bool:
        r"""
        This object can be called on a dual loss object to check its current
        convergence with respect to its allowed max number of iterations.

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
        self.max_iter = (
            dual_loss.n_iter if isinstance(dual_loss.n_iter, int)
            else dual_loss.n_iter[1]
        )

        def flattheta() -> pt.Tensor:
            return self.get_flat_param(dual_loss.primal_loss)

        def flatgrad() -> pt.Tensor:
            return self.get_flat_grad(dual_loss.primal_loss)

        def lam() -> pt.Tensor:
            return dual_loss.lam

        def lamgrad() -> pt.Tensor:
            return maybe_flatten_grad_else_raise(dual_loss._lam)

        ci = self.check_iter(it_number)
        cp, err = self.check_all_params(lam, lamgrad, flattheta, flatgrad)
        if self.verbose and it_number % 100 == 0:
            print(f"[{it_number = }] {ci = } {cp = } {err = }")
        return ci or cp

    def check_all_params(
            self,
            lam: LazyTensor,
            lamgrad: LazyTensor,
            flattheta: LazyTensor,
            flatgrad: LazyTensor) -> ValidAndError:
        r"""
        Checks the dual and primal parameters for convergence by using
        functional monads on the tensors, see
        :py:func:`~OptCondTorch.check_t`
        and :py:func:`~OptCondTorch.check_l`.

        Parameter
        ---------
        lam: LazyTensor
            the dual multiplier
        lam_grad: LazyTensor
            its scalar gradient
        flat_theta: LazyTensor
            the flattened concatenation of all the optimizeable parameters
            of the primal model
        flat_theta_grad: LazyTensor
            the flattened concatenation of the gradients of those parameters

        Returns
        -------
        cond: bool
            green light to stop algorithm
        """
        if self.monitoring in L_AND_T:
            return combine_intersect(
                self.check_l(lam, lamgrad),
                self.check_t(flattheta, flatgrad)
            )
        elif self.monitoring in L_OR_T:
            return combine_union(
                self.check_l(lam, lamgrad),
                self.check_t(flattheta, flatgrad)
            )
        elif self.monitoring in JUST_L:
            return self.check_l(lam, lamgrad)
        elif self.monitoring in JUST_T:
            return self.check_t(flattheta, flatgrad)
        else:
            raise ValueError("Please provide a valid value for the monitoring")

    def check_t(
        self,
        flat_theta: LazyTensor,
        flat_theta_grad: LazyTensor
    ) -> ValidAndError:
        r"""
        Check the convergence of the theta parameter, either in gradient or in
        parameter value.
        The parameters are ``LazyTensor``s which means that they must be called
        as functions to be evaluated.

        Parameter
        ---------
        flat_theta: LazyTensor
            the flattened concatenation of all the optimizeable parameters of
            the primal model
        flat_theta_grad: LazyTensor
            the flattened concatenation of the gradients of those parameters

        Returns
        -------
        cond: bool
            green light to stop algorithm
        """
        cond = False
        if self.tol_theta <= 0.:
            return cond, 0.
        else:
            if self.metric == "grad":
                mem = self.t_grad_0  # nabla_theta first iteration
                if mem is None:
                    # Compute nabla_theta because we are at nabla theta
                    # right now.
                    # Wait for next iteration to verify convergence.
                    self.t_grad_0 = pt.linalg.norm(
                        flat_theta_grad(), self.order)
                    return wrap(False)
                else:
                    # Compute nabla_theta at current iteration and compare it
                    # to first iterate
                    new = pt.linalg.norm(flat_theta_grad(), self.order)
                    return self.check_metric(new, mem, self.tol_theta)
            elif self.metric == "param":
                mem0 = self.t_0  # first param vector theta_0
                mem1 = self.delta_t_1  # |theta_1 - theta_0|
                if mem0 is None:
                    # Define theta_0
                    self.t_0 = flat_theta()
                    return wrap(False)
                elif mem1 is None:
                    assert mem0 is not None
                    # Define theta_1 (=theta_k)
                    self.t_mem = flat_theta()
                    # |theta_1 - theta_0|
                    self.delta_t_1 = pt.linalg.norm(
                        self.t_mem - mem0, self.order)
                    return wrap(False)
                else:
                    # theta_k
                    mem = self.t_mem
                    assert mem is not None
                    # theta_{k+1}
                    new = flat_theta()
                    # make memory advance one step of k
                    self.t_mem = new
                    # |theta_{k+1} - theta_k|
                    delta = pt.linalg.norm(new - mem, self.order)
                    assert self.delta_t_1 is not None
                    # Check current diff wrt first iterate diff
                    return self.check_metric(
                        delta,
                        self.delta_t_1,
                        self.tol_theta
                    )
            else:
                return wrap(False)

    def check_l(self, lam: LazyTensor, lam_grad: LazyTensor) -> ValidAndError:
        r"""
        Check the convergence of the theta parameter, either in gradient or in
        parameter value.
        The parameters are ``LazyTensor``s which means that they must be called
        as functions to be evaluated

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
        if self.tol_lambda <= 0.:
            return wrap(False)
        else:
            if self.metric == "grad":
                mem = self.l_grad_0
                if mem is None:
                    # nabla_lambda at first iterate
                    self.l_grad_0 = pt.abs(lam_grad())
                    return wrap(False)
                else:
                    # New nabla_lambda
                    new = pt.abs(lam_grad())
                    return self.check_metric(new, mem, self.tol_theta)
            elif self.metric == "param":
                mem0 = self.l_0
                mem1 = self.delta_l_1
                if mem0 is None:
                    # first lambda
                    self.l_0 = lam()
                    return wrap(False)
                elif mem1 is None:
                    assert mem0 is not None
                    # lambda_1 (=lambda_k)
                    self.l_mem = lam()
                    # first diff in lambda
                    self.delta_l_1 = pt.abs(self.l_mem - mem0)
                    return wrap(False)
                else:
                    # lambda_k
                    mem = self.l_mem
                    assert mem is not None
                    # lambda_{k+1}
                    new = lam()
                    # Memory advances one step
                    self.l_mem = new
                    # |lambda_{k+1} - lambda_k|
                    delta = pt.abs(new - mem)
                    assert self.delta_l_1 is not None
                    return self.check_metric(
                        delta,
                        self.delta_l_1,
                        self.tol_lambda
                    )
            else:
                return wrap(False)

    def check_metric(
        self,
        new_obs: pt.Tensor,
        memory: pt.Tensor,
        tol: float
    ) -> ValidAndError:
        r"""
        Helper function to get the tolerance check in both the relative and
        absolute error cases.

        Parameters
        ----------
        new_obs: pt.Tensor
            current step metric
        memory: pt.Tensor
            same metric at last step -- initialized at None, so a check must
            be performed before call to this function
        tol: float
            the positive tolerance rate allowed (same for absolute and
            relative tolerance)

        Returns
        -------
        cond: bool
            green light to stop algorithm
        """
        assert tol > 0.
        if self.mode == "rel":
            return (
                new_obs.sum().item() < tol * memory.sum().item(),
                new_obs.sum().item()
            )
        elif self.mode == "abs":
            return new_obs.sum().item() < tol, new_obs.sum().item()
        else:
            raise ValueError(' '.join([
                "Please set the optcond mode to either 'rel'",
                "for relative tolerance or 'abs'"
            ]))

    def check_iter(self, it_number: int) -> bool:
        r"""
        Checks if the maximum number of iterations has been crossed

        Returns
        -------
        cond: bool
            green light to stop algorithm
        """
        return False if self.max_iter <= 0 else it_number >= self.max_iter

    @classmethod
    def get_flat_param(cls, module: pt.nn.Module) -> pt.Tensor:
        """
        Helper function to get a flat vector containing all the primal
        parameters.
        """
        return pt.concat([*map(
            lambda t: t.flatten(),
            module.parameters()
        )])

    @classmethod
    def get_flat_grad(cls, module: pt.nn.Module) -> pt.Tensor:
        """
        Helper function to get a flat vector containing all the gradients
        of the primal model.
        """
        try:
            return pt.concat([*map(
                maybe_flatten_grad_else_raise,
                module.parameters()
            )])
        except NoneGradError as e:
            raise ValueError(' '.join([
                "The module provided as the primal loss",
                "yields None grads for some of its",
                "parameters, preventing the solver from computing",
                "optimality conditions.",
                f"\nShape of original tensor: {e.args[0]}",
                "Please investigate.",
            ]))
