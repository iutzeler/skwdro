from typing import Any, Tuple, Optional
from abc import ABC, abstractmethod, abstractproperty

import torch as pt
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as ptag

from skwdro.base.costs import Cost
from skwdro.base.losses_torch import Loss




class _DualLoss(nn.Module, ABC):
    def __init__(self,
                 loss: Loss,
                 cost: Cost,
                 n_samples: int,
                 epsilon_0: pt.Tensor,
                 rho_0: pt.Tensor,
                 n_iter: int,
                 gradient_hypertuning: bool=False
                 ) -> None:
        super(_DualLoss, self).__init__()
        self.loss = loss
        self.cost = cost.value

        self.epsilon = nn.Parameter(epsilon_0, requires_grad=gradient_hypertuning)
        self.rho = nn.Parameter(rho_0, requires_grad=gradient_hypertuning)

        self._lam = nn.Parameter(1e-2 / rho_0)

        self._sampler = loss._sampler
        self.n_samples = n_samples
        self.n_iter = n_iter

    @abstractmethod
    def forward(self, *args):
        raise NotImplementedError()

    def compute_dual(self, xi, xi_labels, zeta, zeta_labels):
        first_term = self.lam**2 * self.rho

        l = self.loss.value(zeta, zeta_labels)
        c = self.cost(
                xi.unsqueeze(0),
                zeta,
                xi_labels.unsqueeze(0) if xi_labels is not None else None,
                zeta_labels
                )
        integrand = l - self.lam**2 * c
        integrand /= self.epsilon

        # Expectation on the zeta samples
        second_term = pt.logsumexp(integrand, 0).mean(dim=0)
        second_term -= pt.log(pt.tensor(zeta.size(0)))
        return first_term + self.epsilon*second_term.squeeze()

    def generate_zetas(self, n_samples: Optional[int]=None):
        if n_samples is None or n_samples <= 0:
            # Default:
            return self.loss.sampler.sample(self.n_samples)
        else:
            return self.loss.sampler.sample(n_samples)

    def default_sampler(self, xi, xi_labels, epsilon):
        return self.loss.default_sampler(xi, xi_labels, epsilon)

    @abstractproperty
    def presample(self):
        raise NotImplementedError()

    @property
    def sampler(self):
        return self.loss.sampler

    @sampler.deleter
    def sampler(self):
        del self.loss.sampler

    @sampler.setter
    def sampler(self, s):
        self.loss.sampler = s

    @property
    def theta(self):
        return self.loss.theta

    @property
    def intercept(self):
        return self.loss.intercept

    @property
    def lam(self):
        return (self._lam)

    @property
    def optimizer(self) -> pt.optim.Optimizer:
        if self._opti is None:
            raise AttributeError("Optimizer for:\n"+self.__str__()+"\nis ill defined (None), please set it beforehand")
        return self._opti

    @optimizer.setter
    def optimizer(self, o: pt.optim.Optimizer):
        self._opti = o

class DualPostSampledLoss(_DualLoss):
    def __init__(self,
                 loss: Loss,
                 cost: Cost,
                 n_samples: int,
                 epsilon_0: pt.Tensor,
                 rho_0: pt.Tensor,
                 n_iter: int=10000,
                 gradient_hypertuning: bool=False
                 ) -> None:
        super(DualPostSampledLoss, self).__init__(loss, cost, n_samples, epsilon_0, rho_0, n_iter, gradient_hypertuning)

        self._opti = pt.optim.AdamW(
                self.parameters(),
                lr=5e-2,
                betas=(.99, .999),
                weight_decay=0.,
                amsgrad=True,
                foreach=True)

    def reset_sampler_mean(self, xi: pt.Tensor, xi_labels: Optional[pt.Tensor]=None):
        self.loss.sampler.reset_mean(xi, xi_labels)

    def forward(self, xi: pt.Tensor, xi_labels: Optional[pt.Tensor]=None, reset_sampler=False):
        if reset_sampler: self.reset_sampler_mean(xi, xi_labels)
        zeta, zeta_labels = self.generate_zetas(self.n_samples)
        return self.compute_dual(xi, xi_labels, zeta, zeta_labels)

    def __str__(self):
        return "Dual loss (sample IN for loop)\n" + 10*"-" + "\n".join(map(str, self.parameters()))

    @property
    def presample(self):
        return False

class DualPreSampledLoss(_DualLoss):
    def __init__(self,
                 loss: Loss,
                 cost: Cost,
                 n_samples: int,
                 epsilon_0: pt.Tensor,
                 rho_0: pt.Tensor,
                 n_iter: int=50,
                 gradient_hypertuning: bool=False
                 ) -> None:
        super(DualPreSampledLoss, self).__init__(loss, cost, n_samples, epsilon_0, rho_0, n_iter, gradient_hypertuning)

        self._opti = pt.optim.LBFGS(
                self.parameters(),
                lr=1.,
                max_iter=1,
                max_eval=10,
                tolerance_grad=1e-4,
                tolerance_change=1e-6,
                history_size=30)

    def forward(self, xi: pt.Tensor, xi_labels: Optional[pt.Tensor]=None, zeta: Optional[pt.Tensor]=None, zeta_labels: Optional[pt.Tensor]=None):
        if zeta is None:
            raise ValueError("Please provide a zeta value for the forward pass of DualPreSampledLoss, else switch to an instance of DualPostSampledLoss.")
        else:
            return self.compute_dual(xi, xi_labels, zeta, zeta_labels)

    def __str__(self):
        return "Dual loss (sample BEFORE for loop)\n" + 10*"-" + "\n".join(map(str, self.parameters()))

    @property
    def presample(self):
        return True

DualLoss = DualPostSampledLoss













### [WIP] DO NOT TOUCH, HIGH RISK OF EXPLOSION ###
def entropic_loss_oracle(
        lam,
        zeta,
        zeta_labels,
        xi,
        xi_labels,
        rho,
        epsilon,
        loss,
        cost
        ):
    result, _, _ = EntropicLossOracle.apply(
            lam, zeta, zeta_labels, xi, xi_labels, rho, epsilon, loss, cost
            )
    return result

class EntropicLossOracle(ptag.Function):

    @staticmethod
    def forward(
            lam,
            zeta,
            zeta_labels,
            xi,
            xi_labels,
            rho,
            epsilon,
            loss,
            cost
            ):
        first_term = lam * rho

        l = loss.value(zeta, zeta_labels)
        c = cost(xi.unsqueeze(-1), zeta, xi_labels, zeta_labels)
        integrand = l - lam * c
        integrand /= epsilon

        # Expectation on the zeta samples
        second_term = pt.logsumexp(integrand, 0).mean(dim=0)
        second_term -= pt.log(pt.tensor(zeta.size(0)))
        # second_term *= epsilon
        print(c.shape, l.shape)
        return first_term + epsilon*second_term.mean(), c, l

    @staticmethod
    def setup_context(ctx: Any, inputs: Tuple, output: Any):
        lam, _, _, xi, xi_labels, rho, epsilon, _, _ = inputs
        _, c, l = output
        ctx.save_for_backward(lam, xi, xi_labels, rho, epsilon, c, l)

    @staticmethod
    def backward(ctx, grad_result, grad_c, grad_l):
        if grad_result is None:
            return 9*(None,)
        grad_theta = grad_lam = None
        grad_xi = grad_xi_labels = None
        grad_rho = grad_epsilon = None

        lam, xi, xi_labels, rho, epsilon, c, l = ctx.saved_tensors

        print("# gradl #####")
        print(grad_l)
        print("# gradc #####")
        print(grad_c)
        grad_l_theta, grad_l_zeta, grad_l_zeta_labels = grad_l
        grad_c_xi, grad_c_zeta, grad_c_xi_labels, grad_c_zeta_labels = grad_c
        if ctx.needs_input_grad[0]:
            grad_theta = EntropicLossOracle.grad_theta(
                    lam,
                    epsilon,
                    c,
                    l,
                    grad_l_theta
                )
        if ctx.needs_input_grad[1]:
            grad_lam = EntropicLossOracle.grad_lam(
                    lam,
                    rho,
                    epsilon,
                    c,
                    l
                )


        return grad_theta, grad_lam, None, None, grad_xi, grad_xi_labels, grad_rho, grad_epsilon, None, None

    @staticmethod
    def grad_lam(lam, rho, epsilon, c, l):
        integrand = l - lam * c
        integrand /= epsilon
        return rho - (c * F.softmax(integrand, dim=0)).sum(dim=0).mean()

    @staticmethod
    def grad_theta(lam, epsilon, c, l, grad_l_theta):
        integrand = l - lam * c
        integrand /= epsilon
        return (grad_l_theta * F.softmax(integrand, dim=0)).sum(dim=0).mean()
