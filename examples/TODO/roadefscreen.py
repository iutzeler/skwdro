"""
Roadef 24
"""
import torch as pt
import torch.nn as nn
from skwdro.base.problems import EmpiricalDistributionWithLabels
from skwdro.solvers.entropic_dual_torch import solve_dual_wdro
from skwdro.wrap_problem import dualize_primal_loss
from skwdro.solvers.optim_cond import OptCondTorch

class SimpleWeber(nn.Module):
    def __init__(self, d: int) -> None:
        super(SimpleWeber, self).__init__()
        self.pos = nn.Parameter(pt.zeros(d))
        self.d = d

    def forward(self, xi: pt.Tensor, xi_labels: pt.Tensor) -> pt.Tensor:
        distances = pt.linalg.norm(xi - self.pos.unsqueeze(0), dim=-1, keepdims=True)
        val = xi_labels * distances * xi_labels.shape[1]
        return val

def weber(X, y, rho):
    m, d = X.shape
    # Define the empirical Dirac comb
    emp = EmpiricalDistributionWithLabels(m=m,samples_x=X,samples_y=y.reshape(-1,1))

    # Build up the dual formulation for loss function to optimize
    _post_sample = True
    dual_loss = dualize_primal_loss(
            SimpleWeber(d),
            None,
            pt.tensor(rho),
            pt.Tensor(emp.samples_x),
            pt.Tensor(emp.samples_y),
            _post_sample
        )

    # Optimize with the default algorithm
    coef_, _, dual_var_, robust_loss_ = solve_dual_wdro(dual_loss,emp,opt=OptCondTorch(2))
    return coef_, robust_loss_
