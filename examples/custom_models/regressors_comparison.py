r"""
=============================================
Comparison between some regression techniques
=============================================

Regression problems are all written in a simple way, as explained in `other tutorials <wdro.html>`__.

.. math::

   L_\theta(\xi) := \ell\left(\xi^\texttt{label} - \langle\theta\mid\xi^\texttt{input}\rangle\right)

We consider a simple 1D regression problem to highlight the possibility of using various losses in the library, not to showcase their specificities.

We start with logistic regression, which is by far the most covered example of the library, and then present some minor modifications we can make to catter to some other classification techniques.
All of those losses are taken from [#IG08]_.

.. hint:: All those models are optimized in the default library settings, which represent uncertainty sets of type Wasserstein-2-2 (regularized), even though some of them are Lipschitz and thus may benefit from lower-order neighborhoods.
"""
import torch as pt
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

from skwdro.torch import robustify
from skwdro.solvers.oracle_torch import DualLoss

from examples.custom_models.utils.plotting import plot_linear_relation


# %%
# Problem setup
# =============

SEED = 42

NOISE_LEVEL = 3.
radius = pt.tensor(1.)

device = "cpu"

X = pt.arange(-10, 10, .2, device=device).unsqueeze(-1)
A = pt.randn(1).to(X)
y = A * X + pt.randn_like(X) * NOISE_LEVEL

# %%
# Training loop
# =============
# Define a function to train a model so that we can reuse it in various settings

def train(dual_loss: DualLoss, dataset: tuple[pt.Tensor, pt.Tensor], epochs: int=100):

    optimizer = pt.optim.SGD(dual_loss.parameters(), nesterov=True, momentum=.1)   # LBFGS is used to optimize thanks to the nature of the problem

    def closure():          # Closure for the LBFGS solver
        optimizer.zero_grad()
        loss = dual_loss(*dataset).mean()
        loss.backward()
        return loss

    pbar = tqdm(range(epochs))
    # Every now and then, try to rectify the dual parameter (e.g. once per epoch).
    dual_loss.get_initial_guess_at_dual(*dataset)

    for _ in pbar:
        l = closure()
        optimizer.step()
        if dual_loss.lam <= 0.:
            dual_loss._lam.requires_grad_(False)
            dual_loss._lam.mul_(0.)
            dual_loss._lam.requires_grad_(True)
        if l < 1e-3:
            break

        pbar.set_postfix({"lambda": f"{dual_loss.lam.item():.2f}", "loss": f"{l.item():.2e}"})

    t = dual_loss.primal_loss.transform
    assert isinstance(t, nn.Linear)
    return t

# %%
# First model: Ordinary Least Squares
# ===================================
#
# This is the most simple least squares problem, from which we build the others.
#
# .. math::
#
#    \ell(a) = \frac12\|a\|^2
#
# We solve it in exactly the same way as before.

class QuadraticLoss(nn.Module):
    reduction: str = 'none'
    def forward(self, x, y):
        a = y - x
        return .5 * a**2

model = nn.Linear(1, 1).to(X)
loss  = QuadraticLoss()
dro_model = robustify(
    loss,
    model,
    radius,
    X, y,
    seed=SEED,
    imp_samp=False
)

plt.plot(X, loss(X, A*X))
plt.title("OLS loss (quadratic)")
pass


# %%
# First model: Training
# ~~~~~~~~~~~~~~~~~~~~~

model4 = train(dro_model, (X, y), epochs=1000) # type: ignore

model4.eval()  # type: ignore


# %%
# First model: Results
# ~~~~~~~~~~~~~~~~~~~~
#

plot_linear_relation(X, y, A, model4)

# %%
# Second model: SVR (primal)
# ==========================
#
# The classical *Support Vector Machine* model, with the
# :math:`\tau`\ -insensitive loss and linear kernel, can be implemented in
# various ways. The one favored by `scikit-learn` e.g., when no ridge
# regularization is used, is as follows (see
# `this remark <https://scikit-learn.org/stable/modules/svm.html#linearsvr>`__):
#
# .. math::
#
#    \ell(a) = \max\{0, \|a\|-\tau\}
#
# In our case, we suppose that the method is in the "big-data-low-dim" regime,
# in which using the *Kernel trick* would be detrimental.
# In this setting, the implementation is straightforward.

class InsensitiveLoss(nn.Module):
    reduction: str = 'none'
    insensitivity: float = .3

    def __init__(self, insensitivity=.3):
        super().__init__()
        self.insensitivity = insensitivity

    def forward(self, x, y):
        return nn.functional.relu(pt.abs(y - x) - self.insensitivity)

model = nn.Linear(1, 1).to(X)
loss  = InsensitiveLoss()
dro_model = robustify(
    loss,
    model,
    radius,
    X, y,
    seed=SEED,
    imp_samp=False
)

plt.plot(X, loss(X, A*X))
plt.title("$\\tau$-insensitive loss")
pass

# %%
# Second model: Training
# ~~~~~~~~~~~~~~~~~~~~~~

model2 = train(dro_model, (X, y), epochs=1000) # type: ignore

model2.eval()  # type: ignore


# %%
# Second model: Results
# ~~~~~~~~~~~~~~~~~~~~~
#

plot_linear_relation(X, y, A, model2)

# %%
# Third model: Quantile-regression
# ================================
#
# In the setting of quantile regression, we may fallback to a problem that is
# very similar to linear regression, in which the loss on the residuals
# is called the *Pinball Loss*. It is defined relative to a quantile selection
# :math:`\tau`.
#
# .. math::
#
#    \ell(a) = \begin{cases}
#       -(1-\tau)a & \text{if} a\ge 0\\
#       \tau & \text{otherwise.}
#    \end{cases}
#
# We solve it in exactly the same way as before.

class PinballLoss(nn.Module):
    reduction: str = 'none'
    quantile: float = 1.0

    def __init__(self, quantile=.1):
        super().__init__()
        self.quantile = quantile

    def forward(self, x, y):
        a = y - x
        pos = (a >= 0.).float()
        return (self.quantile*pos + (1. - self.quantile)*(1. - pos)) * a.abs()

model = nn.Linear(1, 1).to(X)
loss  = PinballLoss(.05)
dro_model = robustify(
    loss,
    model,
    radius,
    X, y,
    seed=SEED,
    imp_samp=False
)

plt.plot(X, loss(X, A*X))
plt.title("Pinball loss")
pass

# %%
# Third model: Training
# ~~~~~~~~~~~~~~~~~~~~~

model3 = train(dro_model, (X, y), epochs=1000) # type: ignore

model3.eval()  # type: ignore


# %%
# Third model: Results
# ~~~~~~~~~~~~~~~~~~~~
#

plot_linear_relation(X, y, A, model3)

# %%
# Fourth model: Outlier-robust regression
# =======================================
#
# A pervasive way to make a regression robust to outliers (a kind of robustness
# that is fundamentaly different from that of DRO) is to abandon the strict
# convexity of ordinary least squares to make the loss linear after some
# threshold. This loss is called the *Huber Loss* with parameter :math:`\delta`.
#
# .. math::
#
#    \ell(a) = \begin{cases}
#       \frac{|a|}2 & \text{if} |a|\le \delta\\
#       \delta |a| - \frac{\delta^3}2 & \text{otherwise.}
#    \end{cases}

class HuberLoss(nn.Module):
    reduction: str = 'none'
    quantile: float = 1.0

    def __init__(self, quantile=.9):
        super().__init__()
        self.quantile = quantile

    def forward(self, x, y):
        a = y - x
        r = a.abs()
        pos = (r >= self.quantile).float()
        outer = self.quantile*r - .5*self.quantile**3
        inner = .5 * a**2
        return outer*pos + inner*(1. - pos)

model = nn.Linear(1, 1).to(X)
loss  = HuberLoss(.95)
dro_model = robustify(
    loss,
    model,
    radius,
    X, y,
    seed=SEED,
    imp_samp=False
)

plt.plot(X, loss(X, A*X))
plt.title("Huber loss")
pass


# %%
# Fourth model: Training
# ~~~~~~~~~~~~~~~~~~~~~~

model4 = train(dro_model, (X, y), epochs=1000) # type: ignore

model4.eval()  # type: ignore


# %%
# Fourth model: Results
# ~~~~~~~~~~~~~~~~~~~~~
#

plot_linear_relation(X, y, A, model4)

# %%
# Fifth model: Least Absolute Errors
# ==================================
#
# As OLS but with the L1 metric instead, which is more robust to outliers but has
# worse numerical properties than the Huber loss.
#
# .. math::
#
#    \ell(a) = \|a\|

class L1Loss(nn.Module):
    reduction: str = 'none'
    def forward(self, x, y):
        a = y - x
        return a.abs()

model = nn.Linear(1, 1).to(X)
loss  = L1Loss()
dro_model = robustify(
    loss,
    model,
    radius,
    X, y,
    seed=SEED,
    imp_samp=False
)

plt.plot(X, loss(X, A*X))
plt.title("LAE loss")
pass


# %%
# Fifth model: Training
# ~~~~~~~~~~~~~~~~~~~~~

model5 = train(dro_model, (X, y), epochs=1000) # type: ignore

model5.eval()  # type: ignore


# %%
# Fifth model: Results
# ~~~~~~~~~~~~~~~~~~~~
#

plot_linear_relation(X, y, A, model5)

# %%
# Sixth model: Quadratic SVR
# ==========================
#
# As the SVR described above, except it is squared, and hence has common
# properties with the OLS.
#
# .. math::
#
#    \ell(a) = \max\{0, \|a\|\}^2

class L2InsensitiveLoss(nn.Module):
    reduction: str = 'none'
    insensitivity: float = .3

    def __init__(self, insensitivity=.3):
        super().__init__()
        self.insensitivity = insensitivity

    def forward(self, x, y):
        return nn.functional.relu(pt.abs(y - x) - self.insensitivity) ** 2

model = nn.Linear(1, 1).to(X)
loss  = L2InsensitiveLoss()
dro_model = robustify(
    loss,
    model,
    radius,
    X, y,
    seed=SEED,
    imp_samp=False
)

plt.plot(X, loss(X, A*X))
plt.title("Squared-$\\tau$-insensitive loss")
pass


# %%
# Sixth model: Training
# ~~~~~~~~~~~~~~~~~~~~~

model6 = train(dro_model, (X, y), epochs=1000) # type: ignore

model6.eval()  # type: ignore


# %%
# Sixth model: Results
# ~~~~~~~~~~~~~~~~~~~~
#

plot_linear_relation(X, y, A, model6)

# %%
# References
# ==========
#
# .. [#IG08] Ingo and Christmann. **Support vector machines**,
#    *Springer Science*, 2008
