r"""
=================================================
Comparison between some classification techniques
=================================================

Classification problems are all written in a simple way, as explained in `other tutorials <wdro.html>`__.

.. math::

   L_\theta(\xi) := \ell\left(\xi^\texttt{label}.\langle\theta\mid\xi^\texttt{input}\rangle\right)

We consider a simple classification problem to highlight the possibility of classifying samples with various methods, not to showcase their specificities.

We start with logistic regression, which is by far the most covered example of the library, and then present some minor modifications we can make to catter to some other classification techniques.
All of those losses are taken from [#IG08]_.

.. hint:: All those models are optimized in the default library settings, which represent uncertainty sets of type Wasserstein-2-2 (regularized), even though some of them are Lipschitz and thus may benefit from lower-order neighborhoods.
"""
from sklearn.datasets import make_blobs
import torch as pt
import torch.nn as nn
from tqdm import tqdm

from skwdro.torch import robustify
from skwdro.linear_models._logistic_regression import BiDiffSoftMarginLoss
from skwdro.solvers.oracle_torch import DualLoss

from examples.custom_models.utils.plotting import plot_decision_boundary

# %%
# Problem setup
# =============

SEED = 42

n = 100  # Number of observations
radius = pt.tensor(0.01)

X, y, *_ = make_blobs(centers=2, random_state=SEED)

device = "cpu"

X = pt.tensor(X).float().to(device)
y = pt.tensor(y).to(X).unsqueeze(-1) * 2. - 1.

# %%
# Training loop
# =============
# Define a function to train a model so that we can reuse it in various settings

def train(dual_loss: DualLoss, dataset: tuple[pt.Tensor, pt.Tensor], epochs: int=100):

    lbfgs = pt.optim.LBFGS(dual_loss.parameters())   # LBFGS is used to optimize thanks to the nature of the problem

    def closure():          # Closure for the LBFGS solver
        lbfgs.zero_grad()
        loss = dual_loss(*dataset).mean()
        loss.backward()
        return loss

    pbar = tqdm(range(epochs))
    # Every now and then, try to rectify the dual parameter (e.g. once per epoch).
    dual_loss.get_initial_guess_at_dual(*dataset)

    for _ in pbar:
        lbfgs.step(closure)
        if dual_loss.lam <= 0.:
            dual_loss._lam.requires_grad_(False)
            dual_loss._lam.mul_(0.)
            dual_loss._lam.requires_grad_(True)

        pbar.set_postfix({"lambda": f"{dual_loss.lam.item():.2f}"})

    t = dual_loss.primal_loss.transform
    assert isinstance(t, nn.Linear)
    return t

# %%
# First model: logistic regression
# ================================
#
# The loss function :math:`\ell` for this problem is the "soft-margin" function,
# a softened version of the hinge classification loss that we will see in the
# next example.
#
# .. math::
#
#    \ell(a) = \log\left(1+e^{-a}\right)
#
# .. note:: The native :py:class:`torch.nn.SoftMarginLoss` implementation of the
#    desired loss could have been satisfactory, but importantly **it is not
#    differentiable in its label argument**.
#    This is why we implement our own version of the "bi-differentiable"
#    soft-margin loss
#    :py:class:`skwdro.linear_models._logistic_regression.BiDiffSoftMarginLoss`.

model = nn.Linear(2, 1).to(X)
loss  = BiDiffSoftMarginLoss(reduction='none')
dro_model = robustify(
    loss,
    model,
    radius,
    X, y,
    seed=SEED,
    imp_samp=False
)


# %%
# First model: Training
# ~~~~~~~~~~~~~~~~~~~~~

model1 = train(dro_model, (X, y)) # type: ignore

model1.eval()  # type: ignore


# %%
# First model: Results
# ~~~~~~~~~~~~~~~~~~~~
#

plot_decision_boundary(model1, X, y, n_levels=20)

# %%
# Second model: SVM (primal)
# ==========================
#
# The classical *Support Vector Machine* model, with the Hinge loss and linear
# kernel, can be implemented in various ways. The one favored by `scikit-learn`
# e.g., when no ridge regularization is used, is as follows (see
# `this remark <https://scikit-learn.org/stable/modules/svm.html#linearsvc>`__):
#
# .. math::
#
#    \ell(a) = \max\{0, 1-a\}
#
# In our case, we suppose that the method is in the "big-data-low-dim" regime,
# in which using the *Kernel trick* would be detrimental.
# In this setting, the implementation is straightforward.

class HingeLoss(nn.Module):
    reduction: str = 'none'
    def forward(self, x, y):
        return nn.functional.relu(1. - y*x)

model = nn.Linear(2, 1).to(X)
loss  = HingeLoss()
dro_model = robustify(
    loss,
    model,
    radius,
    X, y,
    seed=SEED,
    imp_samp=False
)


# %%
# Second model: Training
# ~~~~~~~~~~~~~~~~~~~~~~

model2 = train(dro_model, (X, y), epochs=50) # type: ignore

model2.eval()  # type: ignore


# %%
# Second model: Results
# ~~~~~~~~~~~~~~~~~~~~~
#

plot_decision_boundary(model2, X, y, n_levels=20)

# %%
# Third model: Soft-SVM
# =====================
#
# This example covers nothing but an extension of the classical linear SVM, with
# a *smoothed* version of the Hinge loss presented above:
#
# .. math::
#
#    \ell(a) = \begin{cases}
#       \frac1{2}\max\{0, 1-a\}^2 & \text{if} a\ge 0\\
#       \frac{1}2-a & \text{otherwise.}
#    \end{cases}
#
# We solve it in exactly the same way.

class SmoothHingeLoss(nn.Module):
    reduction: str = 'none'
    gamma: float = 1.0

    def __init__(self, gamma=1.):
        super().__init__()
        self.gamma = gamma

    def forward(self, x, y):
        a = y*x
        quad_part = (.5 / self.gamma) * nn.functional.relu(1. - a)**2
        lin_part = 1. - 0.5 * self.gamma - a
        dec = (a >= 1. - self.gamma).detach()
        return quad_part * dec.float() + lin_part * pt.logical_not(dec).float()

model = nn.Linear(2, 1).to(X)
loss  = SmoothHingeLoss()
dro_model = robustify(
    loss,
    model,
    radius,
    X, y,
    seed=SEED,
    imp_samp=False
)


# %%
# Third model: Training
# ~~~~~~~~~~~~~~~~~~~~~

model3 = train(dro_model, (X, y), epochs=50) # type: ignore

model3.eval()  # type: ignore


# %%
# Third model: Results
# ~~~~~~~~~~~~~~~~~~~~
#

plot_decision_boundary(model3, X, y, n_levels=20)

# %%
# Fourth model: The Perceptron
# ============================
#
# This example covers the classical perceptron for classification task.
#
# .. math::
#
#    \ell(a) = \begin{cases}
#       \frac1{2}\max\{0, 1-a\}^2 & \text{if} a\ge 0\\
#       \frac{1}2-a & \text{otherwise.}
#    \end{cases}
#
# We solve it in exactly the same way.

class PerceptronLoss(nn.Module):
    reduction: str = 'none'
    def forward(self, x, y):
        return nn.functional.relu(-y*x)

model = nn.Linear(2, 1).to(X)
loss  = PerceptronLoss()
dro_model = robustify(
    loss,
    model,
    radius,
    X, y,
    seed=SEED,
    imp_samp=False
)


# %%
# Fourth model: Training
# ~~~~~~~~~~~~~~~~~~~~~~

model4 = train(dro_model, (X, y), epochs=50) # type: ignore

model4.eval()  # type: ignore


# %%
# Fourth model: Results
# ~~~~~~~~~~~~~~~~~~~~~
#

plot_decision_boundary(model4, X, y, n_levels=20)

# %%
# Fifth model: Quadratic margin loss
# ==================================
#
# This example covers a margin loss that is modeled as a quadratic form.
# It is substantially different from the other losses because it forces the
# cross-product :math:`\xi^\texttt{labels}\langle\theta\mid\xi^\texttt{input}\rangle`
# to be equal to one precisely, not to be greater to a margin like most others.
#
# .. math::
#
#    \ell(a) = (1-a)^2

class L2MarginLoss(nn.Module):
    reduction: str = 'none'
    def forward(self, x, y):
        return pt.pow(1. - y*x, 2)

model = nn.Linear(2, 1).to(X)
loss  = L2MarginLoss()
dro_model = robustify(
    loss,
    model,
    radius,
    X, y,
    seed=SEED,
    imp_samp=False
)


# %%
# Fifth model: Training
# ~~~~~~~~~~~~~~~~~~~~~

model5 = train(dro_model, (X, y), epochs=50) # type: ignore

model5.eval()  # type: ignore


# %%
# Fifth model: Results
# ~~~~~~~~~~~~~~~~~~~~
#

plot_decision_boundary(model5, X, y, n_levels=20)

# %%
# Sixth model: Quadratic Hinge loss
# =================================
#
# Same as last example, without the restriction mentioned: only the negative part
# of the margin is penalized by this loss.
#
# .. math::
#
#    \ell(a) = \max\{0, 1-a\}^2

class L2HingeLoss(nn.Module):
    reduction: str = 'none'
    def forward(self, x, y):
        return pt.pow(
            pt.nn.functional.relu(1. - y*x),
            2
        )

model = nn.Linear(2, 1).to(X)
loss  = L2HingeLoss()
dro_model = robustify(
    loss,
    model,
    radius,
    X, y,
    seed=SEED,
    imp_samp=False
)


# %%
# Sixth model: Training
# ~~~~~~~~~~~~~~~~~~~~~

model6 = train(dro_model, (X, y), epochs=50) # type: ignore

model6.eval()  # type: ignore


# %%
# Sixth model: Results
# ~~~~~~~~~~~~~~~~~~~~
#

plot_decision_boundary(model6, X, y, n_levels=20)

# %%
# Bonus: Exponential Loss
# =======================
#
# From a theoretical perspective, this loss is interesting for its lack of usual
# properties for the WDRO framework: it is not Lipschitz, and does not validate
# a 2nd order growth condition. It is not strongly convex either and not bounded.
#
# .. math::
#
#    \ell(a) = e^{-a}

class ExpLoss(nn.Module):
    reduction: str = 'none'
    def forward(self, x, y):
        return pt.exp(-y*x)

model = nn.Linear(2, 1).to(X)
loss  = ExpLoss()
dro_model = robustify(
    loss,
    model,
    radius,
    X, y,
    seed=SEED,
    imp_samp=False
)


# %%
# Bonus model: Training
# ~~~~~~~~~~~~~~~~~~~~~

model7 = train(dro_model, (X, y), epochs=50) # type: ignore

model7.eval()  # type: ignore


# %%
# Bonus model: Results
# ~~~~~~~~~~~~~~~~~~~~
#

plot_decision_boundary(model7, X, y, n_levels=20)

# %%
# References
# ==========
#
# .. [#IG08] Ingo and Christmann. **Support vector machines**,
#    *Springer Science*, 2008
