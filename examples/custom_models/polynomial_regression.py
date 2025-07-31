r"""
Polynomial regression
=====================

Polynomial regression is a simple 1D regression. The samples are of the form :math:`\xi = (x,y) \in \mathbb{R}\times\mathbb{R}` and the sought predictor is of the form  :math:`f(x) = \sum_{i=0}^d a_i x^i` where :math:`(a_0,..,a_d)` are the :math:`d+1` coefficients to lean.

In the following example, we seek to learn a polynomial fitting the function 

.. math::

    f^\star(x) = \frac{10}{e^{x}+e^{-x}} + x

from :math:`n=100` samples uniformly drawn from :math:`[-2,2]` and corrupted by a Gaussian noise with zero mean and variance :math:`0.1`. 


"""
from typing import Iterable
import numpy as np
import matplotlib.pyplot as plt
import torch as pt
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from skwdro.torch import robustify
from skwdro.solvers.oracle_torch import DualLoss

# %%
# Problem setup
# ~~~~~~~~~~~~~

n = 100  # Number of observations

var = pt.tensor(0.1)  # Variance of the noise

# Generating function
def f_star(x):
    return 10/(pt.exp(x)+pt.exp(-x)) + x
 
xi = pt.rand(n)*4.0 - 2.0  # x_i's are uniformly drawn from (-2,2] 
xi = pt.sort(xi)[0]  # we sort them for easier plotting
yi = f_star(xi) + pt.sqrt(var)*pt.randn(n)  # y_i's are f(x_i) + noise

# Build minibatches lazily with a dataloader
dataset = DataLoader(TensorDataset(xi.unsqueeze(-1), yi.unsqueeze(-1)), batch_size=n//2, shuffle=True)

degree = 4                                  # Degree of the regression

device = "cpu"

# %%
# Polynomial model
# ~~~~~~~~~~~~~~~~

class PolynomialModel(nn.Module):
    def __init__(self, degree : int) -> None:
        super().__init__()
        self._degree = degree
        self.linear = nn.Linear(self._degree, 1)

    def forward(self, x):
        return self.linear(self._polynomial_features(x))

    def _polynomial_features(self, x):
        return pt.cat([x ** i for i in range(1, self._degree + 1)],dim=-1)

model = PolynomialModel(degree).to(device)  # Our polynomial regression model
loss  = nn.MSELoss(reduction='none')        # Our error will be measure in quadratic loss

# %%
# Training loop
# ~~~~~~~~~~~~~
# Define a function to train a model so that we can reuse it in various settings

def train(dual_loss: DualLoss, dataset: Iterable[tuple[pt.Tensor, pt.Tensor]], epochs: int=10):

    lbfgs = pt.optim.LBFGS(dual_loss.parameters())   # LBFGS is used to optimize thanks to the nature of the problem

    def closure():          # Closure for the LBFGS solver
        lbfgs.zero_grad()
        loss = dual_loss(xi, xi_label, reset_sampler=True).mean()
        loss.backward()
        return loss

    pbar = tqdm(range(epochs))

    for _ in pbar:
        # Every now and then, try to rectify the dual parameter (e.g. once per epoch).
        dual_loss.get_initial_guess_at_dual(*next(iter(dataset))) # *

        # Main train loop
        inpbar = tqdm(dataset, leave=False)
        for xi, xi_label in inpbar:
            lbfgs.step(closure)

        pbar.set_postfix({"lambda": f"{dual_loss.lam.item():.2f}"})

    return dual_loss.primal_loss.transform

# %%
# Training
# ~~~~~~~~

radius = pt.tensor(0.001)   # Robustness radius

dual_loss = robustify( 
    loss,
    model,
    radius,
    xi.unsqueeze(-1),
    yi.unsqueeze(-1)
) # Replaces the loss of the model by the dual WDRO loss

model1 = train(dual_loss, dataset, epochs=5) # type: ignore

model1.eval()  # type: ignore


# %%
# Results
# ~~~~~~~
#
# We plot the obtained polynomial and print the coefficients

fig, ax = plt.subplots()
xtrial = pt.linspace(-2.1,2.1,100)
ax.scatter(xi.cpu(), yi.cpu(), c='g', label='train data')
ax.plot(xtrial, f_star(xtrial), 'k', label='generating function')

pred1 = model1(xtrial[:,None,None]).detach().cpu().squeeze() # type: ignore
ax.plot(xtrial,pred1, 'r', label='WDRO prediction')  

fig.legend()
plt.show()


coeffs = model1.linear.weight.tolist()[0] # type: ignore
biais  = model1.linear.bias.tolist()[0] # type: ignore
polyString = "Polynomial regressor (degree {:d}, radius {:3.2e}):\n {:3.2f} ".format(degree,radius.float(),biais)
for i,a in enumerate(coeffs):
    if a>=0.0:
        polyString += "+ {:3.2f}x**{:d} ".format(a,i+1)
    else:
        polyString += "- {:3.2f}x**{:d} ".format(abs(a),i+1)

print(polyString)


# %%
#
# Different degree and radius, with a more compact formulation.

radius2 = pt.tensor(1e-6)   # Robustness radius
degree2 = 7

model2 = train(robustify( 
            nn.MSELoss(reduction='none'),
            PolynomialModel(degree2).to(device),
            radius2,
            xi.unsqueeze(-1),
            yi.unsqueeze(-1)
        ), dataset, epochs=5) # type: ignore

# %%
# 

fig, ax = plt.subplots()
xtrial = pt.linspace(-2.1,2.1,100)
ax.scatter(xi.cpu(), yi.cpu(), c='g', label='train data')
ax.plot(xtrial, f_star(xtrial), 'k', label='generating function')

pred2 = model2(xtrial[:,None,None]).detach().cpu().squeeze() # type: ignore
ax.plot(xtrial,pred2, 'r', label='WDRO prediction')  

fig.legend()
plt.show()


coeffs = model2.linear.weight.tolist()[0] # type: ignore
biais  = model2.linear.bias.tolist()[0] # type: ignore
polyString = "Polynomial regressor (degree {:d}, radius {:3.2e}):\n {:3.2f} ".format(degree2,radius2.float(),biais)
for i,a in enumerate(coeffs):
    if a>=0.0:
        polyString += "+ {:3.2f}x**{:d} ".format(a,i+1)
    else:
        polyString += "- {:3.2f}x**{:d} ".format(abs(a),i+1)

print(polyString)




