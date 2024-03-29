r"""
Distributionally Robust Portfolio optimization
==============================================

The objective and instances are based on [Bertsimas et al. (2018), Sec. 7.5] (for the original robust portfolio problem) and [Esfahani et al. (2018), Sec. 7] for the WDRO counterpart.

We consider a market of m = 10 assets whose yearly returns are modeled as the sum of i) a common risk factor following a normal distribution of mean 0 and variance 0.02; and ii) a specific factor following a normal distribution of mean 0.03*i and variance 0.025*i for asset i (i=1,..,m).

We solve the WDRO version of the portfolio optimization problem

.. math::

    \mathbb{E}[ - \langle x ; \xi \rangle ] + \eta \mathrm{CVar}_\alpha[- \langle x ; \xi \rangle]

which amounts to using the following loss function

.. math::

    \ell(x,\tau;\xi) =  - \langle x ; \xi \rangle + \eta \tau + \frac{1}{\alpha} \max( - \langle x ; \xi \rangle - \tau ; 0)

where :math:`\tau` is an extra real parameter accounting for the threshold of the CVaR (see [Rockafellar and Uryasev (2000)]). The parameter :math:`x` is constrained to live in the simplex (This is encoded in the constraints of the problem in [Esfahani et al. (2018)] and by an exponential reparametrization for the entropy-regularized version). 

We set the confidence level :math:`\alpha` of the CVaR to 0.2 and the robustness coefficient :math:`\eta` to 10.

"""
import numpy as np
import matplotlib.pyplot as plt
from skwdro.operations_research._portfolio import Portfolio


## PROBLEM SETUP
N = 30 # Number of observations
m = 10  # Number of assets

common_var = 0.02               # Variance of the common part
specific_mean_factor = 0.03     # Mean factor of the common part
specific_var_factor  = 0.025    # Variance factor of the specific part

psi = np.random.randn(N, 1) * np.sqrt(common_var)
zeta = np.random.randn(N, m) * np.sqrt(specific_var_factor) + specific_mean_factor * np.arange(1,m+1)[None, :]
returns_by_asset = psi + zeta

## RESOLUTION FOR DIFFERENT RADII
tested_radii = np.logspace(-3, 0, 10)

pbrs = [Portfolio(rho = i, alpha=.2, eta=10.).fit(returns_by_asset) for i in tested_radii]


data = np.vstack([m.coef_ for m in pbrs]).T
data = data[np.argsort(data[:, -1]), :]

# %%
# Robustness illustration
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# We plot the portfolio repartition for different values of the robustness radius. For small values of :math:`\rho`, the portfolio is concentrated on the most rewarding assets. Over a critical radius, the WDRO portfolio choice is equally divided among all assets which is the most robust strategy (regardless of the assets payoffs).


## Plotting
fig, ax = plt.subplots()
_ = plt.stackplot([m.rho for m in pbrs], data)
plt.xlabel("Robustness radius")
plt.xlim([np.min(tested_radii),np.max(tested_radii)])
ax.set_xscale('log')
plt.ylabel("Portfolio repartition")
plt.ylim([0,1])

plt.show()


# %%
# References
# ~~~~~~~~~~
#
# [Bertsimas et al. (2018)] Bertsimas, Dimitris, Vishal Gupta, and Nathan Kallus. "Robust sample average approximation." Mathematical Programming 171 (2018): 217-282.
#
# [Esfahani et al. (2018)] Mohajerin Esfahani, Peyman, and Daniel Kuhn. "Data-driven distributionally robust optimization using the Wasserstein metric: Performance guarantees and tractable reformulations." Mathematical Programming 171.1 (2018): 115-166.
#
# [Rockafellar and Uryasev (2000)] Rockafellar, R. Tyrrell, and Stanislav Uryasev. "Optimization of conditional value-at-risk." Journal of risk 2 (2000): 21-42.