"""
Portfolio
"""
import numpy as np
import matplotlib.pyplot as plt
from skwdro.operations_research._portfolio import Portfolio

def plots(r, pbrs):
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": 'STIXGeneral',
        "mathtext.fontset": 'cm'
    })
    fig, ax = plt.subplots()

    #cm = sb.color_palette("flare_r", as_cmap=True)
    #cn = None#LogNorm(vmin=trafic.min(), vmax=trafic.max())
    #assert isinstance(cm, Colormap)
    data = np.vstack([m.coef_ for m in pbrs]).T
    data = data[np.argsort(data[:, -1]), :]
    ax.stackplot([m.rho for m in pbrs], data)
    ax.set_xscale('log')
    fig.savefig("robust_portfolio.png", transparent=True)

def main():

    psi = np.random.randn(100, 1) * 2e-2
    zeta = (np.random.randn(100, 10) * 2.5e-2 + 3e-2) * np.arange(10)[None, :]
    returns_by_asset = psi + zeta
    #pb = Portfolio(0., n_zeta_samples=-1, cost="t-NC-2-2", seed=42)
    #pb.fit(returns_by_asset)
    pbrs = [Portfolio(i, n_zeta_samples=100, cost="t-NC-2-2", seed=i, solver="entropic_torch", alpha=.2, eta=10., solver_reg=1e-5).fit(returns_by_asset) for i in np.logspace(1, -4, 15)]
    plots(returns_by_asset, pbrs)


if __name__ == '__main__':
    main()
