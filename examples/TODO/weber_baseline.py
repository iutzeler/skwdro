"""
Weber Baseline
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from matplotlib.colors import LogNorm, Colormap
from matplotlib.cm import ScalarMappable
from skwdro.operations_research._weber import Weber

def plots(factories, trafic, pbrs, pb):
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": 'STIXGeneral',
        "mathtext.fontset": 'cm'
    })
    fig, ax = plt.subplots()

    cm = sb.color_palette("flare_r", as_cmap=True)
    cn = LogNorm(vmin=trafic.min(), vmax=trafic.max())
    assert isinstance(cm, Colormap)
    ax.scatter(*factories.T, c='b')
    for factory, w in zip(factories, trafic):
        ax.plot([factory[0], pb.coef_[0]], [factory[1], pb.coef_[1]], c=cm(cn(w)))
    fig.colorbar(ScalarMappable(cmap=cm, norm=cn), ax=ax, label=r'train fees')
    ax.scatter(*pb.coef_, c="k", marker='+', s=50.)
    ax.scatter(*pb.coef_, s=100., facecolors='none', edgecolors='k')
    fig.savefig("baseline_weber.png", transparent=True)

def main():


    factories = np.array([
        [0., 1.],
        [-1., -1.],
        [2, -1]
    ])
    trafic = np.array([2.1, 2., 1.8])
    pb = Weber(0., kappa=1., n_zeta_samples=-1, cost="t-NLC-2-2", random_state=42)
    pb.fit(factories, trafic)
    pbrs = [Weber(1e-1, kappa=10., n_zeta_samples=10, cost="t-NLC-2-2-10", random_state=i).fit(factories, trafic) for i in range(5)]
    plots(factories, trafic, pbrs, pb)


if __name__ == '__main__':
    main()
