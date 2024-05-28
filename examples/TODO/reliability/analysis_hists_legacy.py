"""
Analaysis
"""
import matplotlib.collections
import matplotlib.transforms
import matplotlib.patches
import wandb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
import seaborn.objects as so
import argparse as ap

from matplotlib.colors import LogNorm, Colormap
from matplotlib.cm import ScalarMappable

PATH = "florian-vincent31/toolbox_hists_logreg"
HIDE = False
DISCRIMINATE_TOPBOTTOM = "robust"
KEYS = ['ERM train loss', (' ' if HIDE else '') + 'robust train loss', 'ERM test loss', (' ' if HIDE else '') + 'robust test loss']
PLOT_ORDER = KEYS[-1::-2] + KEYS[-2::-2]
DEFAULTS_COLORS = {
        'ERM train loss': (0., 0.5, .9, .5),
        ('_' if HIDE else '') + 'robust train loss': (1., 0., 0., .9),
        'ERM test loss': (0., 1., 0., .5),
        'erm': (0., 0., 1., .5)
        }

def plot_kde(ax, df, p):
    sb.kdeplot(df, x='loss', hue='loss_type', ax=ax, palette=p, alpha=.5, fill=True, common_norm=False, bw_adjust=1.5, hue_order=[h for h in PLOT_ORDER if h in p.keys()])
    if HIDE: ax.get_legend().set_visible(False)
    else: ax.get_legend()
    return ax.get_legend_handles_labels()

def histplot_diffs(ax, df, c):
    df = pd.DataFrame({'erm': df.loc[:, KEYS[2]] - df.loc[:, KEYS[0]], ('_' if HIDE else '') + 'skwdro': df.loc[:, KEYS[3]] - df.loc[:, KEYS[1]]})
    df = pd.melt(df, var_name='loss_type', value_name='loss')
    return plot_kde(ax, df, {'_skwdro': c, 'erm': DEFAULTS_COLORS['erm']})

def histplot_loss(ax, df, c):
    df = pd.melt(df.loc[:, KEYS], var_name='loss_type', value_name='loss')
    ax.axis('off')
    u_ = plot_kde(ax, df, {KEYS[-1]: c, **{k:v for k, v in DEFAULTS_COLORS.items() if k != 'erm'}})
    for name, c in zip(PLOT_ORDER, ax.get_children()):
        if isinstance(c, matplotlib.collections.PolyCollection) and DISCRIMINATE_TOPBOTTOM not in name:
            c.set_transform(matplotlib.transforms.Affine2D(np.eye(3) * np.array([[1, -1, 1]])) + c.get_transform())
    l = ax.get_ylim()
    ax.set_ylim([-l[1]/5, l[1]/5])
    return u_

def histplot_test(ax, df, c):
    df = pd.melt(df.loc[:, KEYS[2:]], var_name='loss_type', value_name='loss')
    return plot_kde(ax, df, {KEYS[-1]: c, 'test_loss': DEFAULTS_COLORS['test_loss']})

OPTIONCHOICES = {'test': histplot_test, 'all': histplot_loss, 'diffs': histplot_diffs}

def true_then_only_false():
    yield True
    while True: yield False
ttof = true_then_only_false

def check_optional(maybeval, checked, sup=False):
    if sup:
        return maybeval is None or (checked >= maybeval)
    else:
        return maybeval is None or (checked <= maybeval)

def checkrange(min, max, val):
    return check_optional(min, val, sup=True) and check_optional(max, val)

def generate_plots(rhomin=None, rhomax=None, epsmin=None, epsmax=None):
    api = wandb.Api()

    plt.rc('legend',fontsize=15, title_fontsize=15)
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": 'STIXGeneral',
        "mathtext.fontset": 'cm'
    })

    cm = sb.color_palette("flare_r", as_cmap=True)# colormaps['plasma']
    assert isinstance(cm, Colormap)

    ftrs = {}
    runs = api.runs(PATH, filters=ftrs, order="config.logrho.value")

    parser = ap.ArgumentParser(
        prog='HistAnalysis',
        description='Compute the histograms of skwdro from cluster computations.',
        epilog='Copyright: Flo')
    parser.add_argument('-m', "--mode", default='test', choices=OPTIONCHOICES.keys())
    histfunc = OPTIONCHOICES[parser.parse_args().mode]

    rs = []
    for run in runs:
        base = run.config['cvx_wdro']
        if not base:
            rho = np.power(10., run.config['logrho'])
            if checkrange(rhomin, rhomax, rho):
                epsilon = run.config['epsilon']
                if checkrange(epsmin, epsmax, epsilon):
                    sigma = run.config['sigma']
                    h = run.history(keys=list(map(lambda s: s[1:] if s[0] == '_' else s, map(lambda s: s.replace(' ', '_').removeprefix('ERM_'), KEYS))))
                    if '_step' in h.columns:
                        h = h.drop(['_step'], axis=1)
                        h['eps'] = epsilon
                        h['s'] = sigma
                        h['rho'] = rho
                        rs.append(h)
    df = pd.concat(rs).rename(columns={
        'robust_test_loss': ('_' if HIDE else '') + 'robust test loss',
        'robust_train_loss': ('_' if HIDE else '') + 'robust train loss',
        'train_loss': 'ERM train loss',
        'test_loss': 'ERM test loss'
    })
    rhos = df['rho'].unique()
    epsilons = df['eps'].unique()

    fig, axes = plt.subplots(len(rhos), len(epsilons), sharex=True, sharey=True, squeeze=False)
    cn = LogNorm(vmin=float(df['rho'].min()), vmax=float(df['rho'].max()))
    hide = axes.shape[0] > 1
    for axline, (r, rdf), sett in zip(axes, df.groupby('rho'), ttof()):
        assert isinstance(r, float)
        for ax, (e, edf), sety in zip(axline, rdf.groupby('eps'), ttof()):
            results = histfunc(ax, edf, cm(cn(r)))
            if sety and axes.shape[0] > 1:
                ax.set_ylabel(f"$\\rho={r:.1e}$", rotation=60)
            if sett:
                ax.set_title(f"$\\varepsilon={e:.2e}$")
            if sety and sett:
                #fig.legend(*results, loc='upper right')
                #ax.get_legend().set_visible(True)
                pass
    fig.align_ylabels(axes[:, 0])
    if hide:
        fig.colorbar(ScalarMappable(cmap=cm, norm=cn), ax=axes, label=r'$\rho$')
        fig.suptitle(r"Influence of $\rho$\&$\epsilon$ on the robust loss vs. the test loss")
    else:
        fig.tight_layout()
    plt.show()
    fig.savefig("rho2em1_eps1em2.png", transparent=True)

def main():
    generate_plots(2e-1, 3e-1, 1e-2, 1e-1)

if __name__ == '__main__':
    main()
