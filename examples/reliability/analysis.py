#!/usr/bin/env python
# coding: utf-8
"""
Analaysis
"""
import wandb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.colors import LogNorm
from matplotlib import colormaps
from matplotlib.animation import FuncAnimation, PillowWriter

api = wandb.Api()

cm = colormaps['plasma']

path = "florian-vincent31/toolbox_eps"
savepath = None


def get_baseline_plot(d, n, sigma, log_rho_max=float('inf'), log_rho_min=-float('inf'), repeat=100, IS=None):
    filters = {"config.d": d, "state":"Finished", "config.sigma":sigma}
    filters['config.n_zeta'] = 100
    filters['config.cvx_wdro'] = True
    if IS is not None:
        filters.update({"config.IS":IS})
    order = "config.logrho.value"
    runs = api.runs(path=path, filters=filters, order=order)
    logrho, avg, sdev = [], [], []
    for run in runs:
        if run.config["logrho"] > log_rho_max or run.config["logrho"] < log_rho_min:
            continue
        try:
            avg.append(run.summary["avg_rel"])
        except KeyError:
            pass
        else:
            sdev.append(run.summary["sdev_rel"])
            logrho.append(run.config["logrho"])
    return 10**(np.array(logrho)), None, np.array(avg), np.array(sdev)

def get_data_rho_plot(d, n, sigma, log_rho_max=float('inf'), log_rho_min=-float('inf'), repeat=100, IS=None):
    filters = {"config.d": d, "state":"Finished", "config.sigma":sigma}
    filters['config.n_zeta'] = 100
    filters['$not'] = [{'config.cvx_wdro': {'$eq': True}}]
    if IS is not None:
        filters.update({"config.IS":IS})
    order = "config.logrho.value"
    runs = api.runs(path=path, filters=filters, order=order)
    logrho, eps, avg, sdev = [], [], [], []
    for run in runs:
        if run.config["logrho"] > log_rho_max or run.config["logrho"] < log_rho_min:
            continue
        try:
            avg.append(run.summary["avg_rel"])
        except KeyError:
            pass
        else:
            sdev.append(run.summary["sdev_rel"])
            lrh = run.config["logrho"]
            logrho.append(lrh)
            eps.append(run.config.get('epsilon', np.nan))#(10.**lrh) * 5e-3))
    return 10**(np.array(logrho)), np.array(eps), np.array(avg), np.array(sdev)

def grouping_curves(rhos, eps, avg, sdev, base=False):
    if base:
        df =  pd.DataFrame({'rho': rhos, 'avg': avg, 'std': sdev})
        return df, df.groupby('rho').mean().reset_index()
    else:
        df = pd.DataFrame({'rho': rhos, 'eps': eps, 'avg': avg, 'std': sdev})
        avgd = df.groupby(['rho', 'eps']).mean().reset_index()
        vard = df.groupby(['rho', 'eps']).var().reset_index()
        avgd.loc['std'] = (avgd['std']**2 + vard['avg']).apply(np.sqrt)
        return df, avgd

def make_one_plot(d, n, sigma, rho_max, rho_min, repeat, IS, base=False):
    getter = get_baseline_plot if base else get_data_rho_plot
    rhos, eps, avg, sdev = getter(d, n, sigma, log_rho_max=np.log(rho_max), log_rho_min=-np.inf if rho_min <= 0. else np.log(rho_min), repeat=repeat, IS=IS)
    print(f"{len(np.unique(rhos))} unique among {len(rhos)} data points")
    label = "Baseline WDRO (cvx)" if base else "Regularized WDRO (torch)"

    df, gdf = grouping_curves(rhos, eps, avg, sdev, base)

    if base:
        color = 'k'

        rhos, avg, sdev = gdf[['rho', 'avg', 'std']].to_numpy().T
        return [(rhos, avg, sdev, color, label)]
        # plt.plot(rhos, avg, label=label, c=color)
        # plt.fill_between(rhos, np.maximum(avg-sdev, 0.), np.minimum(avg+sdev, 1.0), alpha=0.1, color=color)

    else:
        cn = LogNorm(vmin=float(df['eps'].min()), vmax=1.)
        combis = []
        for e, data in gdf.groupby('eps'):
            assert isinstance(e, float)
            color = cm(cn(e))
            rhos, avg, sdev = data[['rho', 'avg', 'std']].to_numpy().T
            combis.append((rhos, avg, sdev, color, label+f"$\\epsilon={e}$"))
            # plt.plot(rhos, avg, label=label+f"$\\epsilon={e}$", color=color)
            # plt.fill_between(rhos, np.maximum(avg-sdev, 0.), np.minimum(avg+sdev, 1.0), alpha=0.1, color=color)
        return combis

def make_rho_plot(d, n, sigma, title="", rho_max=float('inf'), rho_min=0., repeat=100, IS=None):
    # plt.figure(figsize=(0.7*16,0.7*9))
    fig, ax = plt.subplots(figsize=(0.7*16,0.7*9))
    fig.suptitle(title)
    ax.set_xscale("log")
    ax.set_xlabel("Radius $\\rho$")

    print("CVX...")
    cvx_line, = make_one_plot(d, n, sigma, rho_max, rho_min, repeat, IS, True)
    print("Torch...")
    torch_lines = make_one_plot(d, n, sigma, rho_max, rho_min, repeat, IS, False)

    ax.plot(cvx_line[0], cvx_line[1], c=cvx_line[3], label="Baseline WDRO (cvx)")
    ax.legend()
    framerange = range(len(torch_lines))
    lines = [ax.plot([], [])[0] for _ in framerange]
    areas = [ax.fill_between([], [], [], alpha=.1) for _ in framerange]
    def update_lines(frame, tls, lines, areas):
        ax.get_legend().remove()
        for i, tl, line in zip(framerange, tls, lines):
            if i == frame:
                rhos, avg, sdev, color, label = tl
                line.set(data=(rhos, avg), label=label, color=color)
                areas[i].set_array([])
                areas[i].set_lw(0.)
                areas[i] = ax.fill_between(rhos, np.maximum(avg-sdev, 0.), np.minimum(avg+sdev, 1.0), alpha=0.2, color=color)
            else:
                line.set_data([], [])
                areas[i].set_array([])
                areas[i].set_lw(0.)

        ax.legend()
        return lines

    ani = FuncAnimation(fig, update_lines, len(torch_lines), fargs=(torch_lines, lines, areas), interval=500)
    writer = PillowWriter(fps=1,
                          metadata=dict(artist='Flo'),
                          bitrate=1800)
    ani.save('epsplots.gif', writer=writer)
    # plt.savefig(os.path.join(save_path, model + ".png"), dpi=600)
    plt.show()
    # tikzplotlib.save(title + ".tex", figure=tikzplotlib_fix_ncols(fig))

if __name__ == '__main__':
    make_rho_plot(5, 500, 1., rho_max=.15, title="Logistic Regression", repeat=1000, IS=True)
