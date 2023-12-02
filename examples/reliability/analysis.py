#!/usr/bin/env python
# coding: utf-8
import wandb
import matplotlib.pyplot as plt
import numpy as np
import os


api = wandb.Api()


path = "waiss-azizian/toolbox"
savepath = None


def get_data(d, n, logrho, repeat=100):
    filters = {"config.d": d, "config.n":n, "config.logrho":logrho, "config.repeat":repeat}
    run = api.runs(path=path, filters=filters)[0]
    return run.summary["avg_rel"], run.summary["sdev_rel"]

def get_data_rho_plot(d, n, sigma, log_rho_max=float('inf'), log_rho_min=-float('inf'), repeat=100, IS=None):
    filters = {"config.d": d, "config.n":n, "state":"Finished", "config.repeat":repeat, "config.sigma":sigma}
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
    return 10**(np.array(logrho)), np.array(avg), np.array(sdev)

def make_rho_plot(d, n, sigma, title="", rho_max=float('inf'), rho_min=-float('inf'), repeat=100, IS=None):
    plt.figure(figsize=(0.7*16,0.7*9))
    plt.xscale("log")
    rhos, avg, sdev = get_data_rho_plot(d, n, sigma, log_rho_max=np.log(rho_max), log_rho_min=np.log(rho_min), repeat=repeat, IS=IS)
    print(f"{len(np.unique(rhos))} unique among {len(rhos)} data points")
    label = "Regularized WDRO"
    for (rho, a) in zip(rhos, avg):
        print(f"{rho =} avg = {a}")
    plt.plot(rhos, avg, label=label)
    plt.fill_between(rhos, np.maximum(avg-sdev, 0.), np.minimum(avg+sdev, 1.0), alpha=0.3)
    plt.xlabel("Radius $\\rho$")
    plt.title(title)
    plt.legend()
    # plt.savefig(os.path.join(save_path, model + ".png"), dpi=600)
    plt.show()
    # tikzplotlib.save(title + ".tex", figure=tikzplotlib_fix_ncols(fig))

make_rho_plot(5, 500, 1.5, rho_max=1., title="Logistic Regression", repeat=200, IS=True)



