"""
Weber epsilon
"""
import numpy as np
from skwdro.operations_research._weber import Weber

def main():
    factories = np.array([
        [0., 1.],
        [-1., -1.],
        [2, -1]
    ])
    trafic = np.array([2.1, 2., 1.8])
    pbrs = [Weber(1e-1, n_zeta_samples=100, solver_reg=epsilon, sampler_reg=1e-3, cost="t-NC-2-2", random_state=0).fit(factories, trafic) for epsilon in [1e-7, 1e-5, 1e-4]]


if __name__ == '__main__':
    main()
