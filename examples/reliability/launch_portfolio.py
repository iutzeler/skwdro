"""
Launch Portfolio
"""
import tomllib as toml
from itertools import product
from numpy import linspace, logspace
from subprocess import STDOUT, run
from operator import itemgetter
from os import environ
from launch_logreg import unpack_range

def extract_config(file):
    config = toml.load(file)
    return config['personal'], config['clusterparams'], config['portfolio']

def make_args_string(logrho, d, n_train, n_test, n_zeta, epsilon, repeat, baseline: bool=False):
    output = 'ARGS=\"'
    output += f"--xp portfolio "
    output += f"--logrho {logrho} "
    output += f"--epsilon {epsilon} "
    output += f"--d {d} "
    output += f"--n_train {n_train} "
    output += f"--n_zeta {n_zeta} "
    output += f"--repeat {repeat}"
    if baseline: output += " -c"
    output += "\""
    return output

def launch_combinations(args_combinations, cluster, repeat, paths, baseline):
    cluster_name, username, wall_time, nodes, cores, pname = map(
            str,
            itemgetter('cluster', 'user', 'walltime', 'nodes', 'cores', 'name')(cluster)
            )
    make_path, home_path, import_path = paths
    for n_zeta, d, n_train, n_test, logrho, epsilon in args_combinations:
        message = f"Skwdro run d={d} n={n_train} logrho={logrho}, eps={epsilon}"
        environ['WANDB_NAME'] = message
        print(message, flush=True)
        time = int(wall_time) // 2 if baseline else int(wall_time)
        run(
            [
                make_path,
                'histsubmit',
                make_args_string(logrho, d, n_train, n_test, n_zeta, epsilon, repeat, baseline),
                f"WALLTIME={time}:00:00",
                f"HOME={home_path}",
                f"IMPORT={'/'.join((home_path, import_path))}",
                f"CLUSTER={cluster_name}",
                f"USER={username}",
                f"NODES={nodes}",
                f"CPU={cores}",
                f"NAME={pname}"
            ]
        )

def main():
    with open("params.toml", 'rb') as config:
        personal, cluster, confdict = extract_config(config)
        repeat = confdict['repeat']
        paths = tuple(map(
                str,
                itemgetter('makepath', 'homepath', 'import')(personal)
            ))
        make_path = paths[0]

        environ['WANDB_NOTES'] = 'Test python script launcher'
        environ['LC_NUMERIC'] = '\"en_US.UTF-8\"'

        run([make_path, 'cleanall'])
        args_combinations = product(*map(
            unpack_range,
            itemgetter('n_zeta', 'd', 'n_train', 'n_test', 'logrho', 'epsilon')(confdict)
        ))
        launch_combinations(args_combinations, cluster, repeat, paths, True)

if __name__ == '__main__':
    main()
