"""
TODO
"""
import tomllib as toml
from itertools import product
from numpy import linspace, logspace
from subprocess import STDOUT, run
from operator import itemgetter
from os import environ

def unpack_range(param: dict):
    descriptor = set(param.keys())
    if 'val' in descriptor:
        return [param['val']]
    elif 'num' in descriptor:
        assert 'min' in descriptor and 'max' in descriptor
        _min, _max, _num = param['min'], param['max'], param['num']
        return linspace(_min, _max, _num).data#(_max - _min) / _num)
    elif 'lnum' in descriptor:
        assert 'min' in descriptor and 'max' in descriptor
        _min, _max, _num = param['min'], param['max'], param['lnum']
        return logspace(_min, _max, _num).data#(_max - _min) / _num)
    else:
        raise NotImplementedError()

def extract_config(file):
    config = toml.load(file)
    return config['personal'], config['clusterparams'], config['logreg']

def make_args_string(logrho, d, n_train, n_test, n_zeta, epsilon, repeat, baseline: bool=False):
    output = 'ARGS=\"'
    output += f"--logrho {logrho} "
    output += f"--epsilon {epsilon} "
    output += f"--d {d} "
    output += f"--n_train {n_train} "
    output += f"--n_zeta {n_zeta} "
    output += f"--repeat {repeat}"
    if baseline: output += " -c"
    output += "\""
    return output

def launch_combinations(args_combinations, cluster, repeat, paths, baseline, hists):
    cluster_name, username, wall_time, nodes, cores, pname = map(
            str,
            itemgetter('cluster', 'user', 'walltime', 'nodes', 'cores', 'name')(cluster)
            )
    make_path, home_path, import_path = paths
    iteration = 0
    for n_zeta, d, n_train, n_test, logrho, epsilon in args_combinations:
        print(f"========= {iteration:=iteration+1} ========")
        message = f"Skwdro run d={d} n={n_train} logrho={logrho}, eps={epsilon}"
        environ['WANDB_NAME'] = message
        print(message, flush=True)
        time = int(wall_time) // 2 if baseline else int(wall_time)
        run(
            [
                make_path,
                'histsubmit' if hists else 'sksubmit',
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
    # baseline = input("Do you want to run as cvx (baseline)? [y/N] > ")
    # b = True
    # if baseline in {'true', 'True', '1', 'y', 'Y', 'Yes', 'yes'}:
    #     b = True
    # elif baseline in {'false', 'False', '0', 'n', 'N', 'No', 'no', ' ', ''}:
    #     b = False
    # else:
    #     raise ValueError("Please provide a valid answer to the above question")
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
        xptype = confdict['xptype']
        hists = xptype['hists']
        if xptype['baseline']:
            launch_combinations(args_combinations, cluster, repeat, paths, True, hists)
        if xptype['regularized']:
            launch_combinations(args_combinations, cluster, repeat, paths, False, hists)
            #print(f"make sksubmit ARGS=\"--logrho {logrho} --d {d} --n_train {n_train} --n_zeta {n_zeta} --repeat {repeat}\"")

if __name__ == '__main__':
    main()
