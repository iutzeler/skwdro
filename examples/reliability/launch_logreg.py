import tomllib as toml
from itertools import product
from numpy import linspace
from subprocess import STDOUT, run
from operator import itemgetter
from os import environ

def unpack_range(param: dict):
    descriptor = set(param.keys())
    if 'val' in descriptor:
        return [param['val']]
    elif 'num' in descriptor:
        assert 'min' in descriptor and 'max' in descriptor
        _min, _max, _num = param['min'], param['max'] + 1, param['num']
        return linspace(_min, _max, _num).data#(_max - _min) / _num)
    else:
        raise NotImplementedError()

def extract_config(file):
    config = toml.load(file)
    return config['personal'], config['clusterparams'], config['logreg']

def make_args_string(logrho, d, n_train, n_test, n_zeta, repeat, baseline: bool=False):
    output = 'ARGS=\"'
    output += f"--logrho {logrho} "
    output += f"--d {d} "
    output += f"--n_train {n_train} "
    output += f"--n_zeta {n_zeta} "
    output += f"--repeat {repeat}"
    if baseline: output += " -c"
    output += "\""
    return output

def main():
    baseline = input("Do you want to run as cvx (baseline)? [y/N] > ")
    b = True
    if baseline in {'true', 'True', '1', 'y', 'Y', 'Yes', 'yes'}:
        b = True
    elif baseline in {'false', 'False', '0', 'n', 'N', 'No', 'no', ' ', ''}:
        b = False
    else:
        raise ValueError("Please provide a valid answer to the above question")
    with open("params.toml", 'rb') as config:
        personal, cluster, confdict = extract_config(config)
        make_path, home_path, import_path = map(
                str,
                itemgetter('makepath', 'homepath', 'import')(personal)
            )
        cluster_name, username, wall_time, nodes, cores, pname = map(
                str,
                itemgetter('cluster', 'user', 'walltime', 'nodes', 'cores', 'name')(cluster)
                )
        repeat = confdict['repeat']

        environ['WANDB_NOTES'] = 'Test python script launcher'
        environ['LC_NUMERIC'] = '\"en_US.UTF-8\"'

        run([make_path, 'cleanall'])
        args_combinations = product(*map(
            unpack_range,
            itemgetter('n_zeta', 'd', 'n_train', 'n_test', 'logrho')(confdict)
        ))
        for n_zeta, d, n_train, n_test, logrho in args_combinations:
            message = f"Skwdro run d={d} n={n_train} logrho={logrho}"
            environ['WANDB_NAME'] = message
            print(message, flush=True)
            run(
                [
                    make_path,
                    'sksubmit',
                    make_args_string(logrho, d, n_train, n_test, n_zeta, repeat, b),
                    f"WALLTIME={wall_time}",
                    f"HOME={home_path}",
                    f"IMPORT={'/'.join((home_path, import_path))}",
                    f"CLUSTER={cluster_name}",
                    f"USER={username}",
                    f"NODES={nodes}",
                    f"CPU={cores}",
                    f"NAME={pname}"
                ]
            )
            #print(f"make sksubmit ARGS=\"--logrho {logrho} --d {d} --n_train {n_train} --n_zeta {n_zeta} --repeat {repeat}\"")

if __name__ == '__main__':
    main()
