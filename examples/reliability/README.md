# Singularity job launch

Use the python script `launch_logreg.py` to launch the jobs on dahu with your parameters of choice.
Set the parameters in the `params.toml` file (especialy the `params` part), including the paths for the files.

Launch with:
```shell
$ python3 launch_logreg.py
```

If you need libraries, use your own environment with python>3.12 and the main scientific libraries installed.
