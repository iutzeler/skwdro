#/usr/bin/env bash
cd /home/wazizian/Documents/PhD/WDRO/singularity

make cleanall

for d in 5;
do
    for n in 500;
    do
        for r in $(seq -7.0 0.5 -0.);
        do
            make sksubmit ARGS="--logrho $r --d $d --n $n --repeat 200";
        done;
    done;
done;
