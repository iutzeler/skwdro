#/usr/bin/env bash
#cd /home/wazizian/Documents/PhD/WDRO/singularity

export LC_NUMERIC="en_US.UTF-8"
make cleanall

python3.11 launch_logreg.py

# for n_zeta in 100;
# do
# 	for d in 5;
# 	do
# 		for n in 100;
# 		do
# 			for r in $(seq -7.0 0.5 -0.);
# 			do
# 				# Name and notes optional
# 				export WANDB_NAME="Skwdro run d=$d n=$n r=$r"
# 				export WANDB_NOTES="Smaller learning rate, more regularization."
# 				echo $r
# 				make sksubmit ARGS="--logrho $r --d $d --n_train $n --n_zeta $n_zeta --repeat 1000";
# 			done;
# 		done;
# 	done;
# done;
