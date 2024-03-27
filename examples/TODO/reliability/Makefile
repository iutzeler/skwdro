CLUSTER?=dahu
CLUSTER_USER?=vincenfl

#ifeq ($(CLUSTER),luke)
#	SINGULARITY_BIN=./old_singularity/bin/singularity
#else
SINGULARITY_BIN=singularity
SINGULARITY_BUILD=$(SINGULARITY_BIN) build -f -F
#endif

BASE?=base
DEV?=dev

HOME?=/home/flvincen
IMPORT?=$(HOME)/ljk/skwdro
SSH_CONFIG:=$(HOME)/.ssh/config
SSH=ssh -F $(SSH_CONFIG)

$(BASE).sif:$(BASE).def pyproject.toml
	$(SINGULARITY_BUILD) $(BASE).sif $(BASE).def

skwdro_hists.sif: export SINGULARITY_TMPDIR=exemples/reliability/tmp
$(SK).sif: export SINGULARITY_TMPDIR=exemples/reliability/tmp

.PHONY:shell_base
shell_base:$(BASE).sif
	$(SINGULARITY_BIN) instance start --contain $(BASE).sif base
	$(SINGULARITY_BIN) shell instance://base
	$(SINGULARITY_BIN) instance stop base

.PHONY:test_base
test_base:$(BASE).sif
	$(SINGULARITY_BIN) run $(BASE).sif

# Modern version, unsused for compatibility
# export SINGULARITY_IMPORT=$(IMPORT)

$(DEV).sif:$(DEV).def $(BASE).sif
	echo $(IMPORT) > singularity_import.txt
	$(SINGULARITY_BIN) build --notest --force $(DEV).sif $(DEV).def
	rm singularity_import.txt

.PHONY:shell_dev
shell_dev:$(DEV).sif
	$(SINGULARITY_BIN) instance start --containall $(DEV).sif dev
	$(SINGULARITY_BIN) shell instance://dev
	$(SINGULARITY_BIN) instance stop dev

SK=skwdro
SKIMPORT=$(HOME)/ljk/skwdro
$(SK).sif:$(SK).def
	echo $(SKIMPORT) > singularity_import.txt
	$(SINGULARITY_BUILD) --notest --force $(SK).sif $(SK).def
	rm singularity_import.txt

.PHONY:shell_skwdro
shell_skwdro:$(SK).sif
	$(SINGULARITY_BIN) instance start --containall $(SK).sif dev
	$(SINGULARITY_BIN) shell instance://dev
	$(SINGULARITY_BIN) instance stop dev

.PHONY: clean
clean:
	rm -f $(DEV).sif

.PHONY:cleanall
cleanall:clean
	rm -f $(BASE).sif
	rm -f $(SK).sif
	rm -f skwdro_hists.sif

.PHONY:init_ssh
init_ssh:
	sh $(ssh-agent)
	ssh-add

# SUBMIT_SCRIPT?=submit

.PHONY:deploy
deploy:$(DEV).sif
	# chmod +x $(SUBMIT_SCRIPT).sh
	rsync -Pavu $(DEV).sif $(CLUSTER).gricad:/home/$(CLUSTER_USER)

.PHONY:skdeploy
skdeploy:$(SK).sif
	rsync -Pavu $(SK).sif $(CLUSTER).gricad:/home/$(CLUSTER_USER)

.PHONY:connect
connect:
	$(SSH) $(CLUSTER).gricad

NAME?=invmeasures
NODES?=1
CPU?=16
WALLTIME?=12:00:00
RES=/nodes=$(NODES)/core=$(CPU),walltime=$(WALLTIME)
PROJECT=pr-skwdro

ARGS?=""
ifeq ($(CLUSTER),luke)
ADD_COMMAND=source /applis/environments/conda.sh && conda activate singularity &&
else
ADD_COMMAND=
endif

GIT_BYPASS=git config --global --add safe.directory /skwdro &&
.PHONY: gitbypass
gitbypass:
	$(GIT_BYPASS)

.PHONY:submit
submit:deploy
	$(SSH) $(CLUSTER).gricad "oarsub -n $(NAME) -l $(RES) --project $(PROJECT) \"$(ADD_COMMAND) singularity run --bind /tmp:/import/wandb $(DEV).sif $(ARGS)\""

.PHONY:sksubmit
sksubmit:skdeploy
	$(SSH) $(CLUSTER).gricad "oarsub -n $(NAME) -l $(RES) --project $(PROJECT) \"$(ADD_COMMAND) $(GIT_BYPASS) singularity run --bind /tmp:/skwdro/wandb $(SK).sif $(ARGS)\""


#==============
# Histograms
skwdro_hists.sif:skwdro_hists.def
	echo $(SKIMPORT) > singularity_import.txt
	$(SINGULARITY_BUILD) --notest --force skwdro_hists.sif skwdro_hists.def
	rm singularity_import.txt

.PHONY:histdeploy
histdeploy:skwdro_hists.sif
	rsync -Pavu skwdro_hists.sif $(CLUSTER).gricad:/home/$(CLUSTER_USER)

.PHONY:histsubmit
histsubmit: histdeploy
	$(SSH) $(CLUSTER).gricad "oarsub -n SKWDRO_histograms -l $(RES) --project $(PROJECT) \"$(ADD_COMMAND) singularity run --bind /tmp:/skwdro/wandb skwdro_hists.sif $(ARGS)\""
#==============


.PHONY:clear_logs
clear_logs:
	$(SSH) $(CLUSTER).gricad "rm OAR*"

setup:
	ssh-add
	$(SSH) $(CLUSTER).gricad "$(GIT_BYPASS)"
