#!/bin/bash
#PBS -P CFP01-SF-009
#PBS -j oe
#PBS -k oed
#PBS -N elastic
#PBS -l walltime=10:00:00
#PBS -l select=1:ngpus=1
##----- CPU/Mem will be allocated at 10/200gb per GPU. -----
##----- sample config for ngpus of 2, 4, 8, 16 via either line below ----
###PBS -l select=1:ngpus=2
###PBS -l select=1:ngpus=4
###PBS -l select=1:ngpus=8
###PBS -l select=2:ngpus=8

cd $PBS_O_WORKDIR;

source ~/.bashrc
source .venv/bin/activate

uv sync --extra cu126
uv tree

# feel free to adjust GPU number.

# save commands of eval here.
# must include setups that influence eval, e.g. model name and ckpts.
# Other training setups (e.g. batch size, etc.) are not needed.

uv run python src/train.py logger=[csv,wandb] \
 model=pre-endecoder-A-4.0M \
 train=False \
 paths.restore_ckpts=["/scratch/lingfli/ice-dev/logs/train_scaling_plane_strain/runs/4.0M-340K/checkpoints/last.ckpt"] \
 data=multiBC_ultimix_valid_test_scaling_plane_strain \
 callbacks=ice_stress \
 task_name="test_valid_stress_plane_strain"

echo "Done"

##**************************************************************************
##   WARNING and IMPORTANT NOTICE                                          *
##**************************************************************************
##   DON'T SET  [CUDA_VISIBLE_DEVICES]  in your Python Program!            *
##   PBS Job Scheduler will set GPU devices for the job automatically.     *
##**************************************************************************
