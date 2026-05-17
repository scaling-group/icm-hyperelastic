#!/bin/bash
#PBS -P CFP01-SF-009
#PBS -j oe
#PBS -k oed
#PBS -N scaling
#PBS -l walltime=40:00:00
#PBS -l select=1:ngpus=2
##----- CPU/Mem will be allocated at 10/200gb per GPU. -----
##----- sample config for ngpus of 2, 4, 8, 16 via either line below ----
###PBS -l select=1:ngpus=2
###PBS -l select=1:ngpus=4
###PBS -l select=1:ngpus=8
###PBS -l select=2:ngpus=8

cd $PBS_O_WORKDIR;

uv sync --extra cu126
uv tree

uv run python src/train.py logger=[csv,wandb] data=multiBC_ultimix_valid_train_scaling_plane_strain task_name="train_scaling"\
    model=pre-endecoder-A-9.5M \
    trainer.max_steps=154000 trainer.val_check_interval=15400

echo "Done"

##**************************************************************************
##   WARNING and IMPORTANT NOTICE                                          *
##**************************************************************************
##   DON'T SET  [CUDA_VISIBLE_DEVICES]  in your Python Program!            *
##   PBS Job Scheduler will set GPU devices for the job automatically.     *
##**************************************************************************
