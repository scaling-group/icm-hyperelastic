#!/bin/bash
#PBS -P CFP01-SF-009
#PBS -j oe
#PBS -k oed
#PBS -N scaling
#PBS -l walltime=20:00:00
#PBS -l select=1:ngpus=1

cd $PBS_O_WORKDIR;

uv sync --extra cu126
uv tree

uv run python src/train.py logger=[csv,wandb] \
    task_name="train_scaling" \
    trainer.max_steps=100000 trainer.val_check_interval=50000 \
    model=pre-endecoder-A-4.0M \
    data=multiBC_ultimix_valid_train_scaling_plane_strain \
    "data.train.unify_material.dataset.material_cfg=\"list(range(2000))\"" \
    "data.train.unify_material.dataset.force_type_cfg=[\"uniaxial\", \"biaxial\", \"shear\"]" \
    "data.train.unify_material.dataset.geometry_cfg=[\"A\", \"B\", \"C\", \"D\", \"E\", \"F\", \"H\"]" \
    "data.train.unify_material.dataset.num_force_types_in_pool=\"torch.randint(1, 4, ()).item()\"" \
    "data.train.unify_material.dataset.num_geometries_in_pool=\"torch.randint(1, 8, ()).item()\""

echo "Done"
