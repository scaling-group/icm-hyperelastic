<h1 align="center">In-context modeling</h1>

<h3 align="center">
  Retrain-free foundation modeling for computational science
</h3>

<p align="center">
  <a href="https://arxiv.org/abs/2604.23098"><strong>In-context modeling as a retrain-free paradigm for foundation models in computational science</strong></a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2604.23098"><img src="https://img.shields.io/badge/arXiv-2604.23098-b31b1b.svg" alt="arXiv:2604.23098"></a>
  <a href="https://arxiv.org/pdf/2604.23098"><img src="https://img.shields.io/badge/Paper-PDF-red.svg" alt="Paper PDF"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/-Python_3.11-blue?logo=python&logoColor=white" alt="Python 3.11"></a>
  <a href="https://hydra.cc/"><img src="https://img.shields.io/badge/Config-Hydra_1.3-89b8cd" alt="Hydra 1.3"></a>
  <a href="https://lightning.ai/"><img src="https://img.shields.io/badge/Training-Lightning_2.5-792EE5" alt="Lightning 2.5"></a>
  <a href="https://docs.astral.sh/ruff/"><img src="https://img.shields.io/badge/Code%20Style-Ruff-orange.svg?labelColor=gray" alt="Ruff"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg?labelColor=gray" alt="Apache 2.0 license"></a>
</p>

<p align="center">
  <a href="https://scaling-group.github.io">Scientific Computing and Intelligence Group (Scaling Group) @ NUS</a>
</p>

This repository accompanies our paper on **In-context modeling (ICM)**, a retrain-free paradigm for foundation models in computational science. ICM treats observational fields as context: given a small set of measurements or simulated fields from a new physical system, the model infers the physical relationships directly in a single forward pass, without fitting new system-specific parameters.

We demonstrate the idea on hyperelasticity, where a single model generalizes across unseen materials, geometries, and loading conditions, integrates with finite-element simulations, and is validated against experimental full-field measurements.

## Paper

This repository accompanies the arXiv preprint:

**[In-context modeling as a retrain-free paradigm for foundation models in computational science](https://arxiv.org/abs/2604.23098)**

Lingfeng Li, Zhuoyuan Li, Shun Li, Kaixin Zhan, Huajian Gao, Changqing Chen, and Liu Yang.

## Overview

The paper frames physical modeling as an in-context inference problem. Instead of training a separate surrogate or calibrating material parameters for each new system, ICM assimilates observational fields as physical context and predicts the corresponding response directly.

The main messages are:

- **Retrain-free generalization:** one model adapts to new physical systems through context, without per-system retraining.
- **Physics-informed learning:** the model is trained in a label-free manner using governing equations rather than relying only on supervised labels.
- **Hyperelasticity as a testbed:** ICM predicts stress fields across diverse materials, geometries and loading modes.
- **Simulation and experiment:** the framework connects to finite-element simulations and is validated with experimental full-field measurements.
- **Scaling behavior:** performance improves with increasing data diversity and computational budget, suggesting a path toward foundation models for computational science.

This codebase provides the training, validation, scaling, FEM, experiment, and embedding workflows used to support the study.

## Repository Layout

- `src/`: main training, datasets, datamodules, Lightning modules, callbacks, and utilities.
- `configs/`: Hydra configuration tree for data, models, optimizers, callbacks, loggers, and trainers.
- `scripts/`: batch scripts for scaling, validation, embedding, experiments, and FEM workflows.

Key entry points:

- `src/train.py`: main Hydra training and validation entry point for ICM.
- `configs/train_ice.yaml`: default ICM training composition.
- `configs/train_custom.example.yaml`: local template for machine-specific overrides such as dataset paths.

## First-time Setup

### Recommended: pixi

Install [pixi](https://pixi.sh/latest/installation/) and create one of the predefined environments:

```sh
pixi install
# or
pixi install -e cpu
# or
pixi install -e cu11
# or
pixi install -e cu12
```

Optional FEniCS support:

```sh
pixi install -e fem-cu11
# or
pixi install -e fem-cu12
```

### Alternative: uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and sync one CUDA target:

```sh
uv sync --extra cpu
# or
uv sync --extra cu118
# or
uv sync --extra cu124
# or
uv sync --extra cu126
```

### Alternative: conda + pip

```sh
conda create -n csn python=3.11 -y
conda activate csn
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install lightning
pip install numpy pandas h5py seaborn matplotlib
pip install hydra-core hydra-colorlog rootutils rich
pip install wandb mlflow
pip install tabulate einops
pip install pre-commit ruff yamllint
```

Optional developer setup:

```sh
pre-commit install
```

## Quick Start

### 1. Prepare dataset paths

If `configs/train_custom.yaml` exists, `src/train.py` will use it automatically. A convenient starting point is:

```sh
cp configs/train_custom.example.yaml configs/train_custom.yaml
```

Then set `paths.data_dir` to the directory containing the ICE datasets.

You can also override the path directly from the command line:

```sh
python src/train.py --config-name train_ice paths.data_dir=/absolute/path/to/data
```

### 2. Launch an ICM training run

Minimal example:

```sh
python src/train.py --config-name train_ice
```

A typical sweep is submitted through the provided shell scripts, for example:

```sh
qsub scripts/scaling_model/A-4.0M-340K-plane-strain.sh
```

These scripts are grouped by experiment family:

- `scripts/scaling_model/`: scaling with respect to model size.
- `scripts/scaling_data/`: scaling with respect to dataset size and diversity.
- `scripts/valid/`: validation runs.
- `scripts/experiment/`: experimental validation.
- `scripts/fem/`: FEM workflows.
- `scripts/embedding/`: embedding and representation analysis.

## Data

The datasets are not bundled with the repository. They will be released through Hugging Face:

**[scaling-group/icm-hyperelastic](https://huggingface.co/datasets/scaling-group/icm-hyperelastic)**

Download, decompression, folder layout, and preprocessing notes will be provided with the dataset release.

## Citation

```bibtex
@article{li2026incontext,
  title = {In-context modeling as a retrain-free paradigm for foundation models in computational science},
  author = {Li, Lingfeng and Li, Zhuoyuan and Li, Shun and Zhan, Kaixin and Gao, Huajian and Chen, Changqing and Yang, Liu},
  year = {2026},
  url = {https://arxiv.org/abs/2604.23098},
  eprint = {2604.23098},
  archivePrefix = {arXiv},
  primaryClass = {cs.CE}
}
```
