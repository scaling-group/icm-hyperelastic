<h1 align="center">In-context modeling</h1>

<h3 align="center">
  In-context learning as a retraining-free paradigm for constitutive modeling
</h3>

<p align="center">
  <a href="https://doi.org/10.1073/pnas.2620983123"><strong>Proc. Natl. Acad. Sci. U.S.A. 123 (44), e2620983123 (2026)</strong></a>
</p>

<p align="center">
  <a href="https://doi.org/10.1073/pnas.2620983123"><img src="https://img.shields.io/badge/PNAS-10.1073%2Fpnas.2620983123-1f4e79.svg" alt="PNAS DOI: 10.1073/pnas.2620983123"></a>
  <a href="https://www.pnas.org/doi/pdf/10.1073/pnas.2620983123"><img src="https://img.shields.io/badge/Paper-PDF-red.svg" alt="Paper PDF"></a>
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

This repository accompanies the paper published in *Proceedings of the National Academy of Sciences* (PNAS):

**[In-context learning as a retraining-free paradigm for constitutive modeling](https://doi.org/10.1073/pnas.2620983123)**

Lingfeng Li, Zhuoyuan Li, Shun Li, Kaixin Zhan, Huajian Gao, Chang Qing Chen, and Liu Yang.

*Proc. Natl. Acad. Sci. U.S.A.* **123** (44), e2620983123 (2026). https://doi.org/10.1073/pnas.2620983123

An earlier preprint version is available on [arXiv:2604.23098](https://arxiv.org/abs/2604.23098).

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
pip install tabulate einops optree frozendict
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

Then set `paths.data_dir` to the directory that contains the `ICE-elastic/` folder (see [Data](#data)), e.g. `${paths.root_dir}/data`.

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

### 3. Experimental validation

`scripts/experiment/valid-exp.sh` evaluates the plane-stress model on the experimental data (`dataset-exp-test`, see [Data](#data)). The context consists of DIC strain fields measured on perforated specimens of three geometries (three tests each), and the query is a uniaxial test of a standard dog-bone specimen.

By default (`configs/data/valid/valid_exp_mat*.yaml`) the context is sampled from all three geometries, as in the main text. The Supplementary Information repeats the prediction with one geometry at a time to show that the result does not depend on the specimen geometry. To run this, restrict the context to one geometry, e.g. geometry 3:

```sh
python src/train.py ... data=valid-exp \
  "data.valid.valid_exp_mat1.dataset.prompt_dataset.geometry_cfg.cfg=['3']" \
  "data.valid.valid_exp_mat1.dataset.prompt_dataset.num_geometries_in_pool='1'" \
  "data.valid.valid_exp_mat2.dataset.prompt_dataset.geometry_cfg.cfg=['3']" \
  "data.valid.valid_exp_mat2.dataset.prompt_dataset.num_geometries_in_pool='1'"
```

The number of context strain fields is set by `prompt_dataset.num_meshes_from_pool`, and the random sampling of the context by `data.valid_base_seed`.

## Data

The datasets are not bundled with the repository. They are released on Hugging Face as `.tar.zst` archives:

**[scaling-group/icm-hyperelastic](https://huggingface.co/datasets/scaling-group/icm-hyperelastic)**

| Archive | Used as | Configs |
| --- | --- | --- |
| `dataset-train-pe` / `dataset-train-ps` | Training set (plane strain / plane stress) | `data/train/*`, `data/valid/valid_0_*` |
| `dataset-val105-pe` / `-ps` | Test-ID | `data/valid/valid_105_*` |
| `dataset-val11-pe` / `-ps` | Test-M | `data/valid/valid_11_*` |
| `dataset-val13D-pe` / `-ps` | Test-MGL and Test-MGL+ | `data/valid/valid_13D_*` |
| `dataset-exp-test` | Experimental validation (two materials) | `data/valid/valid_exp_*` |

Download the archives and extract them into a folder named `ICE-elastic`:

```sh
hf download scaling-group/icm-hyperelastic --repo-type dataset --local-dir data/archives
mkdir -p data/ICE-elastic
for f in data/archives/*.tar.zst; do tar -I zstd -xf "$f" -C data/ICE-elastic; done
```

This gives the layout the configs expect, with `paths.data_dir=${paths.root_dir}/data`:

```text
data/ICE-elastic/
├── dataset-train-pe/Model-A/Model-A-shear-Poly-00000/{mesh.json, nodeX.npy, nodeU.npy, ...}
├── dataset-val105-pe/Model-A/Model-A-shear-00200/...
├── ...
└── dataset-exp-test/{exp-mat1,exp-mat2}/
```

Notes:

- Keep the folder name `ICE-elastic`; the data loaders use it to resolve file paths.
- The archives contain the raw simulation data only. On first use, the data loaders write preprocessed caches (`cache/`, `cache_snapshot/`) and sampled-context records into each dataset folder, so the folder must be writable. The caches take much more disk space than the archives (for example, about 60 GB for `dataset-val11-pe`), and building them the first time takes a while.
- Except for the experimental validation, which uses the plane-stress data, the results in the paper use the plane-strain (`pe`) data.

## Citation

```bibtex
@article{li2026incontext,
  title = {In-context learning as a retraining-free paradigm for constitutive modeling},
  author = {Li, Lingfeng and Li, Zhuoyuan and Li, Shun and Zhan, Kaixin and Gao, Huajian and Chen, Chang Qing and Yang, Liu},
  journal = {Proceedings of the National Academy of Sciences},
  volume = {123},
  number = {44},
  pages = {e2620983123},
  year = {2026},
  doi = {10.1073/pnas.2620983123},
  url = {https://doi.org/10.1073/pnas.2620983123}
}
```
