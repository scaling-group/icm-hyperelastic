import hydra
import numpy as np
import torch

# typing
from omegaconf import DictConfig
from optree import PyTree

from src.datasets import ice_data_multi_utils as idu
from src.datasets import pytree_utils as ptu

from . import dataloader_utils as dlu
from .base_datamodule import BaseDataModule


class IceDataModule(BaseDataModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

    def get_train_dataset_from_cfg(self, cfg) -> list[dict]:
        """
        Different from the base class, we return a list of datasets, each with its own cfg.
        While it returns a list of datasets, semantically, it is still one dataset.
        This is useful for the case if we want to cycle over geometries,
        e.g, snapshot split with no slice
        """
        if cfg.geometry_loop == "cycle":
            # cycle through the geometry ids
            datasets = []
            for g_id in cfg.geometry_ids:
                ds_cfg = cfg.dataset.copy()
                ds_cfg.geometry_cfg = [g_id]
                dataset = hydra.utils.instantiate(ds_cfg)
                datasets.append({"dataset": dataset, "cfg": ds_cfg})
            return datasets
        elif cfg.geometry_loop == "mix":
            # mix the geometries
            dataset = hydra.utils.instantiate(cfg.dataset)
            return [{"dataset": dataset, "cfg": cfg}]
        else:
            raise ValueError(f"Invalid geometry loop: {cfg.geometry_loop}")

    def _slice_eqn(self, batch: PyTree, num_eqns: int):
        # TODO: add rng to control the slice
        indices = torch.randperm(batch["data"]["A"].shape[1])[:num_eqns]
        new_batch = {
            "description": batch["description"],
            "data": {k: v[:, indices, ...] for k, v in batch["data"].items()},
        }
        if "data_boundary" in batch:
            new_batch["data_boundary"] = batch["data_boundary"]
        return new_batch

    def get_train_collate_fn(self, cfg):
        """
        Here cfg is still one dataset cfg.
        """

        def collate_fn_slice(batch_list: list[PyTree]):
            range_low, range_high = cfg.eqns_per_prompt
            # slice the data to get a random number of equations
            min_num_eqns_from_each = min(d["data"]["A"].shape[1] for d in batch_list)
            # max slice-able size
            if min_num_eqns_from_each < range_low:
                num_eqns = min_num_eqns_from_each  # slice to the minimum number of equations
                description_prefix = f"sliced to {num_eqns} smaller than [{range_low}, {range_high}]"
            else:
                # slice in the range of [min, min(max, min_num_eqns_from_each)]
                range_high = min(range_high, min_num_eqns_from_each)
                num_eqns = torch.randint(range_low, range_high + 1, (1,)).item()
                description_prefix = f"sliced to {num_eqns} in [{range_low}, {range_high}]"

            collated_batch = ptu.concat([self._slice_eqn(batch, num_eqns) for batch in batch_list])
            sizes = [batch["data"]["A"].shape[1] for batch in batch_list]
            collated_batch["description"] = np.array(
                [
                    f"original size={s} {description_prefix}, {d}"
                    for s, d in zip(sizes, collated_batch["description"], strict=True)
                ],
                dtype=np.dtypes.StringDType(),
            )
            return collated_batch

        # if cfg.eqns_per_prompt is None, we do not slice the data
        return ptu.concat if cfg.eqns_per_prompt is None else collate_fn_slice

    def get_valid_test_collate_fn(self, cfg):
        """
        we don't collate the MeshListData for validation/test
        set batch_size = 1, take the first element in the list
        """

        def collate_fn(data_list: list[idu.DataMeshList]):
            return data_list[0]

        return collate_fn

    def train_dataloader(self):
        """
        Cycle through the dataloaders of the training set.
        :return: a CycleLoader of the training dataloaders

        This is different from the base class
        now self.train_datasets is a list of list of datasets
        we need to create a CycleLoader for each sublist
        """
        return dlu.CycleLoader(
            [
                {
                    "dataloader": dlu.CycleLoader([self.get_train_dataloader(**ds) for ds in ds_sublist]),
                    "sampler": None,
                }
                for ds_sublist in self.train_datasets
            ]
        )
