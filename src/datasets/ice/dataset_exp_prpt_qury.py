from itertools import product
from pathlib import Path
from typing import Literal

import einops
import numpy as np
import torch
from optree import PyTree
from torch.utils.data import Dataset

# from src.datasets import ice_data_multi_utils as idu
from src.datasets import ice_data_exp as ide
from src.datasets import pytree_utils as ptu
from src.utils import RankedLogger

from .utils import (
    ExperimentConfig,
    GeometryConfig,
)

log = RankedLogger(__name__, rank_zero_only=True)

PackIdType = tuple[str, str]
# g_id, e_id, t_id


def _get_pytree_from_mesh(mesh: ide.DataMeshExp) -> PyTree:
    # add A and XI to IFdata
    for boundary in mesh.IFdata:
        boundary["A"] = ide.slice_nodes_set(mesh.A, boundary["nodeset"])
        boundary["XI"] = ide.slice_nodes_set(mesh.XI, boundary["nodeset"])
        # no need to add mask, since we assume padded A = 0
        # boundary["mask"] = idu.slice_nodes_set(mesh["mask"], boundary["nodeset"])
    pytree = {
        "mesh": mesh.mesh,
        "X": mesh.X,
        "U": mesh.U,
        "BCset": mesh.BCset,
        "FREEset": mesh.FREEset,
        "Xip": mesh.Xip,
        "B": mesh.B,
        "Bcoeff": mesh.Bcoeff,
        "F": mesh.F,
        "P": mesh.P,
        "S": mesh.S,
        "IFdata": mesh.IFdata,
        "E": mesh.E,  # property for _E
        "C": mesh.C,  # property for _C
        "I": mesh.I,  # property for _I
        "grad_I_grad_C": mesh.grad_I_grad_C,  # property for _grad_I_grad_C
        "A": mesh.A,  # property for linear_equation["A"]
        "XI": mesh.XI,  # property for linear_equation["XI"]
        "mask": mesh.mask,  # property for linear_equation["mask"]
    }
    pytree = ptu.to_tensor(pytree)  # TODO: only move necessary data to tensor
    return pytree


class ExpPrpt:
    def __init__(
        self,
        data_dir: str,
        geometry_cfg: GeometryConfig,  # all available geometries
        experiment_cfg: ExperimentConfig,  # all available experiments
        num_meshes: int,
        retrieve_mode: Literal["same", "proportional"],  # 'proportional' or 'same'
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)

        # self.material_cfg = material_cfg
        self.geometry_cfg = geometry_cfg
        self.experiment_cfg = experiment_cfg

        # self.material_ids: Sequence[int] = eval(self.material_cfg) #! unique for now
        self.geometry_ids = self.geometry_cfg.geometry_ids
        # self.force_type_ids = self.force_type_cfg.force_type_ids
        self.experiment_ids = self.experiment_cfg.experiment_ids

        self.num_meshes = num_meshes
        self.retrieve_mode = retrieve_mode

        dataset_name = (
            # f"material:{self.material_cfg}, "
            f"geometry:{self.geometry_cfg}, experiment:{self.experiment_cfg}, "
        )
        self.dataset_name = f"Exp, {dataset_name}, "

        # be careful, this may cache the data in disk if not exist
        # therefore might have multi-process issue
        # be sure the instantiate the dataset on prepare_data() of lightning datamodule
        self.feasible_pack_id: set[PackIdType] = set()

        num_success_samples = 0
        num_total_samples = 0
        for g_id, e_id in product(self.geometry_ids, self.experiment_ids):
            num_total_samples += 1
            try:
                ide.DataMeshExp.touch_by_id(data_dir=self.data_dir, g_id=g_id, e_id=e_id)
                data = ide.DataMeshExp.load_by_id(data_dir=self.data_dir, g_id=g_id, e_id=e_id)
                self.feasible_pack_id.add((g_id, e_id))
                num_success_samples += 1
                log.info(f"({g_id=}, {e_id=}) is valid for num stretches: {data.X.shape[0]}")
            except Exception as e:
                log.warning(f"Failed to touch mesh with: {g_id=}; {e_id=}. Error: {e}")
                continue

        assert self.num_meshes <= len(self.feasible_pack_id), (
            "The number of meshes should not exceed the number of experimental settings"
        )

    @staticmethod
    def _get_id(tgt_len: int, idx: int, src_len: int, mode: str) -> int:
        if mode == "proportional":
            return round(idx * 1.0 / (src_len - 1) * (tgt_len - 1))
        elif mode == "same":
            if idx < tgt_len:
                return idx
            else:
                return tgt_len - 1
        else:
            raise ValueError

    @staticmethod
    def I_dist(src: torch.Tensor, tgt: torch.Tensor) -> float:
        src_flatten: torch.Tensor = einops.rearrange(src, "... d -> (...) 1 d")  # d = 2 for spatial dimension
        tgt_flatten: torch.Tensor = einops.rearrange(tgt, "... d -> 1 (...) d")  # d = 2 for spatial dimension
        dists = (src_flatten - tgt_flatten).square().sum(dim=-1)  # (src_len, tgt_len)
        return dists.min(dim=-1)[0].max(dim=-1)[0].item()

    def retrieve(self, idx: int, src_len: int, qury_XI: torch.Tensor) -> list[PyTree]:
        if self.retrieve_mode in ["same", "proportional"]:
            feasible_mesh_cfg: set[tuple[str, str, int]] = set()

            for g_id, e_id in self.feasible_pack_id:
                mesh = ide.DataMeshExp.load_by_id(data_dir=self.data_dir, g_id=g_id, e_id=e_id)
                num_stretches = mesh.X.shape[0]
                t_id = self._get_id(num_stretches, idx, src_len, self.retrieve_mode)
                feasible_mesh_cfg.add((g_id, e_id, t_id))

            random_idxs = [idx for idx in torch.randperm(len(feasible_mesh_cfg))[: self.num_meshes]]
            selected_keys = [list(feasible_mesh_cfg)[idx] for idx in random_idxs]

        elif self.retrieve_mode == "similarity":
            mesh_cfg_to_dist: dict[tuple[str, str, int], int] = {}

            for g_id, e_id in self.feasible_pack_id:
                mesh = ide.DataMeshExp.load_by_id(data_dir=self.data_dir, g_id=g_id, e_id=e_id)
                for t_id in range(mesh.XI.shape[0]):
                    mesh_cfg_to_dist[(g_id, e_id, t_id)] = self.I_dist(qury_XI, mesh.XI[t_id])

            sorted_kv = sorted(mesh_cfg_to_dist.items(), key=lambda kv: kv[1])
            selected_keys = [k for k, _ in sorted_kv[: self.num_meshes]]

        elif self.retrieve_mode == "random":
            feasible_mesh_cfg: set[tuple[str, str, int]] = set()
            for g_id, e_id in self.feasible_pack_id:
                mesh = ide.DataMeshExp.load_by_id(data_dir=self.data_dir, g_id=g_id, e_id=e_id)
                for t_id in range(mesh.XI.shape[0]):
                    feasible_mesh_cfg.add((g_id, e_id, t_id))

            random_idxs = [idx for idx in torch.randperm(len(feasible_mesh_cfg))[: self.num_meshes]]
            selected_keys = [list(feasible_mesh_cfg)[idx] for idx in random_idxs]

        else:
            raise ValueError

        mesh_list = []
        for g_id, e_id, t_id in selected_keys:
            mesh = ide.DataMeshExp.load_by_id(data_dir=self.data_dir, g_id=g_id, e_id=e_id)
            mesh = _get_pytree_from_mesh(mesh)
            mesh = ptu.get_one_sample(mesh, t_id, keep_dim=True)
            mesh_list.append(mesh)

        return mesh_list


class ExpQury(Dataset):
    def __init__(
        self,
        data_dir: str,
        geometry_cfg: GeometryConfig,  # all available geometries
        experiment_cfg: ExperimentConfig,  # all available experiments
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)

        # self.material_cfg = material_cfg
        self.geometry_cfg = geometry_cfg
        self.experiment_cfg = experiment_cfg

        # self.material_ids: Sequence[int] = eval(self.material_cfg) #! unique for now
        self.geometry_ids = self.geometry_cfg.geometry_ids
        # self.force_type_ids = self.force_type_cfg.force_type_ids
        self.experiment_ids = self.experiment_cfg.experiment_ids

        dataset_name = (
            # f"material:{self.material_cfg}, "
            f"geometry:{self.geometry_cfg}, experiment:{self.experiment_cfg}, "
        )
        self.dataset_name = f"Exp, {dataset_name}, "

        # be careful, this may cache the data in disk if not exist
        # therefore might have multi-process issue
        # be sure the instantiate the dataset on prepare_data() of lightning datamodule
        self.pack_id_to_num_stretches: dict[PackIdType, int] = {}
        self.all_pack_ids: set[tuple[str, str, int]] = set()

        num_success_samples = 0
        num_total_samples = 0
        for g_id, e_id in product(self.geometry_ids, self.experiment_ids):
            num_total_samples += 1
            try:
                ide.DataMeshExp.touch_by_id(data_dir=self.data_dir, g_id=g_id, e_id=e_id)
                data = ide.DataMeshExp.load_by_id(data_dir=self.data_dir, g_id=g_id, e_id=e_id)
                self.pack_id_to_num_stretches[(g_id, e_id)] = data.X.shape[0]
                for t_id in range(data.X.shape[0]):
                    self.all_pack_ids.add((g_id, e_id, t_id))
                num_success_samples += 1
                log.info(f"({g_id=}, {e_id=}) is valid for num stretches: {data.X.shape[0]}")
            except Exception as e:
                log.warning(f"Failed to touch mesh with: {g_id=}; {e_id=}. Error: {e}")
                continue

        self.all_pack_ids = sorted(list(self.all_pack_ids))

    def __len__(self) -> int:
        return len(self.all_pack_ids)

    def __getitem__(self, idx: int) -> PyTree:
        g_id, e_id, t_id = self.all_pack_ids[idx]

        description = np.array(
            [f"ExpQury, material id: <the unique one>, meshlist_id: ({g_id=}, {e_id=}, {t_id=})"],
            dtype=np.dtypes.StringDType(),
        )

        mesh = ide.DataMeshExp.load_by_id(data_dir=self.data_dir, g_id=g_id, e_id=e_id)
        mesh = _get_pytree_from_mesh(mesh)
        mesh = ptu.get_one_sample(mesh, t_id, keep_dim=True)

        return {
            "dataset": "ExpDataset",
            "description": description,
            "qury": [mesh],
            "scale_idxs": np.array([0]),
        }


class ExpTestDataset(Dataset):  #! only for valid_10
    def __init__(
        self,
        prpt_dataset: ExpPrpt,
        qury_dataset: ExpQury,
    ) -> None:
        super().__init__()
        self.prpt_dataset = prpt_dataset
        self.qury_dataset = qury_dataset

    def __len__(self) -> int:
        return len(self.qury_dataset)

    def __getitem__(self, idx: int):
        qury = self.qury_dataset[idx]
        prpt = self.prpt_dataset.retrieve(idx, src_len=len(self.qury_dataset), qury_XI=qury["qury"][0]["XI"])

        query_meshlist_id = qury["description"][0].split("meshlist_id: ")[1]

        qury_description = np.array(
            [f"ExpTestList, material id: 0, meshlist_id: {query_meshlist_id}"],
            dtype=np.dtypes.StringDType(),
        )

        return {
            "dataset": "ExpTestDataset",
            "description": qury_description,
            "prpt": prpt,
            "qury": qury["qury"],
            "scale_idxs": [0],
        }
