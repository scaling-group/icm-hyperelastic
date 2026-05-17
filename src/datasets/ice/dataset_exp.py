import json
from collections.abc import Sequence
from itertools import product
from pathlib import Path

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

PackIdType = tuple[str, str, int]
# g_id, e_id, t_id


class ExpDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        geometry_cfg: GeometryConfig,  # all available geometries
        experiment_cfg: ExperimentConfig,  # all available experiments
        stretch_cfg: str,  # will be eval()ed
        num_geometries_in_pool: str,  # will be eval()ed
        num_experiment_in_pool: str,  # will be eval()ed
        num_stretches_in_pool: str,  # will be eval()ed
        num_meshes_from_pool: str,  # will be eval()ed
        base_seed: int,  # the seed for random number generator
        # deterministic: bool = False,
        mesh_idxs: str | None = None,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)

        # self.material_cfg = material_cfg
        self.geometry_cfg = geometry_cfg
        self.experiment_cfg = experiment_cfg
        self.stretch_cfg = stretch_cfg

        # self.material_ids: Sequence[int] = eval(self.material_cfg) #! unique for now
        self.geometry_ids = self.geometry_cfg.geometry_ids
        # self.force_type_ids = self.force_type_cfg.force_type_ids
        self.experiment_ids = self.experiment_cfg.experiment_ids
        self.stretch_ids: Sequence[int] = eval(self.stretch_cfg)

        self.num_geometries_in_pool = num_geometries_in_pool  # don't eval() for now
        self.num_experiment_in_pool = num_experiment_in_pool  # don't eval() for now
        self.num_stretches_in_pool = num_stretches_in_pool  # don't eval() for now
        self.num_meshes_from_pool = num_meshes_from_pool  # don't eval() for now

        # self.deterministic = deterministic
        self.mesh_idxs = mesh_idxs if mesh_idxs is None else eval(mesh_idxs)

        dataset_name = (
            # f"material:{self.material_cfg}, "
            f"geometry:{self.geometry_cfg}, "
            f"experiment:{self.experiment_cfg}, "
            f"stretch:{self.stretch_cfg}, "
            f"num_geometries_in_pool:{self.num_geometries_in_pool}, "
            f"num_experiment_in_pool:{self.num_experiment_in_pool}, "
            f"num_stretches_in_pool:{self.num_stretches_in_pool}, "
            f"num_meshes_from_pool:{self.num_meshes_from_pool}, "
            f"seed:{base_seed}, "
        )
        self.dataset_name = f"Exp, {dataset_name}, "

        # we will use seed to enforce a very strong deterministic validation
        # same randomness even for different number of devices
        self.base_seed = base_seed

        # be careful, this may cache the data in disk if not exist
        # therefore might have multi-process issue
        # be sure the instantiate the dataset on prepare_data() of lightning datamodule
        all_pack_ids: set[PackIdType] = set()
        # m_id_to_all_pack_ids: dict[int, set[PackIdType]] = {}

        num_success_samples = 0
        num_total_samples = 0
        for g_id, e_id in product(
            self.geometry_ids,
            self.experiment_ids,
        ):
            num_total_samples += 1
            try:
                ide.DataMeshExp.touch_by_id(data_dir=self.data_dir, g_id=g_id, e_id=e_id)
                for t_id in self.stretch_ids:
                    all_pack_ids.add((g_id, e_id, t_id))
                num_success_samples += 1
            except Exception as e:
                print(f"Failed to touch mesh with: {g_id=}; {e_id=}. Error: {e}")
                continue

        log.info(f"{num_success_samples}/{num_total_samples} success samples for\n{self.data_dir}")

        self.all_pack_ids = sorted(all_pack_ids)
        self.effective_m_ids = [0]

    def _get_pytree_from_mesh(self, mesh: ide.DataMeshExp) -> PyTree:
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

    def _subsample_pack_ids(self, rng: torch.Generator | None = None):
        num_geometries_in_pool = eval(self.num_geometries_in_pool)
        num_experiment_in_pool = eval(self.num_experiment_in_pool)
        num_stretches_in_pool = eval(self.num_stretches_in_pool)
        num_meshes_from_pool = eval(self.num_meshes_from_pool)

        assert num_geometries_in_pool <= len(self.geometry_ids)
        assert num_stretches_in_pool <= len(self.stretch_ids)

        # put pack_id one by one into pool
        g_id_pool = set()
        e_id_pool = set()
        t_id_pool = set()
        pool = []

        if self.mesh_idxs is None:
            p_ids = torch.randperm(len(self.all_pack_ids), generator=rng).tolist()
        else:
            p_ids = [list(range(len(self.all_pack_ids)))[idx] for idx in self.mesh_idxs]

        for p_id in p_ids:
            g_id, e_id, t_id = self.all_pack_ids[p_id]
            if (
                (len(g_id_pool) < num_geometries_in_pool or g_id in g_id_pool)
                and (len(e_id_pool) < num_experiment_in_pool or e_id in e_id_pool)
                and (len(t_id_pool) < num_stretches_in_pool or t_id in t_id_pool)
            ):
                pool.append((g_id, e_id, t_id))
                g_id_pool.add(g_id)
                e_id_pool.add(e_id)
                t_id_pool.add(t_id)
            if len(pool) == num_meshes_from_pool:
                break

        return pool

    def __len__(self):
        return 1  #! unique material

    def __getitem__(self, idx: int) -> PyTree:
        description = np.array(
            ["DataMeshList, material id: <the unique one>, meshlist_id: from ExpDataset"],
            dtype=np.dtypes.StringDType(),
        )

        rng = torch.Generator().manual_seed(self.base_seed)
        pack_ids = self._subsample_pack_ids(rng=rng)

        self._save_prompt_id(pack_ids, self.dataset_name)

        mesh_list = []
        for pack_id in pack_ids:
            g_id, e_id, t_id = pack_id
            mesh = ide.DataMeshExp.load_by_id(data_dir=self.data_dir, g_id=g_id, e_id=e_id)
            mesh = self._get_pytree_from_mesh(mesh)
            mesh = ptu.get_one_sample(mesh, t_id, keep_dim=True)
            mesh_list.append(mesh)

        return {
            "dataset": "ExpDataset",
            "description": description,
            "prpt": mesh_list,
            "qury": mesh_list[:1],
            "scale_idxs": np.array([0]),
        }

    def _save_prompt_id(self, pack_ids, dataset_name):
        prpt_dirpath = Path(self.data_dir) / ("prompts_ids_" + dataset_name)
        prpt_dirpath.mkdir(parents=True, exist_ok=True)
        full_path = prpt_dirpath / "prompt.json"
        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(list(pack_ids), f, indent=2)
