from itertools import product

import numpy as np
import torch
from optree import PyTree
from torch.utils.data import Dataset

from src.datasets import ice_data_multi_utils as idu
from src.datasets import pytree_utils as ptu
from src.utils import RankedLogger

from .dataset_base import IceBaseDataset
from .utils import (
    ForceTypeRawCfg,
    GeometryRawCfg,
)

log = RankedLogger(__name__, rank_zero_only=True)

PackIdType = tuple[str, str, int]
# g_id, f_id, t_id


class MeshListDataset(IceBaseDataset):
    def __init__(
        self,
        data_dir: str,
        geometry_cfg: GeometryRawCfg,  # all available geometries
        material_cfg: str,  # will be eval()ed
        stretch_cfg: str,  # will be eval()ed
        force_type_cfg: ForceTypeRawCfg,  # all available force types
        num_geometries_in_pool: str,  # will be eval()ed
        num_stretches_in_pool: str,  # will be eval()ed
        num_force_types_in_pool: str,  # will be eval()ed
        num_meshes_from_pool: str,  # will be eval()ed
        base_seed: int,  # the seed for random number generator
        # deterministic: bool = False,
        mesh_idxs: str | None = None,
    ) -> None:
        super().__init__(
            data_dir=data_dir,
            geometry_cfg=geometry_cfg,
            material_cfg=material_cfg,
            stretch_cfg=stretch_cfg,
            force_type_cfg=force_type_cfg,
            num_geometries_in_pool=num_geometries_in_pool,
            num_stretches_in_pool=num_stretches_in_pool,
            num_force_types_in_pool=num_force_types_in_pool,
            num_meshes_from_pool=num_meshes_from_pool,
            # deterministic=deterministic,
            mesh_idxs=mesh_idxs,
        )

        self.dataset_name = f"MeshList Dataset, {self.dataset_name}, "

        # we will use seed to enforce a very strong deterministic validation
        # same randomness even for different number of devices
        self.base_seed = base_seed

        # be careful, this may cache the data in disk if not exist
        # therefore might have multi-process issue
        # be sure the instantiate the dataset on prepare_data() of lightning datamodule
        m_id_to_all_pack_ids: dict[int, set[PackIdType]] = {}

        num_success_samples = 0
        num_total_samples = 0
        for g_id, f_id, m_id in product(
            self.geometry_ids,
            self.force_type_ids,
            self.material_ids,
        ):
            num_total_samples += 1
            try:
                idu.DataMesh.touch_by_id(data_dir=self.data_dir, g_id=g_id, f_id=f_id, m_id=m_id)
                if m_id not in m_id_to_all_pack_ids:
                    m_id_to_all_pack_ids[m_id] = set()
                for t_id in self.stretch_ids:
                    m_id_to_all_pack_ids[m_id].add((g_id, f_id, t_id))
                num_success_samples += 1
            except Exception:
                # print(f"Failed to touch mesh with: {g_id=}; {f_id=}; {m_id=}")
                continue

        log.info(f"{num_success_samples}/{num_total_samples} success samples for\n{self.data_dir}")

        self.m_id_to_all_pack_ids = {k: sorted(list(v)) for k, v in m_id_to_all_pack_ids.items()}

        self.effective_m_ids = list(self.m_id_to_all_pack_ids.keys())

    def _get_pytree_from_mesh(self, mesh: idu.DataMesh) -> PyTree:
        # add A and XI to IFdata
        for boundary in mesh.IFdata:
            boundary["A"] = idu.slice_nodes_set(mesh.A, boundary["nodeset"])
            boundary["XI"] = idu.slice_nodes_set(mesh.XI, boundary["nodeset"])
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

    def __len__(self):
        return len(self.effective_m_ids)

    def __getitem__(self, idx: int) -> PyTree:
        m_id = self.effective_m_ids[idx]

        rng = torch.Generator()
        rng.manual_seed(self.base_seed + idx)  # randomness is only from the seed and idx

        pack_ids = self._subsample_pack_ids(self.m_id_to_all_pack_ids[m_id], rng=rng)

        description = np.array(
            [f"DataMeshList, material id: {m_id}, meshlist_id: {pack_ids}"],
            dtype=np.dtypes.StringDType(),
        )

        mesh_list = []
        for pack_id in pack_ids:
            g_id, f_id, t_id = pack_id
            mesh = idu.DataMesh.load_by_id(data_dir=self.data_dir, g_id=g_id, f_id=f_id, m_id=m_id)
            mesh = self._get_pytree_from_mesh(mesh)
            mesh = ptu.get_one_sample(mesh, t_id, keep_dim=True)
            mesh_list.append(mesh)

        return {
            "dataset": "MeshListDataset",
            "description": description,
            "prpt": mesh_list,
            "qury": mesh_list[:1],
            "scale_idxs": np.array([0]),
        }


class ParallelMeshListDataset(Dataset):  #! only for valid_10
    def __init__(
        self,
        prompt_dataset: MeshListDataset,
        query_dataset: MeshListDataset,
        scale_idxs: str,
    ) -> None:
        super().__init__()
        self.prpt_dataset = prompt_dataset
        self.qury_dataset = query_dataset

        self.prpt_m_ids = prompt_dataset.effective_m_ids
        self.qury_m_ids = query_dataset.effective_m_ids

        self.m_ids = list(set(self.prpt_m_ids) & set(self.qury_m_ids))
        self.scale_idxs = np.asarray(eval(scale_idxs))

    def __len__(self) -> int:
        return len(self.m_ids)

    def __getitem__(self, idx: int):
        prpt_idx = self.prpt_m_ids.index(self.m_ids[idx])
        qury_idx = self.qury_m_ids.index(self.m_ids[idx])
        prpt = self.prpt_dataset[prpt_idx]
        qury = self.qury_dataset[qury_idx]

        query_meshlist_id = qury["description"][0].split("meshlist_id: ")[1]

        qury_description = np.array(
            [f"ParallelMeshList, material id: {self.m_ids[idx]}, meshlist_id: {query_meshlist_id}"],
            dtype=np.dtypes.StringDType(),
        )

        return {
            "dataset": "ParallelMeshListDataset",
            "description": qury_description,
            "prpt": prpt["prpt"],
            "qury": qury["prpt"],
            "scale_idxs": self.scale_idxs,
        }
