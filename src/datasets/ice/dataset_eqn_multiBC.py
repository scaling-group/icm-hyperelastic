import os
from itertools import product
from pathlib import Path

import h5py
import numpy as np
import torch
from optree import PyTree

from src.datasets import ice_data_multi_utils as idu
from src.datasets import pytree_utils as ptu
from src.utils import RankedLogger

from .dataset_base import IceBaseDataset
from .utils import (
    ForceTypeRawCfg,
    GeometryRawCfg,
)

log = RankedLogger(__name__, rank_zero_only=True)

SnapshotPackIdType = tuple[str, str, int]
# g_id, f_id, t_id


class SnapshotPack:
    """A h5py pack consisting of all the snapshots with a specific `g_id`, `f_id`, and `t_id`."""

    def __init__(self, data_dir: str, cache_dir: str, pack_id: SnapshotPackIdType, mode: str) -> None:
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)  # cache dir relative to data_dir
        self.g_id, self.f_id, self.t_id = pack_id
        self.mode = mode

        #! bottleneck
        self.create_cache()

        # Keep file handle open
        self.file = h5py.File(self.cache_filepath, "r")
        self.valid_m_ids: list[int] = self.file["valid_m_ids"][()]

    def __del__(self):
        # Clean up file handle when object is destroyed
        if hasattr(self, "file"):
            self.file.close()

    def __len__(self):
        return len(self.valid_m_ids)

    def create_cache(self):
        try:
            f = h5py.File(self.cache_filepath, "r")
            _ = f["valid_m_ids"][()]
            log.debug(f"Cache file {self.cache_filepath} already exists, skipping creation.")
            return
        except Exception:
            pass

        with h5py.File(self.cache_filepath, "w") as f:
            valid_m_id_list = []
            for m_id in range(3100):  # TODO: pass all possible m_ids here
                try:
                    mesh = idu.DataMesh.load_by_id(
                        data_dir=self.data_dir,
                        g_id=self.g_id,
                        f_id=self.f_id,
                        m_id=m_id,
                    )
                    log.debug(f"success loading Material {m_id}")
                    valid_m_id_list.append(m_id)
                except Exception:
                    log.debug(f"error loading Material {m_id}")
                    continue

                #! slice out free nodes
                A = idu.slice_nodes_set(mesh.A[self.t_id], mesh.FREEset[self.t_id], with_batch_dim=False)
                XI = idu.slice_nodes_set(mesh.XI[self.t_id], mesh.FREEset[self.t_id], with_batch_dim=False)
                mask = idu.slice_nodes_set(mesh.mask[self.t_id], mesh.FREEset[self.t_id], with_batch_dim=False)
                group = f.create_group(f"Material-{m_id}")
                group.create_dataset("A", data=A)  # (n_free_node, N, 2, 2), N ~ 10
                group.create_dataset("XI", data=XI)  # (n_free_node, N, 2)
                group.create_dataset("mask", data=mask)  # (n_free_node, N)

                if "boundary" not in self.mode:
                    continue
                # cache boundary data
                n_bc_node_max = 30
                boundary_A = []
                boundary_XI = []
                for boundary in mesh.IFdata:
                    # pad [n_bc_node, N, 2, 2] -> [n_bc_node_max, N, 2, 2]
                    this_A = idu.slice_nodes_set(
                        mesh.A[self.t_id], boundary["nodeset"][self.t_id], with_batch_dim=False
                    )
                    n_bc_node = this_A.shape[0]
                    this_A = np.pad(this_A, ((0, n_bc_node_max - n_bc_node), (0, 0), (0, 0), (0, 0)))
                    boundary_A.append(this_A)

                    # pad [n_bc_node, N, 2] -> [n_bc_node_max, N, 2]
                    this_XI = idu.slice_nodes_set(
                        mesh.XI[self.t_id], boundary["nodeset"][self.t_id], with_batch_dim=False
                    )
                    this_XI = np.pad(this_XI, ((0, n_bc_node_max - n_bc_node), (0, 0), (0, 0)))
                    boundary_XI.append(this_XI)

                boundary_direction = [b["direction"][self.t_id] for b in mesh.IFdata]
                boundary_force = [b["force"][self.t_id] for b in mesh.IFdata]

                boundary_num = len(mesh.IFdata)
                group.create_dataset("boundary_num", data=boundary_num)  # ()
                for i in range(boundary_num):
                    group.create_dataset(f"boundary_A-{i}", data=boundary_A[i])  # (n_bc_node_max, N, 2, 2)
                    group.create_dataset(f"boundary_XI-{i}", data=boundary_XI[i])  # (n_bc_node_max, N, 2)
                    group.create_dataset(f"boundary_direction-{i}", data=boundary_direction[i])  # (2,)
                    group.create_dataset(f"boundary_force-{i}", data=boundary_force[i])  # ()

            f.create_dataset("geometry_id", data=self.g_id)
            f.create_dataset("stretch_id", data=self.t_id)
            f.create_dataset("force_id", data=self.f_id)
            f.create_dataset("valid_m_ids", data=np.array(valid_m_id_list, dtype=np.int32))

    @property
    def cache_filepath(self) -> Path:
        cache_dir = self.data_dir / self.cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        return cache_dir / f"geometry-{self.g_id}_force-{self.f_id}_stretch-{self.t_id}.h5"

    def __getitem__(self, idx) -> PyTree:
        group = self.file[f"Material-{idx}"]
        data = {
            "description": f"{self.g_id}-{self.f_id}-{self.t_id}-{idx},",
            "data": {
                "A": torch.from_numpy(group["A"][:][None, ...]).float(),  # (1, n_free_node, N, 2, 2)
                "XI": torch.from_numpy(group["XI"][:][None, ...]).float(),  # (1, n_free_node, N, 2)
                "mask": torch.from_numpy(group["mask"][:][None, ...]).bool(),  # (1, n_free_node, N)
            },
        }

        if "boundary" not in self.mode:
            return data

        boundary_num = group["boundary_num"][()]
        data["data_boundary"] = [
            {
                "A": torch.from_numpy(group[f"boundary_A-{i}"][:][None, ...]).float(),  # (1, n_bc_node_max, N, 2, 2)
                "XI": torch.from_numpy(group[f"boundary_XI-{i}"][:][None, ...]).float(),  # (1, n_bc_node_max, N, 2)
                "direction": torch.from_numpy(group[f"boundary_direction-{i}"][:][None, ...]).float(),  # (1, 2)
                "force": torch.tensor((group[f"boundary_force-{i}"][()],), dtype=torch.float32),  # (1,)
            }
            for i in range(boundary_num)
        ]
        return data


class EquationDatasetMultiBC(IceBaseDataset):
    def __init__(
        self,
        data_dir: str,
        cache_dir: str,
        geometry_cfg: GeometryRawCfg,  # all available geometries
        material_cfg: str,  # will be eval()ed
        stretch_cfg: str,  # will be eval()ed
        force_type_cfg: ForceTypeRawCfg,  # all available force types
        num_geometries_in_pool: str,  # will be eval()ed
        num_stretches_in_pool: str,  # will be eval()ed
        num_force_types_in_pool: str,  # will be eval()ed
        num_meshes_from_pool: str,  # will be eval()ed
        mode: str,  # mode for return
        num_eqns: int | None = None,  # for pre-slicing before collate
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
        )

        self.mode = mode
        self.num_eqns = num_eqns
        self.dataset_name = f"Eqn Dataset, {self.dataset_name}, num_eqns:{self.num_eqns}, "
        # be careful, this may cache the data in disk if not exist
        # therefore might have multi-process issue
        # be sure the instantiate the dataset on prepare_data() of lightning datamodule
        self.pack_id_to_snapshot: dict[SnapshotPackIdType, SnapshotPack] = {}
        m_id_to_all_pack_ids: dict[int, set[SnapshotPackIdType]] = {}

        for pack_id in product(
            self.geometry_ids,
            self.force_type_ids,
            self.stretch_ids,
        ):
            snapshot_pack = SnapshotPack(data_dir, cache_dir, pack_id, mode=self.mode)
            for m_id in snapshot_pack.valid_m_ids:
                if m_id not in m_id_to_all_pack_ids:
                    m_id_to_all_pack_ids[m_id] = {pack_id}
                else:
                    m_id_to_all_pack_ids[m_id].add(pack_id)
            if len(snapshot_pack) > 0:
                log.info(f"length = {len(snapshot_pack)} for {pack_id}")
                self.pack_id_to_snapshot[pack_id] = snapshot_pack
            else:
                log.warning(f"length = 0 for {pack_id}, skipped!")

        self.m_id_to_all_pack_ids = {k: sorted(list(v)) for k, v in m_id_to_all_pack_ids.items()}

        self.effective_m_ids = list(self.m_id_to_all_pack_ids.keys())

        if self.num_eqns is not None:
            raise NotImplementedError("slicing num_eqns inside dataset is not implemented yet")

    def __len__(self):
        return len(self.effective_m_ids)

    def __getitem__(self, idx: int) -> PyTree:
        m_id = self.effective_m_ids[idx]

        pack_ids = self._subsample_pack_ids(self.m_id_to_all_pack_ids[m_id])

        # sample num_meshes_from_pool meshes from the pool
        snapshots = [self.pack_id_to_snapshot[pack_id][m_id] for pack_id in pack_ids]

        data = ptu.concat([snapshot["data"] for snapshot in snapshots], dim=1)
        description = np.array(
            [
                (
                    f"{self.dataset_name}, "
                    f"{len(snapshots)} snapshots: "
                    f"{' '.join([snapshot['description'] for snapshot in snapshots])}"
                )
            ],
            dtype=np.dtypes.StringDType(),
        )

        batch = {
            "description": description,
            "data": data,
        }

        if "boundary" in self.mode:
            # merge into one list
            all_boundary_data = [bc_data for snapshot in snapshots for bc_data in snapshot["data_boundary"]]
            # random select 2
            data_boundary = [all_boundary_data[i] for i in torch.randint(0, len(all_boundary_data), (2,))]
            batch["data_boundary"] = data_boundary

        return batch
