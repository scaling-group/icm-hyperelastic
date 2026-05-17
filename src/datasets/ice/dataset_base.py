from collections.abc import Sequence
from pathlib import Path
from typing import TypeVar

import torch
from torch.utils.data import Dataset

from .utils import (
    ForceTypeConfig,
    ForceTypeRawCfg,
    GeometryConfig,
    GeometryRawCfg,
)

T = TypeVar("T")


class IceBaseDataset(Dataset):
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
        # deterministic: bool,
        mesh_idxs: str | None = None,
    ) -> None:
        super().__init__()

        self.data_dir = Path(data_dir)

        self.material_cfg = material_cfg
        self.geometry_cfg = GeometryConfig(geometry_cfg)
        self.force_type_cfg = ForceTypeConfig(force_type_cfg)
        self.stretch_cfg = stretch_cfg

        self.material_ids: Sequence[int] = eval(self.material_cfg)
        self.geometry_ids = self.geometry_cfg.geometry_ids
        self.force_type_ids = self.force_type_cfg.force_type_ids
        self.stretch_ids: Sequence[int] = eval(self.stretch_cfg)

        self.num_geometries_in_pool = num_geometries_in_pool  # don't eval() for now
        self.num_force_types_in_pool = num_force_types_in_pool  # don't eval() for now
        self.num_stretches_in_pool = num_stretches_in_pool  # don't eval() for now
        self.num_meshes_from_pool = num_meshes_from_pool  # don't eval() for now

        # self.deterministic = deterministic
        self.mesh_idxs = mesh_idxs if mesh_idxs is None else eval(mesh_idxs)

        dataset_name = (
            f"material:{self.material_cfg}, "
            f"geometry:{self.geometry_cfg}, "
            f"force_type:{self.force_type_cfg}, "
            f"stretch:{self.stretch_cfg}, "
            f"num_geometries_in_pool:{self.num_geometries_in_pool}, "
            f"num_force_types_in_pool:{self.num_force_types_in_pool}, "
            f"num_stretches_in_pool:{self.num_stretches_in_pool}, "
            f"num_meshes_from_pool:{self.num_meshes_from_pool}, "
        )
        self.dataset_name = dataset_name

    def _subsample_pack_ids(self, all_pack_ids: list[T], rng: torch.Generator | None = None) -> list[T]:
        num_geometries_in_pool = eval(self.num_geometries_in_pool)
        num_force_types_in_pool = eval(self.num_force_types_in_pool)
        num_stretches_in_pool = eval(self.num_stretches_in_pool)
        num_meshes_from_pool = eval(self.num_meshes_from_pool)

        assert num_geometries_in_pool <= len(self.geometry_ids)
        assert num_force_types_in_pool <= len(self.force_type_ids)
        assert num_stretches_in_pool <= len(self.stretch_ids)
        num_meshes_from_pool = min(
            num_meshes_from_pool, num_geometries_in_pool * num_force_types_in_pool * num_stretches_in_pool
        )
        # assert num_meshes_from_pool <= num_geometries_in_pool * num_force_types_in_pool * num_stretches_in_pool

        # create a pool for sampling meshes, size: num_geometries_in_pool * num_stretches_in_pool

        # put pack_id one by one into pool
        g_id_pool = set()
        f_id_pool = set()
        t_id_pool = set()
        pool = []

        if self.mesh_idxs is None:
            p_ids = torch.randperm(len(all_pack_ids), generator=rng).tolist()
        else:
            p_ids = [list(range(len(all_pack_ids)))[idx] for idx in self.mesh_idxs]

        for p_id in p_ids:
            g_id, f_id, t_id = all_pack_ids[p_id]
            if (
                (len(g_id_pool) < num_geometries_in_pool or g_id in g_id_pool)
                and (len(f_id_pool) < num_force_types_in_pool or f_id in f_id_pool)
                and (len(t_id_pool) < num_stretches_in_pool or t_id in t_id_pool)
            ):
                pool.append((g_id, f_id, t_id))
                g_id_pool.add(g_id)
                f_id_pool.add(f_id)
                t_id_pool.add(t_id)
            if len(pool) == num_meshes_from_pool:
                break

        return pool
