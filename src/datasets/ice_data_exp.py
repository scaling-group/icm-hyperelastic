from pathlib import Path

import numpy as np

from .field_data import FieldData_tri_multiBC
from .ice_data_multi_utils import (
    DataMesh,
    IFdataBCType,
    slice_nodes_set,  # noqa: F401
)


class DataMeshExp(DataMesh):
    @classmethod
    def load_from_raw(cls, rawpath: Path):
        fd = FieldData_tri_multiBC(rawpath)
        IFdata: list[IFdataBCType] = [
            {
                "nodeset": np.array([data["nodeset"]] * fd.N_t, dtype=np.int32),
                "direction": np.array([data["direction"]] * fd.N_t, dtype=np.float32),
                "force": np.array(data["force"], dtype=np.float32),
            }
            for data in fd.IFdata
        ]
        return cls(
            rawpath=[rawpath] * fd.N_t,
            problem=[str(rawpath) + f"_t{i}" for i in range(fd.N_t)],
            param=[None] * fd.N_t,
            mesh=np.array([fd.mesh] * fd.N_t, dtype=np.int32),
            X=np.array([fd.X] * fd.N_t, dtype=np.float32),
            U=np.array(fd.U, dtype=np.float32),
            BCset=np.array([fd.BCset] * fd.N_t, dtype=np.int32),
            FREEset=np.array([fd.FREEset] * fd.N_t, dtype=np.int32),
            Xip=np.array([fd.Xip] * fd.N_t, dtype=np.float32),
            B=np.array([fd.B] * fd.N_t, dtype=np.float32),
            Bcoeff=np.array([fd.Bcoeff] * fd.N_t, dtype=np.float32),
            F=np.array(fd.F, dtype=np.float32),
            P=np.array(fd.stress_P, dtype=np.float32),
            S=np.array(fd.stress_S, dtype=np.float32),
            IFdata=IFdata,
        )

    @classmethod
    def touch_by_id(cls, data_dir: Path, g_id: str, e_id: int):
        """
        just touch the cache file, if not exists, load from raw and save to cache
        """
        filepath = cls.cache_filepath(data_dir, g_id, e_id)
        rawpath = cls.raw_filepath(data_dir, g_id, e_id)
        if not filepath.exists():
            data = cls.load_from_raw(rawpath)
            data.save(filepath)

    @classmethod
    def load_by_id(cls, data_dir: Path, g_id: str, e_id: int):
        return cls.load(
            filepath=cls.cache_filepath(data_dir, g_id, e_id),
            rawpath=cls.raw_filepath(data_dir, g_id, e_id),
        )

    @staticmethod
    def cache_filepath(data_dir: Path, g_id: str, e_id: int) -> Path:
        return data_dir / "cache" / f"geometry-{g_id}" / f"Exp-{e_id}.npy"

    @staticmethod
    def raw_filepath(data_dir: Path, g_id: str, e_id: int) -> Path:
        return data_dir / f"{g_id}-{e_id}"
