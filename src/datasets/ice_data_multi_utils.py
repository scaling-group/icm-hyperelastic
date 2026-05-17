import os
import pickle
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypedDict, TypeVar

import numpy as np
import torch
from numpy.typing import NDArray
from optree import PyTree

from src.datasets import pytree_utils as ptu

from .field_data import FieldData_tri, FieldData_tri_multiBC

T = TypeVar("T", bound="BaseData")
DataMeshList = PyTree


def slice_nodes_set(
    nodal_value: torch.Tensor | NDArray,
    set_idx: torch.Tensor | NDArray,
    with_batch_dim: bool = True,
) -> torch.Tensor | NDArray:
    """
    if with_batch_dim:
        nodal_value: (bs, n_node, ...)
        set_idx: (bs, n_set)
        return: (bs, n_set, ...)
    else:
        nodal_value: (n_node, ...)
        set_idx: (n_set,)
        return: (n_set, ...)
    """
    if isinstance(nodal_value, torch.Tensor):
        if with_batch_dim:
            batch_idx = torch.arange(nodal_value.shape[0]).unsqueeze(1)  # shape: (bs, 1)
            nodal_value_set = nodal_value[batch_idx, set_idx, ...]  # (bs, n_set, ...)
        else:
            nodal_value_set = nodal_value[set_idx, ...]  # (n_set, ...)
    elif isinstance(nodal_value, np.ndarray):
        if with_batch_dim:
            batch_idx = np.arange(nodal_value.shape[0]).reshape(-1, 1)  # shape: (bs, 1)
            nodal_value_set = nodal_value[batch_idx, set_idx, ...]  # (bs, n_set, ...)
        else:
            nodal_value_set = nodal_value[set_idx, ...]  # (n_set, ...)
    else:
        raise ValueError(f"Unknown type: {type(nodal_value)}")
    return nodal_value_set


class IFdataBCType(TypedDict):
    nodeset: NDArray[np.int32]  # (bs, N_node,)
    # [65,  67,  80,  81,  82,  83, 395, 396, 397, 398, 399, 944, 945, 946, 947, 948, 949, 950, 951]
    direction: NDArray[np.float32]  # (bs, 2,)
    # [0., 1.]
    force: NDArray[np.float32]  # (bs,)
    # [0.        , 0.00417688, 0.00846787, 0.01288612, 0.01744267, 0.0221472, 0.02700807, 0.03203239,
    # 0.03722609, 0.042594, 0.04814002]


class BaseData:
    def _apply(self: T, func: Callable[[str, Any], Any], *args, **kwargs) -> T:
        """
        Helper method to apply a function to each attribute of the instance.
        Creates a new instance of the same type and assigns the transformed values.
        """
        new_obj = type(self)(**{k: None for k in self.__dict__})
        # in case some fields have no defaults
        for attr, value in self.__dict__.items():
            setattr(new_obj, attr, func(attr, value, *args, **kwargs))
        return new_obj

    def to(self: T, device: torch.device, *args, **kwargs) -> T:
        """
        Return a new instance with each torch.Tensor moved to the specified device.
        """
        return self._apply(lambda attr, v: v.to(device, *args, **kwargs) if isinstance(v, torch.Tensor) else v)

    def to_numpy(self: T) -> T:
        """
        Return a new instance with torch.Tensors converted to numpy arrays.
        """
        return self._apply(lambda attr, v: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v)

    def to_tensor(self: T, *args, **kwargs) -> T:
        """
        Return a new instance with numpy arrays converted to torch.Tensors.
        """
        return self._apply(lambda attr, v: torch.tensor(v, *args, **kwargs) if isinstance(v, np.ndarray) else v)

    def get_one_batch(self: T, bid: int, keep_dim=False) -> T:
        """
        return new data with bid-th batch, can keep batch dim or not
        """
        new_data = type(self)(**{k: None for k in self.__dict__})
        for attr, value in self.__dict__.items():
            if isinstance(value, torch.Tensor | np.ndarray):
                new_value = value[bid : bid + 1] if keep_dim else value[bid]
                setattr(new_data, attr, new_value)
            elif isinstance(value, list | tuple):
                new_value = [value[bid]] if keep_dim else value[bid]
                setattr(new_data, attr, new_value)
            else:
                setattr(new_data, attr, value)
        return new_data

    def get_slice_batch(self: T, bid_list: Sequence[int]) -> T:
        """
        return new data with bid-th batch, can keep batch dim or not
        """
        new_data = type(self)(**{k: None for k in self.__dict__})
        for attr, value in self.__dict__.items():
            if isinstance(value, torch.Tensor | np.ndarray):
                setattr(new_data, attr, value[bid_list])
            elif isinstance(value, list | tuple):
                setattr(new_data, attr, [value[i] for i in bid_list])
            else:
                setattr(new_data, attr, value)
        return new_data

    def get_shape(self):
        data_shape = type(self)(**{k: None for k in self.__dict__})
        for attr, value in self.__dict__.items():
            if isinstance(value, torch.Tensor | np.ndarray):
                setattr(data_shape, attr, value.shape)
            elif isinstance(value, list | tuple | str):
                setattr(data_shape, attr, len(value))
            else:
                setattr(data_shape, attr, None)
        return data_shape


@dataclass
class DataMesh(BaseData):
    # N_t: number of stretches
    # N_e: number of elements (triangles)
    # N_n: number of nodes
    # N_BC: number of boundary nodes
    # N_free: number of free nodes
    # N_ifx1: number of ifx1 nodes
    # N_ifx2: number of ifx2 nodes

    rawpath: list[Path]
    problem: list[str]  # type of material, tuple of strings
    param: list  # unknown, keep it None for now
    mesh: NDArray[np.int32]  # (N_t, N_e, r), (11, 3427, 3)
    X: NDArray[np.float32]  # (N_t, N_n, 2), (11, 1920, 2)
    U: NDArray[np.float32]  # (N_t, N_n, 2), (11, 1920, 2)
    BCset: NDArray[np.int32]  # (N_t, N_BC), [19, 20, 21, 22, 59, 60, ..., 128, 129, 130] (11, 66)
    FREEset: NDArray[np.int32]  # (N_t, N_free), (11, 1854)
    # IFX1set: np.ndarray = None  # (N_t, N_ifx1), [19, 21, 59, 60, 61, ..., 79, 80, 81, 82]
    # IFX2set: np.ndarray = None  # (N_t, N_ifx2), [20, 22, 107, 108, 109, ..., 128, 129, 130]
    # IFX: np.ndarray = None  # (N_t,), (11,)
    Xip: NDArray[np.float32]  # (N_t, N_e, 1, 2), (11, 3427, 1, 2)
    B: NDArray[np.float32]  # (N_t, N_e, 1, 3, 2), (11, 3427, 1, 3, 2)
    Bcoeff: NDArray[np.float32]  # (N_t, N_e, 1), (11, 3427, 1)
    F: NDArray[np.float32]  # (N_t, N_e, 1, 2, 2), (11, 3427, 1, 2, 2)
    P: NDArray[np.float32]  # (N_t, N_e, 1, 2, 2)
    S: NDArray[np.float32]  # (N_t, N_e, 1, 2, 2)

    IFdata: list[IFdataBCType]
    max_neighbors: int = 10  # maximum number of neighbors for each node

    _E: NDArray[np.float32] | None = None  # (N_t, N_e, 1, 2, 2)
    _C: NDArray[np.float32] | None = None  # (N_t, N_e, 1, 2, 2)
    _I: NDArray[np.float32] | None = None  # (N_t, N_e, 1, 2)
    _grad_I_grad_C: NDArray[np.float32] | None = None  # (N_t, N_e, 1, 2, 2, 2)

    class _LinearEquationType(TypedDict):
        A: NDArray[np.float32]  # (N_t, N_n, max_neighbors, 2, 2)
        XI: NDArray[np.float32]  # (N_t, N_n, max_neighbors, 2)
        mask: NDArray[np.bool_]  # (N_t, N_n, max_neighbors) [mask for A and XI]

    _linear_equation: _LinearEquationType | None = None  # (A, XI, mask)

    def save(self, filepath: Path) -> None:
        """
        Save all dataclass fields into a dictionary and then pickle it to 'filepath'.
        """
        os.makedirs(filepath.parent, exist_ok=True)
        self.cache_all()
        with open(filepath, "wb") as f:
            pickle.dump(asdict(self), f)

    @classmethod
    def load(cls, filepath: Path, rawpath: Path) -> "DataMesh":
        try:
            return cls.load_from_file(filepath)
        except FileNotFoundError:
            data = cls.load_from_raw(rawpath)
            data.save(filepath)
            return data

    @classmethod
    def load_from_file(cls, filepath: Path) -> "DataMesh":
        with open(filepath, "rb") as f:
            data_dict = pickle.load(f)
        return cls(**data_dict)

    @classmethod
    def load_from_raw(cls, rawpath: Path) -> "DataMesh":
        if "ICE-elastic" in str(rawpath):  # TODO: hack for multiBC
            fd = FieldData_tri_multiBC(rawpath)
            IFdata: list[IFdataBCType] = [
                {
                    "nodeset": np.array([data["nodeset"]] * fd.N_t, dtype=np.int32),
                    "direction": np.array([data["direction"]] * fd.N_t, dtype=np.float32),
                    "force": np.array(data["force"], dtype=np.float32),
                }
                for data in fd.IFdata
            ]
        else:  # TODO: hack for singleBC
            fd = FieldData_tri(rawpath)
            IFdata = [
                {
                    "nodeset": np.array([fd.IFX1set] * fd.N_t, dtype=np.int32),
                    "direction": np.array([[1.0, 0.0]] * fd.N_t, dtype=np.float32),
                    "force": np.array(fd.IFX, dtype=np.float32),
                },
                {
                    "nodeset": np.array([fd.IFX2set] * fd.N_t, dtype=np.int32),
                    "direction": np.array([[-1.0, 0.0]] * fd.N_t, dtype=np.float32),
                    "force": np.array(fd.IFX, dtype=np.float32),
                },
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
    def touch_by_id(cls, data_dir: Path, g_id: str, m_id: int, f_id: str = "default"):
        """
        just touch the cache file, if not exists, load from raw and save to cache
        """
        filepath = cls.cache_filepath(data_dir, g_id, f_id, m_id)
        rawpath = cls.raw_filepath(data_dir, g_id, f_id, m_id)
        if not filepath.exists():
            data = cls.load_from_raw(rawpath)
            data.save(filepath)

    @classmethod
    def load_by_id(cls, data_dir: Path, g_id: str, m_id: int, f_id: str = "default"):
        # TODO: use `f_id = "default"` for singleBC for hacking
        return cls.load(
            filepath=cls.cache_filepath(data_dir, g_id, f_id, m_id),
            rawpath=cls.raw_filepath(data_dir, g_id, f_id, m_id),
        )

    @staticmethod
    def cache_filepath(data_dir: Path, g_id: str, f_id: str, m_id: int) -> Path:
        if "ICE-elastic" in str(data_dir):  # TODO: hack for multiBC
            return data_dir / "cache" / f"Model-{g_id}" / f"{f_id}" / f"Poly-{m_id:05d}.npy"
        else:  # TODO: hack for singleBC
            return data_dir / "cache" / f"Model-{g_id}" / f"data-Model-{g_id}-PolyHyper-{m_id:05d}.npy"

    @staticmethod
    def raw_filepath(data_dir: Path, g_id: str, f_id: str, m_id: int) -> Path:
        if "ICE-elastic" in str(data_dir):  # TODO: hack for multiBC
            if "train" in str(data_dir):
                return data_dir / f"Model-{g_id}" / f"Model-{g_id}-{f_id}-Poly-{m_id:05d}"  # train-7z
            elif "valid-10" in str(data_dir):  # hacking for valid-10
                return data_dir / f"{f_id}-{m_id:05d}"
            else:
                return data_dir / f"Model-{g_id}" / f"Model-{g_id}-{f_id}-{m_id:05d}"  # test, val-3, val-4
        else:  # TODO: hack for singleBC
            return data_dir / f"Model-{g_id}" / f"data-Model-{g_id}-PolyHyper-{m_id:05d}"

    @property
    def E(self):
        """get_green_strain
        F: (..., 2, 2), the deformation gradient
        return green strain E: (..., 2, 2)
        E = 0.5 * (F^T @ F - I)
        """
        if self._E is None:
            F = self.F
            F_T = F.swapaxes(-2, -1)  # swap the last two dimensions
            self._E = 0.5 * (F_T @ F - np.eye(2, dtype=np.float32))  # broadcast I to the same shape as F

        return self._E

    @property
    def C(self):
        """right_cauchy_green_deformation
        C = F^T @ F
        """
        if self._C is None:
            F = self.F
            F_T = F.swapaxes(-2, -1)  # swap the last two dimensions
            self._C = F_T @ F  # broadcast I to the same shape as F

        return self._C

    @property
    def I(self):  # noqa: E743
        """principal_invariants
        2D principal invariants
        I1 = tr(C)
        I2 = det(C)
        """
        if self._I is None:
            C = self.C
            I1 = C[..., 0, 0] + C[..., 1, 1]
            I2 = C[..., 0, 0] * C[..., 1, 1] - C[..., 0, 1] * C[..., 1, 0]
            self._I = np.stack([I1, I2], axis=-1)  # (..., 2)
        return self._I

    @property
    def grad_I_grad_C(self):
        """
        C: (..., 2, 2), the right cauchy green deformation, numpy array
        return: (..., 2, 2, 2), the gradient of the principal invariants with respect to C, numpy array
        (i,j,k) = grad I_i / grad C_jk
        """
        if self._grad_I_grad_C is None:
            C = self.C
            grad_1 = np.zeros_like(C)
            grad_1[..., 0, 0] = 1
            grad_1[..., 1, 1] = 1  # identity matrix
            grad_2 = np.zeros_like(C)
            grad_2[..., 0, 0] = C[..., 1, 1]
            grad_2[..., 0, 1] = -C[..., 0, 1]
            grad_2[..., 1, 0] = -C[..., 1, 0]
            grad_2[..., 1, 1] = C[..., 0, 0]
            self._grad_I_grad_C = np.stack([grad_1, grad_2], axis=-3)  # (..., 2, 2, 2)
        return self._grad_I_grad_C

    @property
    def linear_equation(self):
        """
        build the input sequence that describes the snapshot of the system
        return a tuple (A, X, mask):
        A: (bs, N_n, 8, 2, 2), same type of data.X
        X: (bs, N_n, 8, 2), same type of data.X
        mask: (bs, N_n, 8), bool type
        units are the number of equilibrium units
        max_elements is the maximum number of elements in one unit. Set as 8 for now,
        i.e. at most 8 elements in one unit (should be enough)
        dim is the dimension of the equilibrium unit. Should be 8 as in our discussion.
        prompt contains the information of the equilibrium unit. Some elements are filled with 0 if not used.
        Mask is used to indicate whether the elements are used.
        """
        if self._linear_equation is None:
            # get shape
            bs, N_n, _ = self.X.shape
            N_e = self.B.shape[1]

            # P = F@S, S = 2 * sum_m grad_psi_grad_I_m * grad_I_m_grad_C
            # f = torch.einsum("bet,betnj,betij ->beni", Bcoeff, B, P) # (bs, ne, 3, 2)
            # Here I want to get the coefficients of f for grad_psi_grad_I_m

            # b: batch, e: element, t: t = 1, integral points,
            # n: n = 3, represent the contribution to three vertices of the element
            # m: the index of principal invariant
            # P = 2 * torch.einsum("betil, betm, betmlj ->betij", F, grad_psi_grad_I, grad_I_grad_C)
            # f = torch.einsum("bet,betnj,betij -> beni", Bcoeff, B, P)
            # so f = 2 * torch.einsum("bet,betnj,betil, betm, betmlj ->beni",
            # Bcoeff, B, F, grad_psi_grad_I, grad_I_grad_C)
            # so the coefficient of grad_psi_grad_I_m is
            # f_coeff = 2 * torch.einsum("bet, betnj, betil, betmlj ->betnim", Bcoeff, B, F, grad_I_grad_C)
            # so that f = torch.einsum("betnim, betm -> beni", f_coeff, grad_psi_grad_I)
            # Here t is the number of integral points in each element, t = 1 for linear elements
            # for simplicity, we drop t in f_coeff
            # then f = torch.einsum("benim, bem -> beni", f_coeff, grad_psi_grad_I)
            # Here e is the element, n is the vertex n = 1,2,3
            # f is the contribution of the nodal force from element e to it's three vertices

            f_coeff = 2 * np.einsum(
                "bet, betnj, betil, betmlj ->betnim", self.Bcoeff, self.B, self.F, self.grad_I_grad_C
            )  # (bs, ne, 1, 3, 2, 2)
            f_coeff = f_coeff[:, :, 0, :, :, :]  # (bs, ne, 3, 2, 2)

            # Now we need to gather the coefficients in each equation, or each node (including the boundary nodes)
            # initialize A, X, mask
            A = np.zeros([bs, N_n, self.max_neighbors, 2, 2], dtype=self.X.dtype)
            XI = np.zeros([bs, N_n, self.max_neighbors, 2], dtype=self.X.dtype)
            mask = np.zeros([bs, N_n, self.max_neighbors], dtype=bool)
            # just point to the next position to fill
            pointer = np.zeros([bs, N_n], dtype=np.int64)

            for i in range(bs):
                for j in range(N_e):
                    for k in range(3):
                        node_idx = self.mesh[i, j, k]
                        pos = pointer[i, node_idx]
                        A[i, node_idx, pos] = f_coeff[i, j, k, :, :]
                        XI[i, node_idx, pos] = self.I[i, j, 0, :]
                        mask[i, node_idx, pos] = 1
                        pointer[i, node_idx] += 1

            self._linear_equation = {
                "A": A,
                "XI": XI,
                "mask": mask,
            }

        return self._linear_equation

    @property
    def A(self):
        return self.linear_equation["A"]

    @property
    def XI(self):
        return self.linear_equation["XI"]

    @property
    def mask(self):
        return self.linear_equation["mask"]

    def cache_all(self):
        _ = self.E, self.C, self.I, self.grad_I_grad_C, self.linear_equation
        _ = self.A, self.XI, self.mask


class DataEqn(TypedDict):
    A: torch.Tensor | NDArray[np.float32]
    XI: torch.Tensor | NDArray[np.float32]
    mask: torch.Tensor | NDArray[np.float32]


def to_eqn(mesh_list: DataMeshList) -> DataEqn:
    data_eqn = [
        {
            "A": slice_nodes_set(mesh["A"], mesh["FREEset"]),
            "XI": slice_nodes_set(mesh["XI"], mesh["FREEset"]),
            "mask": slice_nodes_set(mesh["mask"], mesh["FREEset"]),
        }
        for mesh in mesh_list
    ]
    data_eqn = ptu.concat(data_eqn, dim=1)
    return data_eqn


if __name__ == "__main__":
    # Test the DataMesh class
    data_dir = Path("/home/xxx/MyProjects/ice-dev/data/ICE-elastic/dataset-2504-sample")
    # Model-H-shear-Poly-01300
    g_id = "H"
    f_id = "shear"
    m_id = 1300

    data_mesh = DataMesh.load_by_id(data_dir, g_id, f_id, m_id)
    data_mesh.cache_all()

    for field in data_mesh.__dataclass_fields__:
        print(field)
        param = getattr(data_mesh, field)
        if isinstance(param, tuple):
            for item in param:
                if isinstance(item, np.ndarray):
                    print(item.shape)
                else:
                    print(item, type(item))
        elif isinstance(param, np.ndarray):
            print(param.shape, param.dtype)
        else:
            print(param, type(param))

    # print(data_mesh)
