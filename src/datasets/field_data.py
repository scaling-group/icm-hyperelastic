import json
from pathlib import Path

import numpy as np


def make_C3(x):
    Xip = x.mean(0)

    a = (
        x[0, 0] * x[1, 1]
        - x[0, 1] * x[1, 0]
        + x[1, 0] * x[2, 1]
        - x[2, 0] * x[1, 1]
        + x[2, 0] * x[0, 1]
        - x[2, 1] * x[0, 0]
    )

    B = np.zeros((3, 2))
    B[0, 0] = x[1, 1] - x[2, 1]
    B[0, 1] = x[2, 0] - x[1, 0]
    B[1, 0] = x[2, 1] - x[0, 1]
    B[1, 1] = x[0, 0] - x[2, 0]
    B[2, 0] = x[0, 1] - x[1, 1]
    B[2, 1] = x[1, 0] - x[0, 0]

    B /= a

    Bcoeff = np.array([abs(a) / 2])

    return Xip.reshape(1, 2), Bcoeff, B.reshape(1, 3, 2)


class FieldData_tri:
    #! DO NOT MODIFY THIS !!!
    # N_t: number of stretches
    # N_e: number of elements (triangles)
    # N_n: number of nodes
    # N_BC: number of boundary nodes
    # N_free: number of free nodes
    # N_ifx1: number of ifx1 nodes
    # N_ifx2: number of ifx2 nodes

    # X: (N_n, 2)
    # U: (N_t, N_n, 2)
    # mesh: (N_e, 3)
    # BCset: (N_BC,)
    # FREEset: (N_free,)
    # IFX1set: (N_ifx1,)
    # IFX2set: (N_ifx2,)
    # IFX: (N_t,)

    # Xip: (N_e, 1, 2)
    # B: (N_e, 1, 3, 2)
    # Bcoeff: (N_e, 1)
    # F: (N_t, N_e, 1, 2, 2)
    # P: (N_t, N_e, 1, 2, 2)
    # S: (N_t, N_e, 1, 2, 2)
    # lam3: (N_t, N_e)

    def __init__(self, filepath: Path):
        self.mesh = None
        with open(filepath / "mesh.json") as f:
            self.mesh = np.array(json.load(f))

        self.X = np.load(filepath / "nodeX.npy")[:, :2]
        self.U = np.load(filepath / "nodeU.npy")

        self.N_t = self.U.shape[0]
        self.N_n = self.X.shape[0]
        self.N_e = len(self.mesh)

        # BCset: all nodes at boundaries
        self.BCset = None
        with open(filepath / "BCset.json") as f:
            self.BCset = json.load(f)

        self.FREEset = np.setdiff1d(np.arange(self.N_n), self.BCset)

        self.IFX1set = None
        self.IFX2set = None
        with open(filepath / "IFsets.json") as f:
            IFsets = json.load(f)
            self.IFX1set = IFsets[0]
            self.IFX2set = IFsets[1]

        self.IFX = np.load(filepath / "IFX.npy")

        S_val = np.load(filepath / "elementS.npy")
        S_cauchy = np.zeros((self.N_t, self.N_e, 1, 2, 2))
        S_cauchy[:, :, :, 0, 0] = S_val[:, :, :, 0]
        S_cauchy[:, :, :, 1, 1] = S_val[:, :, :, 1]
        # S_cauchy[:,:,:,2,2] = S_val[:,:,:,2]
        S_cauchy[:, :, :, 0, 1] = S_val[:, :, :, 3]
        S_cauchy[:, :, :, 1, 0] = S_val[:, :, :, 3]

        self.Xip = np.zeros((self.N_e, 1, 2))  # positions of integration points
        self.B = np.zeros((self.N_e, 1, 3, 2))  # {u}[B] = F-I
        self.Bcoeff = np.zeros((self.N_e, 1))  # weights of integration points

        self.F = np.zeros((self.N_t, self.N_e, 1, 2, 2))

        for eid, nlist in enumerate(self.mesh):
            x_e = self.X[nlist, :]

            x_ip, Bcoeff, B = make_C3(x_e)

            self.Xip[eid, :, :] = x_ip

            self.Bcoeff[eid, :] = Bcoeff

            self.B[eid, :, :, :] = B

            self.F[:, eid, :, :, :] = np.einsum("tni,knj->tkij", self.U[:, nlist, :], B) + np.eye(2).reshape(1, 1, 2, 2)

        self.J = np.linalg.det(self.F)
        self.Finv = np.linalg.inv(self.F)

        self.stress_P = np.einsum("tep,tepik,tepjk->tepij", self.J, S_cauchy, self.Finv)
        self.stress_S = np.einsum("tepik,tepkj->tepij", self.Finv, self.stress_P)

        lam3 = np.ones((self.N_t, self.N_e))
        try:
            lam3 = np.load(filepath / "lam3.npy")
            print("Loaded lam3 shape:", lam3.shape, flush=True)
            print("lam3 stats: min=", lam3.min(), "max=", lam3.max(), "mean=", lam3.mean())
            print("lam3 sample:", lam3[5, :10])
        except FileNotFoundError:
            print("lam3 not found", flush=True)

        self.lam3 = lam3

        lam3_expanded = lam3[:, :, None, None, None]

        self.stress_P = self.stress_P * lam3_expanded
        self.stress_S = self.stress_S * lam3_expanded


class FieldData_tri_multiBC:
    #! DO NOT MODIFY THIS !!!
    # N_t: number of stretches
    # N_e: number of elements (triangles)
    # N_n: number of nodes
    # N_BC: number of boundary nodes
    # N_free: number of free nodes
    # N_ifx1: number of ifx1 nodes
    # N_ifx2: number of ifx2 nodes

    # X: (N_n, 2)
    # U: (N_t, N_n, 2)
    # mesh: (N_e, 3)
    # BCset: (N_BC,)

    # IFdata: list[dict]
    #   {
    #       nodeset: (N_nodes,)
    #       direction: (2,)
    #       force: (N_t,)
    #   } x N_BC

    # Xip: (N_e, 1, 2)
    # B: (N_e, 1, 3, 2)
    # Bcoeff: (N_e, 1)
    # F: (N_t, N_e, 1, 2, 2)
    # P: (N_t, N_e, 1, 2, 2)
    # S: (N_t, N_e, 1, 2, 2)
    # lam3: (N_t, N_e)
    def __init__(self, filepath: Path):
        self.mesh = None
        with open(filepath / "mesh.json") as f:
            self.mesh = np.array(json.load(f))

        self.X = np.load(filepath / "nodeX.npy")[:, :2]
        self.U = np.load(filepath / "nodeU.npy")

        self.N_t = self.U.shape[0]
        self.N_n = self.X.shape[0]
        self.N_e = len(self.mesh)
        self.N_BC = 0

        # BCset: all nodes at boundaries
        self.BCset = None
        with open(filepath / "BCset.json") as f:
            self.BCset = json.load(f)

        self.FREEset = np.setdiff1d(np.arange(self.N_n), self.BCset)

        # * new added (IFX1set, IFX2set) => (IFdata, N_BC)
        self.IFdata = []
        with open(filepath / "IFdata.json") as f:
            IFdata = json.load(f)

        self.N_BC = len(IFdata)
        for data in IFdata:
            self.IFdata.append(
                {
                    "nodeset": np.array(data["nodeset"], dtype=int),
                    "direction": np.array(data["direction"], dtype=float),
                    "force": np.array(data["force"], dtype=float),
                }
            )

        S_val = np.load(filepath / "elementS.npy")

        S_cauchy = np.zeros((self.N_t, self.N_e, 1, 2, 2))
        S_cauchy[:, :, :, 0, 0] = S_val[:, :, :, 0]
        S_cauchy[:, :, :, 1, 1] = S_val[:, :, :, 1]
        # S_cauchy[:, :, :, 2, 2] = S_val[:, :, :, 2]
        S_cauchy[:, :, :, 0, 1] = S_val[:, :, :, 3]
        S_cauchy[:, :, :, 1, 0] = S_val[:, :, :, 3]

        self.Xip = np.zeros((self.N_e, 1, 2))  # positions of integration points
        self.B = np.zeros((self.N_e, 1, 3, 2))  # {u}[B] = F-I
        self.Bcoeff = np.zeros((self.N_e, 1))  # weights of integration points

        self.F = np.zeros((self.N_t, self.N_e, 1, 2, 2))

        for eid, nlist in enumerate(self.mesh):
            x_e = self.X[nlist, :]

            x_ip, Bcoeff, B = make_C3(x_e)

            self.Xip[eid, :, :] = x_ip

            self.Bcoeff[eid, :] = Bcoeff

            self.B[eid, :, :, :] = B

            self.F[:, eid, :, :, :] = np.einsum("tni,knj->tkij", self.U[:, nlist, :], B) + np.eye(2).reshape(1, 1, 2, 2)

        self.J = np.linalg.det(self.F)
        self.Finv = np.linalg.inv(self.F)

        self.stress_P = np.einsum("tep,tepik,tepjk->tepij", self.J, S_cauchy, self.Finv)
        self.stress_S = np.einsum("tepik,tepkj->tepij", self.Finv, self.stress_P)

        lam3 = np.ones((self.N_t, self.N_e))
        try:
            lam3 = np.load(filepath / "lam3.npy")
            print("Loaded lam3 shape:", lam3.shape, flush=True)
            print("lam3 stats: min=", lam3.min(), "max=", lam3.max(), "mean=", lam3.mean())
            print("lam3 sample:", lam3[5, :10])
        except FileNotFoundError:
            print("lam3 not found", flush=True)

        self.lam3 = lam3

        lam3_expanded = lam3[:, :, None, None, None]

        self.stress_P = self.stress_P * lam3_expanded
        self.stress_S = self.stress_S * lam3_expanded
