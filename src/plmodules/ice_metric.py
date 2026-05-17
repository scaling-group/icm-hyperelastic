import einops
import torch

from src.datasets import ice_data_multi_utils as idu


def get_error_nodal_force(calibrated_nodal_force: torch.Tensor, mesh: idu.DataMesh) -> tuple[torch.Tensor, ...]:
    """
    get the error (with signs) of the nodal force on all points
    for free points, the reference is 0
    for IFX1 and IFX2, we don't know the ground truth, therefore we use (total force / IFX size) as the reference
    """

    nodal_force_error = calibrated_nodal_force.clone()  # (bs, n_nodes, 2)

    # set boundary force error to 0
    batch_idx = torch.arange(calibrated_nodal_force.size(0)).reshape(-1, 1)  # (bs, 1)
    nodal_force_error[batch_idx, mesh["BCset"], :] = 0

    scale = 1  # TODO: how to get the scale?
    nodal_force_rel_error = nodal_force_error / scale  # (bs, n_nodes, 2)

    return nodal_force_error, nodal_force_rel_error  # (bs, n_n, 2), (bs, n_n, 2)


def get_error_S(calibrated_stress_S: torch.Tensor, mesh: idu.DataMesh) -> tuple[torch.Tensor, ...]:
    """
    get the error (with signs) of the stress on the elements
    stress_S: (bs, N_e, 1, 2, 2)
    """
    ref_stress_S = mesh["S"]

    ref_area = mesh["Bcoeff"]  # (bs, N_e, 1)
    ref_area_sum = einops.reduce(ref_area, "bs N_e 1 -> bs 1 1 1 1", "sum")  # (bs, 1, 1, 1, 1)
    ref_area = einops.rearrange(ref_area, "bs N_e 1 -> bs N_e 1 1 1")  # (bs, N_e, 1, 1, 1)
    weight_area = ref_area / ref_area_sum  # (bs, N_e, 1, 1, 1)

    # Frobenius norm over last two dims (dims are kept):
    # (bs, N_e, 1, 2, 2) -> (bs, N_e, 1, 1, 1)
    ref_stress_S_norm = ref_stress_S.norm(dim=(-1, -2), keepdim=True)
    scale = einops.reduce(ref_stress_S_norm**2, "bs N_e 1 1 1 -> bs 1 1 1 1", "mean") ** 0.5

    # so we use the range of the stress
    # (bs, N_e, 1, 2, 2) -> (bs, N_e, 2, 2) since dim has to be int in torch.max and torch.min
    ref_stress_S_max = einops.reduce(ref_stress_S, "bs N_e n i j -> bs i j", "max")[:, None, None, :, :]
    ref_stress_S_min = einops.reduce(ref_stress_S, "bs N_e n i j -> bs i j", "min")[:, None, None, :, :]
    scale_minmax = ref_stress_S_max - ref_stress_S_min  # (bs, N_e, 1, 2, 2)

    # scale by the area
    # element-wise multiplication:
    # ((bs, N_e, 1, 1, 1)** 2) * (bs, N_e, 1, 1, 1) -> (bs, N_e, 1, 1, 1)
    weighted_S_norm_square = ref_stress_S_norm**2 * weight_area
    scale_stress_S_with_area = (
        einops.reduce(weighted_S_norm_square, "bs N_e n 1 1 -> bs 1 1 1 1", "sum") ** 0.5
    )  # (bs, 1, 1, 1, 1)

    stress_S_error = calibrated_stress_S - ref_stress_S  # (bs, N_e, 1, 2, 2)
    stress_S_rel_error = stress_S_error / scale  # (bs, N_e, 1, 2, 2)
    stress_S_rel_error_minmax = stress_S_error / scale_minmax  # (bs, N_e, 1, 2, 2)
    stress_S_rel_error_area = stress_S_error / scale_stress_S_with_area  # (bs, N_e, 1, 2, 2)
    stress_S_area_rel_error_area = stress_S_error * weight_area / scale_stress_S_with_area  # (bs, N_e, 1, 2, 2)

    return (
        stress_S_error,
        stress_S_rel_error,
        stress_S_rel_error_minmax,
        stress_S_rel_error_area,
        stress_S_area_rel_error_area,
        weight_area,
    )


def get_errors(preds: dict[str, list[torch.Tensor]], mesh_list: idu.DataMeshList) -> dict[str, list[torch.Tensor]]:
    """
    get the detailed errors for each mesh
    """
    nodal_force_errors = []
    nodal_force_rel_errors = []
    stress_S_errors = []
    stress_S_rel_errors = []
    stress_S_rel_errors_minmax = []
    stress_S_rel_errors_area = []
    stress_S_area_rel_errors_area = []
    weight_areas = []
    for i, mesh in enumerate(mesh_list):
        nodal_force_error, nodal_force_rel_error = get_error_nodal_force(
            preds["calibrated_nodal_forces"][i], mesh
        )  # (bs, N_n, 2), (bs, N_n, 2)
        (
            stress_S_error,
            stress_S_rel_error,
            stress_S_rel_error_minmax,
            stress_S_rel_error_area,
            stress_S_area_rel_error_area,
            weight_area,
        ) = get_error_S(preds["calibrated_stress_Ss"][i], mesh)  # (bs, N_e, 1, 2, 2), (bs, N_e, 1, 2, 2)

        nodal_force_errors.append(nodal_force_error)
        nodal_force_rel_errors.append(nodal_force_rel_error)
        stress_S_errors.append(stress_S_error)
        stress_S_rel_errors.append(stress_S_rel_error)
        stress_S_rel_errors_minmax.append(stress_S_rel_error_minmax)
        stress_S_rel_errors_area.append(stress_S_rel_error_area)
        stress_S_area_rel_errors_area.append(stress_S_area_rel_error_area)
        weight_areas.append(weight_area)
    return {
        "nodal_force_errors": nodal_force_errors,
        "nodal_force_rel_errors": nodal_force_rel_errors,
        "stress_S_errors": stress_S_errors,
        "stress_S_rel_errors": stress_S_rel_errors,
        "stress_S_rel_errors_minmax": stress_S_rel_errors_minmax,
        "stress_S_rel_errors_area": stress_S_rel_errors_area,
        "stress_S_area_rel_errors_area": stress_S_area_rel_errors_area,
        "weight_areas": weight_areas,
    }


def get_metrics(
    preds: dict[str, list[torch.Tensor]], errors: dict[str, list[torch.Tensor]], mesh_list: idu.DataMeshList
) -> dict[str, torch.Tensor]:
    """
    get the error collection, averaged over elements/nodes, so the shape is (bs, meshes)-like
    """

    # abs_nf_bdry = []
    abs_nf_free = []
    abs_nf_all = []

    # rel_nf_bdry = []
    rel_nf_free = []
    rel_nf_all = []

    abs_S = []
    rel_S = []
    rel_S_minmax = []
    rel_S_area = []
    for i, mesh in enumerate(mesh_list):
        nodal_force_error = errors["nodal_force_errors"][i]
        nodal_force_rel_error = errors["nodal_force_rel_errors"][i]
        stress_S_error = errors["stress_S_errors"][i]
        stress_S_rel_error = errors["stress_S_rel_errors"][i]
        stress_S_rel_error_minmax = errors["stress_S_rel_errors_minmax"][i]
        stress_S_rel_error_area = errors["stress_S_rel_errors_area"][i]
        weight_area = errors["weight_areas"][i]

        abs_errors_free = idu.slice_nodes_set(nodal_force_error, mesh["FREEset"])  # (bs, N_free, 2)
        abs_nf_free.append(abs_errors_free.norm(dim=-1).mean(dim=-1))  # (bs, )
        abs_nf_all.append(nodal_force_error.norm(dim=-1).mean(dim=-1))  # (bs, )

        rel_errors_free = idu.slice_nodes_set(nodal_force_rel_error, mesh["FREEset"])  # (bs, N_free, 2)
        rel_nf_free.append(rel_errors_free.norm(dim=-1).mean(dim=-1))  # (bs, )
        rel_nf_all.append(nodal_force_rel_error.norm(dim=-1).mean(dim=-1))  # (bs, )

        abs_S.append(stress_S_error.norm(dim=(-1, -2)).mean(dim=(-1, -2)))  # (bs, )
        rel_S.append((stress_S_rel_error.norm(dim=(-1, -2)) ** 2).mean(dim=(-1, -2)) ** 0.5)  # (bs, )
        rel_S_minmax.append(stress_S_rel_error_minmax.norm(dim=(-1, -2)).mean(dim=(-1, -2)))  # (bs, )
        stress_S_rel_error_area_norm = stress_S_rel_error_area.norm(dim=(-1, -2), keepdim=True)  # (bs, N_e, 1)

        # element-wise multiplication:
        # (bs, N_e, 1, 1, 1) * ((bs, N_e, 1, 1, 1)**2) -> (bs, N_e, 1, 1, 1)
        weighted_S_rel_error_area_norm_square = weight_area * (stress_S_rel_error_area_norm**2)
        rel_S_area.append(
            einops.reduce(weighted_S_rel_error_area_norm_square, "bs N_e n 1 1 -> bs", "sum") ** 0.5
        )  # (bs, )

    abs_nf_free = torch.stack(abs_nf_free, dim=1)  # (bs, meshes)
    abs_nf_all = torch.stack(abs_nf_all, dim=1)  # (bs, meshes)

    rel_nf_free = torch.stack(rel_nf_free, dim=1)  # (bs, meshes)
    rel_nf_all = torch.stack(rel_nf_all, dim=1)  # (bs, meshes)

    abs_S = torch.stack(abs_S, dim=1)  # (bs, meshes)
    rel_S = torch.stack(rel_S, dim=1)  # (bs, meshes)
    rel_S_minmax = torch.stack(rel_S_minmax, dim=1)  # (bs, meshes)
    rel_S_area = torch.stack(rel_S_area, dim=1)  # (bs, meshes)

    scales = torch.stack(preds["scales"], dim=1)  # (bs, meshes)
    scales_ratio = torch.cat(preds["scales_ratio"], dim=1)  # (bs, n_bc * meshes)
    scales_rel_std = torch.stack(preds["scales_rel_std"], dim=1)  # (bs, meshes)

    return {
        "scales": scales,
        "scales_ratio": scales_ratio,
        "scales_rel_std": scales_rel_std,
        "abs_nf_free": abs_nf_free,
        "abs_nf_all": abs_nf_all,
        "rel_nf_free": rel_nf_free,
        "rel_nf_all": rel_nf_all,
        "abs_S": abs_S,
        "rel_S": rel_S,
        "rel_S_minmax": rel_S_minmax,
        "rel_S_area": rel_S_area,
    }
