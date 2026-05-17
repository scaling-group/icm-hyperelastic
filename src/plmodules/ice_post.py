import einops
import torch

## post-processing


def get_single_calibrate_scale(grad_psi_grad_I: torch.Tensor, boundary: dict, mode: str) -> dict[str, torch.Tensor]:
    """
    Get the raw scale for ONE boundary condition

    Args:
        grad_psi_grad_I (torch.Tensor): (bs, N_bc, 8, 2)
        boundary (dict):
                'A': (bs, N_bc, 8, 2, 2)
                'XI': (bs, N_bc, 8, 2)
                'direction': (bs, 2)
                'force': (bs,)
                here we didn't use mask, since we assume padded elements has zero A.
        mode (str): _description_

    Returns:
        out: (dict[str, torch.Tensor]):
        {
            "pred_force": ...,
            "label_force": ...,
        }

    """

    if mode == "raw_unprojected":
        # sum over all boundary nodes and surrounding elements of each node
        pred_force = einops.einsum(
            boundary["A"],  # (bs, N_bc, 8, 2, 2)
            grad_psi_grad_I,  # (bs, N_bc, 8, 2)
            "b n e d1 d2, b n e d2 -> b d1",  # (bs, 2)
        )
        label_force = boundary["force"] * boundary["direction"]  # (bs, 2)
        return {"pred_force": pred_force, "label_force": label_force}
    elif mode == "raw_projected":
        # sum over all boundary nodes and surrounding elements of each node
        # then inner product with the direction vector
        pred_force = einops.einsum(
            boundary["A"],  # (bs, N_bc, 8, 2, 2)
            grad_psi_grad_I,  # (bs, N_bc, 8, 2)
            boundary["direction"],  # (bs, 2)
            "b n e d1 d2, b n e d2, b d1 -> b",  # (bs,)
        )
        label_force = boundary["force"]  # (bs,)
        return {"pred_force": pred_force, "label_force": label_force}
    else:
        raise ValueError(f"Got unknown mode: {mode}")


def pool_calibrate_scales(raw_scales: list[list[dict]], mode: str) -> tuple[list[torch.Tensor], ...]:
    """
    pool the raw calibrate scales, should be from "get_single_calibrate_scale" with "raw_projected" mode
    raw_scales: [dict * n_bc] * meshes, dict: {pred_force, label_force}
    """
    num_meshes = len(raw_scales)
    # the calculated scales for each mesh, each boundary
    if "pred_vs_label" in mode:
        scales = [[bc["pred_force"] / bc["label_force"] for bc in mesh] for mesh in raw_scales]
    else:
        raise ValueError(f"Got unknown pred label calibration mode: {mode}")

    scales_ratio = [torch.stack(mesh, dim=1) for mesh in scales]  # [(bs, n_bc)] * meshes

    if "pcal" in mode:
        scales_ratio_flat = torch.cat(scales_ratio, dim=1)  # (bs, n_bc * meshes)
        scales_mean_flat = torch.mean(scales_ratio_flat, dim=1)  # (bs,)
        if "pred_vs_label" in mode:
            pooled_scales_tensor = 1 / scales_mean_flat
            pooled_scales_std = torch.std(scales_ratio_flat, dim=1)
        else:
            raise ValueError(f"Got unknown pred label calibration mode: {mode}")
        pooled_scales = [pooled_scales_tensor] * num_meshes
        scales_std = [pooled_scales_std] * num_meshes
    else:
        raise ValueError(f"Got unknown calibration mode: {mode}")

    scales_rel_std = [std / ((1 / mean) + 1e-6) for mean, std in zip(pooled_scales, scales_std, strict=True)]

    return pooled_scales, scales_ratio, scales_rel_std  # [(bs,)] * meshes, [(bs, n_bc)] * meshes, [(bs,)] * meshes


def calibrate(
    pooled_scales: list[torch.Tensor], uncalibrated_values: list[torch.Tensor], scale_idxs: list[int]
) -> list[torch.Tensor]:
    """
    calibrate the raw values using the scales, mesh-by-mesh
    pooled_scales: [(bs,)] * prpt_meshes
    uncalibrated_values: [(bs, ...)] * qury_meshes
    scale_idxs: [int] * qury_meshes
    we use strict=True, since the length of `scale_idxs` and `uncalibrated_scales` should be the same
    """
    calibrated_values = [
        einops.einsum(pooled_scales[scale_idx], value, "b, b ... -> b ...")
        for scale_idx, value in zip(scale_idxs, uncalibrated_values, strict=True)
    ]
    return calibrated_values
