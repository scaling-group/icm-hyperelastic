import einops
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import src.utils.ice_project_utils as pu
from src.datasets.pytree_utils import to_numpy

from . import viz_scatter as vs
from . import viz_utils as vu
from .viz import Viz


class VizStressDistribute(Viz):
    def __init__(
        self,
        dirpath: str,
        valid_batches_local: str,  # save batches in local machine
        valid_batches_log: str,  # log batches to remote wandb
        test_batches_local: str,  # save batches in local machine
        test_batches_log: str,  # log batches to remote wandb
    ):
        super().__init__(
            dirpath,
            valid_batches_local,
            valid_batches_log,
            test_batches_local,
            test_batches_log,
        )
        self.category = "stress_distribute"

    def get_image(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> Image.Image:
        data = batch["qury"]
        cmode = pl_module.cfg.loss.calibrate_modes[0]
        description = batch["description"]
        material_id, meshlist_ids = pu.get_mesh_id(description[0])
        if len(meshlist_ids) > 3:
            title_description = (
                f"step {trainer.global_step}, "
                f"Mesh {material_id}_{meshlist_ids[0][0]}_{meshlist_ids[0][2]}~{meshlist_ids[-1][2]}\n"
            )
        else:
            title_description = f"step {trainer.global_step}, {description}"

        data_np = to_numpy(data)
        outputs_np = to_numpy(outputs)

        # concat area of each batch
        area = np.concatenate([mesh["Bcoeff"] for mesh in data_np], axis=1)  # (bs, N_e_total, 1)
        # take the first batch for visualization
        area = area[0]

        # concat stress_Ss_pred of each batch
        stress_Ss_pred = np.concatenate(
            [outputs_np["preds"][f"{cmode}/calibrated_stress_Ss"][i] for i in range(len(data_np))], axis=1
        )
        # take the first batch for visualization
        stress_Ss_pred = stress_Ss_pred[0]
        stress_Ss_pred_flattened = einops.rearrange(stress_Ss_pred, "n c h w -> n c (h w)", h=2, w=2)

        # concat stress_Ss_ref of each batch
        stress_Ss_ref = np.concatenate([mesh["S"] for mesh in data_np], axis=1)
        # take the first batch for visualization
        stress_Ss_ref = stress_Ss_ref[0]
        stress_Ss_ref_flattened = einops.rearrange(stress_Ss_ref, "n c h w -> n c (h w)", h=2, w=2)

        stress_Ss_combined = [
            np.stack((stress_Ss_ref_flattened[..., i], stress_Ss_pred_flattened[..., i]), axis=-1) for i in range(4)
        ]

        fig_stress_Ss = []
        for i in range(4):
            fig = vs.viz_scatter_value_colored(
                stress_Ss_combined[i],
                area[..., None],
                title=f"{title_description}",
                subplot_title=["$S_{11}$", "$S_{12}$", "$S_{21}$", "$S_{22}$"][i],
                subplot_xlabel="ref_S",
                subplot_ylabel="pred_S",
                colorbar_range="normal",
            )
            fig_stress_Ss.append(fig)

        img = vu.merge_images([fig_stress_Ss])  # merge plots of all meshes in one line, PIL image
        plt.close("all")
        return img  # PIL image
