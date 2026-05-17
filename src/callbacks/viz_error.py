import matplotlib.pyplot as plt
import torch
from PIL import Image

import src.utils.ice_project_utils as pu
from src.datasets.pytree_utils import to_numpy

from . import viz_mesh as vm
from . import viz_node as vn
from . import viz_utils as vu
from .viz import Viz


def plot_element(mesh_data: dict, tensor_dict: dict[str, torch.Tensor], title_prefix: str, dim: int):
    bid = 0  # only plot the first in batch. Actually bs = 1

    new_X = mesh_data["U"][bid] + mesh_data["X"][bid]

    X_fig = vm.visualize_mesh(mesh_data["mesh"][bid], mesh_data["X"][bid], title=f"{title_prefix}X")
    imgs = [[X_fig]]

    for key, tensor in tensor_dict.items():
        img_list = []
        if dim == 2:
            for i in range(tensor.shape[-1]):
                for j in range(tensor.shape[-2]):
                    img_list.append(
                        vm.visualize_mesh(
                            mesh_data["mesh"][bid],
                            new_X,
                            tensor[bid, :, 0, i, j],
                            title=f"{title_prefix} {key}_{i + 1}{j + 1}",
                        )
                    )
        elif dim == 1:
            for i in range(tensor.shape[-1]):
                img_list.append(
                    vm.visualize_mesh(
                        mesh_data["mesh"][bid],
                        new_X,
                        tensor[bid, :, 0, i],
                        title=f"{title_prefix} {key}_{i + 1}",
                    )
                )
        imgs.append(img_list)

    return imgs  # 2D list


def plot_node(mesh_data: dict, tensor_dict: dict[str, torch.Tensor], title_prefix: str):
    bid = 0  # only plot the first in batch. Actually bs = 1

    new_X = mesh_data["U"][bid] + mesh_data["X"][bid]

    imgs = []
    for key, tensor in tensor_dict.items():
        force_figs = [
            vn.visualize_node(
                mesh_data["mesh"][bid],
                new_X,
                tensor[bid, :, i],
                title=f"{title_prefix}{key}_{'x' if i == 0 else 'y'}",
            )
            for i in [0, 1]
        ] + [
            vn.visualize_node_vector(
                mesh_data["mesh"][bid],
                new_X,
                tensor[bid, :, :],
                title=f"{title_prefix}{key}",
            )
        ]
        imgs.append(force_figs)
    return imgs  # 2D list


class VizError(Viz):
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
        self.category = "error"

    def get_image(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> Image.Image:
        if batch["dataset"] == "MeshListDataset":
            bid = 0  # only plot the first in batch. Actually bs = 1
            data = batch["data"]
            cmode = pl_module.cfg.loss.calibrate_modes[0]
            plots = []
            material_id, meshlist_ids = pu.get_mesh_id(batch["description"][bid])

            outputs_np = to_numpy(outputs)
            data_np = to_numpy(data)

            for t in range(len(data_np)):
                mesh_data = {}
                for k, v in data_np[t].items():
                    mesh_data[k] = v

                # meshlist_ids = [ [geometry_id, force_type, stretch_id], ... ]
                title_prefix = (
                    f"step {trainer.global_step}, Mesh {material_id}_{meshlist_ids[t][0]}_{meshlist_ids[t][2]}\n"
                )

                stress_dict = {
                    "S_pred": outputs_np["preds"][f"{cmode}/calibrated_stress_Ss"][t],
                    "S_ref": mesh_data["S"],
                    "S_err": outputs_np["errors"][f"{cmode}/stress_S_errors"][t],
                    "S_rerr_minmax": outputs_np["errors"][f"{cmode}/stress_S_rel_errors_minmax"][t],
                    "S_rerr_area": outputs_np["errors"][f"{cmode}/stress_S_rel_errors_area"][t],
                    "S_rerr_times_area": outputs_np["errors"][f"{cmode}/stress_S_area_rel_errors_area"][t],
                }

                stress_plot = plot_element(mesh_data, stress_dict, title_prefix, dim=2)

                nodal_force_dict = {
                    "nodal_force": outputs_np["preds"][f"{cmode}/calibrated_nodal_forces"][t],
                    "nodal_force_err": outputs_np["errors"][f"{cmode}/nodal_force_errors"][t],
                    "nodal_force_rel_err": outputs_np["errors"][f"{cmode}/nodal_force_rel_errors"][t],
                }

                nodal_force_plot = plot_node(mesh_data, nodal_force_dict, title_prefix)

                plot = vu.merge_images(stress_plot + nodal_force_plot)
                plots.append(plot)

            img = vu.merge_images([plots])
            plt.close("all")
            return img
        elif batch["dataset"] == "ParallelMeshListDataset":
            return None
        else:
            raise ValueError(f"Unknown dataset: {batch['dataset']}")
