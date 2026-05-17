import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from PIL import Image

import src.utils.ice_project_utils as pu
from src.datasets.pytree_utils import to_numpy

from . import viz_error as ve
from . import viz_utils as vu
from .viz import Viz


def viz_scatter_value_colored(
    viz_coordinate: np.ndarray,
    viz_value: np.ndarray,
    title: str | None = None,
    subplot_title: list[str] | str | None = None,
    subplot_xlabel: str | None = None,
    subplot_ylabel: str | None = None,
    colorbar_range: str = "symmetry",
) -> Figure:
    """
    Visualize a scatter plot of the value colored by the value.
    ---------------------------------------------------------
    Parameters
    ----------
    viz_coordinate: (n_nodes, 1, 2)
    viz_value: (n_nodes, 1, n_features)
    Returns
    -------
    fig: matplotlib.figure.Figure
        The created figure.
    """
    fig_num = viz_value.shape[-1]
    fig, axs = plt.subplots(1, fig_num, figsize=(8 * fig_num, 8))
    fig.suptitle(f"{title}")

    if fig_num == 1:
        axs = np.array([axs])
        if not isinstance(subplot_title, str) and subplot_title is not None:
            raise ValueError(f"subplot_title should be a string or None, got {type(subplot_title)}")
        axs[0].set_title(subplot_title)
        axs[0].set_xlabel(subplot_xlabel)
        axs[0].set_ylabel(subplot_ylabel)
        axs[0].grid(True)
    else:
        for i in range(fig_num):
            if not isinstance(subplot_title, list) and subplot_title is not None:
                raise ValueError(f"subplot_title should be a list or None, got {type(subplot_title)}")
            axs[i].set_title(subplot_title[i])
            axs[i].set_xlabel(subplot_xlabel)
            axs[i].set_ylabel(subplot_ylabel)
            axs[i].grid(True)

    scatter_plots = []

    x = viz_coordinate[:, 0, 0]
    y = viz_coordinate[:, 0, 1]
    if colorbar_range == "symmetry":
        for i in range(fig_num):
            color = viz_value[:, 0, i]
            vmax = np.maximum(abs(np.min(viz_value[:, 0, i])), abs(np.max(viz_value[:, 0, i])))
            vmin = -vmax
            threshold = 0.002 * (vmax - vmin)
            mask = np.abs(color) < threshold

            scatter_plot = axs[i].scatter(x, y, c=color, cmap="coolwarm", s=10, marker="o", vmin=vmin, vmax=vmax)
            axs[i].scatter(x[mask], y[mask], c="green", s=10, marker="o", alpha=1.0)
            scatter_plots.append(scatter_plot)
    elif colorbar_range == "normal":
        for i in range(fig_num):
            color = viz_value[:, 0, i]
            vmax = np.max(viz_value[:, 0, i])
            vmin = np.min(viz_value[:, 0, i])
            scatter_plot = axs[i].scatter(x, y, c=color, cmap="viridis", s=10, marker="o", vmin=vmin, vmax=vmax)
            scatter_plots.append(scatter_plot)
            x_line = np.array([min(x.min(), y.min()), max(x.max(), y.max())])
            axs[i].plot(x_line, x_line, "r--", label="y=x")

    for i in range(fig_num):
        fig.colorbar(scatter_plots[i], ax=axs[i])
        axs[i].legend()

    return fig


def get_quadratic_roots(coeffs: np.ndarray) -> np.ndarray:
    """
    Calculate the roots of the quadratic equation x^2-c1*x+c2=0.
    Parameters
    ----------
    coeffs: np.ndarray of shape (a, b, 2)
        The coefficients (c1, c2) for each point.
    Returns
    -------
    roots: np.ndarray of shape (a, b, 2)
        The two roots for each point.
    """
    a, b, _ = coeffs.shape
    roots = np.zeros((a, b, 2))

    for i in range(a):
        for j in range(b):
            c1, c2 = coeffs[i, j]
            # Use numpy's roots function to solve the equation
            # Note: The roots function returns the roots in no particular order
            r = np.roots([1, -c1, c2])
            # Sort the roots
            r.sort()
            roots[i, j] = r

    return roots


class Viz_scatter(Viz):
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
        self.category = "scatter"

    def get_mesh_grad_psi_grad_I_image(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> Image.Image:
        """
        Visualize I and grad_psi_grad_I on mesh.
        """
        data = batch["data"]
        plots = []
        bid = 0  # only plot the first in batch
        material_id, meshlist_ids = pu.get_mesh_id(batch["description"][bid])

        data_np = to_numpy(data)
        outputs_np = to_numpy(outputs)

        for i in range(len(data_np)):
            mesh_data = {}
            for k, v in data_np[i].items():
                mesh_data[k] = v

            geometry = meshlist_ids[i][0]
            snapshot = meshlist_ids[i][2]
            title_prefix = f"step {trainer.global_step}, Mesh {material_id}_{geometry}_{snapshot}"

            grad_psi_grad_I_plot = ve.plot_element(
                mesh_data,
                {
                    "I": mesh_data["I"],
                    "grad_psi_grad_I": outputs_np["preds"]["raw/grad_psi_grad_Is"][i],
                },
                title_prefix,
                dim=1,
            )

            plot = vu.merge_images(grad_psi_grad_I_plot)
            plots.append(plot)

        img = vu.merge_images([plots])
        plt.close("all")
        return img

    def get_scatter_grad_psi_grad_I_image(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> Image.Image:
        """
        Visualize grad_psi_grad_I on scatter plot.
        Figure 1: I1 and I2 are the coordinates, grad_psi_grad_I is the value.
        Figure 2: lambda1 and lambda2 are the coordinates, grad_psi_grad_I is the value.
        """
        data = batch["qury"]
        plots_scatter_grad_psi_grad_I_to_I = []
        plots_scatter_grad_psi_grad_I_to_lambda = []
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

        # concatenate the results of all meshes in the batch, the first dimension is the number of points in all meshes
        mesh_I_all = np.concatenate([mesh["I"] for mesh in data_np], axis=1).squeeze(axis=0)
        mesh_lambda_all = get_quadratic_roots(mesh_I_all)
        grad_psi_grad_Is_all = np.concatenate(
            [outputs_np["preds"]["raw/grad_psi_grad_Is"][i] for i in range(len(data))], axis=1
        ).squeeze(axis=0)

        fig_grad_psi_grad_I_to_I = viz_scatter_value_colored(
            mesh_I_all,
            grad_psi_grad_Is_all,
            title=f"{title_description}",
            subplot_title=[r"$\partial \psi / \partial I_{1}$", r"$\partial \psi / \partial I_{2}$"],
            subplot_xlabel=r"$I_{1}$",
            subplot_ylabel=r"$I_{2}$",
        )
        fig_grad_psi_grad_I_to_lambda = viz_scatter_value_colored(
            mesh_lambda_all,
            grad_psi_grad_Is_all,
            title=f"{title_description}",
            subplot_title=[r"$\partial \psi / \partial I_{1}$", r"$\partial \psi / \partial I_{2}$"],
            subplot_xlabel=r"$\lambda_{1}$",
            subplot_ylabel=r"$\lambda_{2}$",
        )
        plots_scatter_grad_psi_grad_I_to_I.append(fig_grad_psi_grad_I_to_I)
        plots_scatter_grad_psi_grad_I_to_lambda.append(fig_grad_psi_grad_I_to_lambda)
        img_scatter_grad_psi_grad_I = vu.merge_images(
            [plots_scatter_grad_psi_grad_I_to_I, plots_scatter_grad_psi_grad_I_to_lambda]
        )
        plt.close("all")
        return img_scatter_grad_psi_grad_I

    def get_scatter_stress_image(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> Image.Image:
        """
        Visualize stress_S on scatter plot.
        Figure 1: I1 and I2 are the coordinates, stress_S is the value.
        Figure 2: lambda1 and lambda2 are the coordinates, stress_S is the value.
        """
        data = batch["qury"]
        plots_scatter_stress_to_I = []
        plots_scatter_stress_to_lambda = []
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

        # concatenate the results of all meshes in the batch, the first dimension is the number of points in all meshes
        mesh_I_all = np.concatenate([mesh["I"] for mesh in data_np], axis=1).squeeze(axis=0)
        mesh_lambda_all = get_quadratic_roots(mesh_I_all)
        stress_Ss_numpy = np.concatenate(
            [outputs_np["preds"]["raw/stress_Ss"][i] for i in range(len(data))], axis=1
        ).squeeze(axis=0)
        stress_Ss_flattened = stress_Ss_numpy.reshape(*stress_Ss_numpy.shape[:-2], 4)

        fig_stress_S_to_I = viz_scatter_value_colored(
            mesh_I_all,
            stress_Ss_flattened,
            title=f"{title_description}",
            subplot_title=[r"$S_{11}$", r"$S_{12}$", r"$S_{21}$", r"$S_{22}$"],
            subplot_xlabel=r"$I_{1}$",
            subplot_ylabel=r"$I_{2}$",
        )
        fig_stress_S_to_lambda = viz_scatter_value_colored(
            mesh_lambda_all,
            stress_Ss_flattened,
            title=f"{title_description}",
            subplot_title=[r"$S_{11}$", r"$S_{12}$", r"$S_{21}$", r"$S_{22}$"],
            subplot_xlabel=r"$\lambda_{1}$",
            subplot_ylabel=r"$\lambda_{2}$",
        )
        plots_scatter_stress_to_I.append(fig_stress_S_to_I)
        plots_scatter_stress_to_lambda.append(fig_stress_S_to_lambda)

        img_scatter_stress = vu.merge_images([plots_scatter_stress_to_I, plots_scatter_stress_to_lambda])
        plt.close("all")
        return img_scatter_stress

    def get_image(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> Image.Image:
        """
        override this function, merge three images together
        """
        # img_grad_psi_grad_I = self.get_mesh_grad_psi_grad_I_image(
        #     trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        # )
        img_scatter_grad_psi_grad_I = self.get_scatter_grad_psi_grad_I_image(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )
        img_scatter_stress = self.get_scatter_stress_image(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )

        # merge images together
        img = vu.merge_images([[img_scatter_grad_psi_grad_I, img_scatter_stress]])
        plt.close("all")
        return img  # PIL image
