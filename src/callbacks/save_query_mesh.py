from pathlib import Path

import lightning as L
import numpy as np
from optree import PyTree

import src.datasets.pytree_utils as ptu
import src.utils.icon_core_utils as cu


class SaveQueryMesh(L.Callback):
    def __init__(
        self,
        dirpath: str,
    ) -> None:
        super().__init__()
        self.dirpath = dirpath

    def on_validation_batch_end(self, trainer, pl_module, outputs: dict, batch: PyTree, batch_idx, dataloader_idx=0):
        """Cache valid batch outputs. Save querymesh to plot results."""
        # Only save first 20 batches
        if batch_idx >= 20:
            return
        dataset_name = cu.get_dataset_name(pl_module.cfg.data.valid, dataloader_idx)
        valid_dirpath = Path(self.dirpath) / "valid" / f"step_{trainer.global_step}" / dataset_name
        self._save_querymesh(valid_dirpath, batch, outputs, trainer.global_rank, batch_idx)

    def on_test_batch_end(self, trainer, pl_module, outputs: dict, batch: PyTree, batch_idx, dataloader_idx=0):
        """Cache valid batch outputs. Save querymesh to plot results."""
        # Only save first 20 batches
        if batch_idx >= 20:
            return
        dataset_name = cu.get_dataset_name(pl_module.cfg.data.test, dataloader_idx)
        test_dirpath = Path(self.dirpath) / "test" / f"step_{trainer.global_step}" / dataset_name
        self._save_querymesh(test_dirpath, batch, outputs, trainer.global_rank, batch_idx)

    def _save_querymesh(self, dirpath: Path, batch: PyTree, outputs: dict, rank: int, batch_idx: int) -> None:
        dirpath.mkdir(parents=True, exist_ok=True)

        # save descriptions
        full_path = dirpath / f"description_rank{rank}.txt"
        description = ptu.get_discription_list(batch)  # [str] * bs
        with open(full_path, "a") as f:
            for desc in description:
                f.write(desc + "\n")

        for i, mesh_dict in enumerate(ptu.to_numpy(outputs["query_mesh"])):
            if "mesh" in mesh_dict:
                mesh_data = np.array(mesh_dict["mesh"])
                npy_path = dirpath / f"mesh_rank{rank}_batch{batch_idx}_item{i}.npy"
                np.save(npy_path, mesh_data)
                print(f"Saved {npy_path}, shape={mesh_data.shape}")

            if "X" in mesh_dict:
                X_data = np.array(mesh_dict["X"])
                npy_path = dirpath / f"X_rank{rank}_batch{batch_idx}_item{i}.npy"
                np.save(npy_path, X_data)
                print(f"Saved {npy_path}, shape={X_data.shape}")

            if "F" in mesh_dict:
                F_data = np.array(mesh_dict["F"])
                npy_path = dirpath / f"F_rank{rank}_batch{batch_idx}_item{i}.npy"
                np.save(npy_path, F_data)
                print(f"Saved {npy_path}, shape={F_data.shape}")
