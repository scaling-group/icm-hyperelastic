from pathlib import Path

import lightning as L
import numpy as np
from optree import PyTree

import src.datasets.pytree_utils as ptu
import src.utils.icon_core_utils as cu


class SaveError(L.Callback):
    def __init__(
        self,
        dirpath: str,
    ) -> None:
        super().__init__()
        self.dirpath = dirpath

    def on_validation_batch_end(self, trainer, pl_module, outputs: dict, batch: PyTree, batch_idx, dataloader_idx=0):
        """Cache valid batch outputs. Save true and predicted error to plot results."""
        # Only save first 20 batches
        if batch_idx >= 20:
            return
        dataset_name = cu.get_dataset_name(pl_module.cfg.data.valid, dataloader_idx)
        valid_dirpath = Path(self.dirpath) / "valid" / f"step_{trainer.global_step}" / dataset_name
        self._save_error(valid_dirpath, batch, outputs, trainer.global_rank, batch_idx)

    def on_test_batch_end(self, trainer, pl_module, outputs: dict, batch: PyTree, batch_idx, dataloader_idx=0):
        """Cache valid batch outputs. Save true and predicted error to plot results."""
        # Only save first 20 batches
        if batch_idx >= 20:
            return
        dataset_name = cu.get_dataset_name(pl_module.cfg.data.test, dataloader_idx)
        test_dirpath = Path(self.dirpath) / "test" / f"step_{trainer.global_step}" / dataset_name
        self._save_error(test_dirpath, batch, outputs, trainer.global_rank, batch_idx)

    def _save_error(self, dirpath: Path, batch: PyTree, outputs: dict, rank: int, batch_idx: int) -> None:
        dirpath.mkdir(parents=True, exist_ok=True)

        # save descriptions
        full_path = dirpath / f"description_rank{rank}.txt"
        description = ptu.get_discription_list(batch)  # [str] * bs
        with open(full_path, "a") as f:
            for desc in description:
                f.write(desc + "\n")

        # save error
        for key, tensor in ptu.to_numpy(outputs["errors"]).items():
            if "stress" in key:
                file_key = key.replace("/", "_")
                npy_path = dirpath / f"{file_key}_rank{rank}_batch{batch_idx}.npy"
                tensor = np.array(tensor)
                np.save(npy_path, tensor)
                print(f"Saved {npy_path}, shape = {tensor.shape}")
