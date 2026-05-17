import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import asnumpy, rearrange

import src.utils.icon_core_utils as cu

from . import viz_utils as vu
from .viz import Viz


class VizStretchwiseErr(Viz):
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
        self.category = "stretchwise_err"

    def on_validation_epoch_start(self, trainer, pl_module):
        self.metrics_acc: dict[tuple[str, int], np.ndarray] = {}
        self.metrics_cnt: dict[tuple[str, int], int] = {}

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        for k in outputs["metrics"]:
            if "rel_S_area" not in k:
                continue

            # fill the NaN and Inf values with a number sufficiently enough (1e8)
            metric_value = torch.nan_to_num(outputs["metrics"][k], nan=1.0e8, posinf=1.0e8, neginf=1.0e8)

            if k not in self.metrics_acc:
                self.metrics_acc[(k, dataloader_idx)] = asnumpy(metric_value)
                self.metrics_cnt[(k, dataloader_idx)] = 1
            else:
                self.metrics_acc[(k, dataloader_idx)] += asnumpy(metric_value)
                self.metrics_cnt[(k, dataloader_idx)] += 1

    def on_validation_epoch_end(self, trainer, pl_module):
        figs_dict = {}
        for k, dataloader_idx in self.metrics_acc:
            fig, ax = plt.subplots()
            data = self.metrics_acc[(k, dataloader_idx)] / self.metrics_cnt[(k, dataloader_idx)]
            assert data.shape[0] == 1, "Expecting only one batch for visualization"
            data_np = rearrange(data, "... -> (...)")
            ax.plot(data_np)
            ax.set_title(f"{k} (step {trainer.global_step})")
            ax.set_xlabel("stretch index")
            ax.set_ylabel(k)
            ax.set_yscale("log")
            ax.set_ylim(top=10)

            if dataloader_idx not in figs_dict:
                figs_dict[dataloader_idx] = []
            figs_dict[dataloader_idx].append(fig)
            plt.close("all")

        for dataloader_idx, figs in figs_dict.items():
            img = vu.merge_images([figs])
            dataset_name = cu.get_dataset_name(pl_module.cfg.data.valid, dataloader_idx)
            self.log_image(
                trainer,
                img,
                key=f"{dataset_name}",
                artifact_file=f"valid/{self.__class__.__name__}/{dataset_name}/step_{trainer.global_step}.png",
            )
