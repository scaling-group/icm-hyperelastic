from typing import Literal

import einops
import optree
import torch
from lightning.fabric.utilities import throughput
from omegaconf import DictConfig
from optree import PyTree
from torchmetrics import MeanMetric, MetricCollection

import src.utils.icon_core_utils as cu
from src.datasets import ice_data_multi_utils as idu

from . import ice_metric as im
from . import ice_post as ipost
from . import ice_pre as ipre
from .base_lit_module import BaseLitModule


class IceLitModule(BaseLitModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        self.train_metric_names = [
            "loss",
            "loss_eqn",
        ]

        if self.cfg.loss.compute_flops:
            self.train_flops = MeanMetric()
            self.train_metric_names.append("flops")

        self.train_metrics = MetricCollection({k: MeanMetric() for k in self.train_metric_names})

        # Use MetricCollection to group metrics
        self.metric_names = ["loss/total"]  # total loss

        if self.cfg.loss.eval_loss_eqn:
            self.metric_names.extend(
                [
                    "loss/eqn",  # loss for the equation-based training
                ]
            )

        if len(self.cfg.loss.calibrate_modes) > 0:
            self.metric_names.extend(
                [
                    "raw_scales",  # raw calibrate scales, pred vs label (bs, meshes * n_bc * 2)
                ]
            )

        for calibrate_mode in self.cfg.loss.calibrate_modes:
            # pooled calibration scales
            self.metric_names.extend(
                [
                    f"{calibrate_mode}/scales",  # pooled calibrate scales (bs, meshes)
                    f"{calibrate_mode}/scales_ratio",  # mean of calibrate scales (bs, meshes)
                    f"{calibrate_mode}/scales_rel_std",  # relative std of calibrate scales (bs, meshes)
                ]
            )
            # element-wise prediction, e.g., stress
            if self.cfg.loss.eval_error_element:
                self.metric_names.extend(
                    [
                        f"{calibrate_mode}/abs_S",  # absolute S error, (bs, meshes)
                        f"{calibrate_mode}/rel_S",  # Relative RMSE of stress, (bs, meshes)
                        f"{calibrate_mode}/rel_S_minmax",  # scaled by max-min of reference stress, (bs, meshes)
                        f"{calibrate_mode}/rel_S_area",  # area_weight RelativeRMSE of stress, (bs, meshes)
                    ]
                )
            # node-wise prediction, e.g., nodal force
            if self.cfg.loss.eval_error_node:
                self.metric_names.extend(
                    [
                        # f"{calibrate_mode}/abs_nf_bdry",  # L2 absolute error for bdry nodal force (bs, n_bc * meshes)
                        f"{calibrate_mode}/abs_nf_free",  # L2 absolute error for free nodal force (bs, meshes)
                        f"{calibrate_mode}/abs_nf_all",  # L2 absolute error for all nodal force (bs, meshes)
                        # f"{calibrate_mode}/rel_nf_bdry",  # scaled by the external force averaged on bdry nodes
                        f"{calibrate_mode}/rel_nf_free",
                        f"{calibrate_mode}/rel_nf_all",
                    ]
                )

        self.valid_metrics = torch.nn.ModuleList(
            [
                MetricCollection({k: MeanMetric() for k in self.metric_names})
                for _ in range(len(self.cfg.data.valid))  # initialize metrics for each valid_loader
            ]
        )

        self.test_metrics = torch.nn.ModuleList(
            [
                MetricCollection({k: MeanMetric() for k in self.metric_names})
                for _ in range(len(self.cfg.data.test))  # initialize metrics for each test_loader
            ]
        )

    def network_inference(self, raw_prompt: idu.DataEqn | idu.DataMeshList, raw_queries: PyTree) -> PyTree:
        """
        Use model forward compute the gradient of psi w.r.t. I, grad_psi_grad_I, given the prompt and query.

        Args:
            raw_prompt (idu.DataEqn | idu.DataMeshList):
                if idu.DataEqn:
                    A: (bs, N_eqn, 8, 2, 2);
                    XI: (bs, N_eqn, 8, 2);
                    mask: (bs, N_eqn, 8).
                if idu.DataMeshList: will be converted to idu.DataEqn
            raw_queries (PyTree): a PyTree of the input queries, each of shape (bs, ..., 2)
        """
        if isinstance(raw_prompt, list):
            raw_prompt = idu.to_eqn(raw_prompt)  # DataEqn

        scaled_prompt, scaled_queries = ipre.scale_prompt_queries(raw_prompt, raw_queries, self.cfg)
        # convert the PyTree to a list for concat
        query_processor = ipre.QueryProcessor(scaled_queries)
        scaled_query = query_processor.flatten()

        if self.cfg.loss.output == "psi":  # the model outputs psi
            with torch.inference_mode(False):  # enable gradient calculation, even in validation
                scaled_query.requires_grad = True
                psi = self._model_forward(prompt=scaled_prompt, query=scaled_query)
                # (bs, ..., 1)
                grad_psi_grad_I = torch.autograd.grad(
                    outputs=psi,
                    inputs=scaled_query,
                    grad_outputs=torch.ones_like(psi),
                    create_graph=True,
                    # retain_graph=True,
                )[0]
                # (bs, ..., 2)

        elif self.cfg.loss.output == "grad_psi_grad_I":  # the model directly outputs grad_psi_grad_I
            grad_psi_grad_I = self._model_forward(prompt=scaled_prompt, query=scaled_query)
            # (bs, ..., 2)
        else:
            raise ValueError(f"Got unknown loss.output: {self.cfg.loss.output}")

        grad_psi_grad_Is = query_processor.unflatten(grad_psi_grad_I)

        return {
            "raw_prompt": raw_prompt,  # idu.DataEqn
            "scaled_prompt": scaled_prompt,  # idu.DataEqn
            "raw_queries": raw_queries,  # PyTree
            "scaled_queries": scaled_queries,  # PyTree
            "grad_psi_grad_Is": grad_psi_grad_Is,  # PyTree
        }

    def _loss_eqn(self, network_output: PyTree) -> torch.Tensor:
        """
        Compute the loss function for the equation-based training

        Args:
            network_output (PyTree): output of network_inference

        Returns:
            loss (torch.Tensor): the loss for equilibrium units
        """

        raw_prompt = network_output["raw_prompt"]
        scaled_prompt = network_output["scaled_prompt"]
        # the first element in each list corresponds to equilibrium units.
        grad_psi_grad_I = network_output["grad_psi_grad_Is"]["eqn"]

        # it's OK to use scaled A when calculating the loss function.
        # here we didn't use mask, since we assume padded elements has zero A.
        f = einops.einsum(
            scaled_prompt["A"] if self.cfg.loss.scale_A_in_loss else raw_prompt["A"],  # (bs, N_eqn, 8, 2, 2)
            grad_psi_grad_I,  # (bs, N_eqn, 8, 2)
            "b n e d1 d2, b n e d2 -> b n e d1",  # (bs, N_eqn, 8, 2)
        )

        # get the scale of f by batch-wise standard deviation, f_scale: (bs, )
        f_scale = torch.sqrt(torch.sum(f**2, dim=(-3, -2, -1)) / torch.sum(scaled_prompt["mask"], dim=(-2, -1)))
        scaled_f = f / f_scale[:, None, None, None]  # (bs, N_eqn, 8, 2)

        nodal_force = torch.sum(scaled_f, dim=-2)  # (bs, N_eqn, 2) sum over components
        loss = torch.mean(nodal_force**2, dim=(-1, -2))  # (bs,) keep batch dimension

        return loss

    ############ training #############

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        for metrics in self.valid_metrics:
            metrics.reset()

    def _count_flops(self, raw_prompt: idu.DataEqn, raw_queries: PyTree) -> int:
        """
        Count the FLOPs of the network inference
        """

        def model_fwd():
            return self.network_inference(raw_prompt=raw_prompt, raw_queries=raw_queries)

        def model_loss(y):
            return self._loss_eqn(y).mean()

        flops = throughput.measure_flops(self.net, model_fwd, model_loss)
        return flops

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        data: idu.DataEqn = batch["data"]

        raw_queries = {"eqn": data["XI"]}
        if self.cfg.loss.loss_boundary_weight is not None:
            raw_queries["boundary"] = [boundary["XI"] for boundary in batch["data_boundary"]]

        if self.cfg.loss.compute_flops:
            flops = self._count_flops(data, raw_queries)
            self.train_flops.update(flops)
            self.train_metrics["flops"](flops)
            self.log("train/avg_flops", self.train_flops, on_step=True, on_epoch=True)  # averaged over steps
            self.log("train/flops", self.train_metrics["flops"], on_step=True, on_epoch=False)  # per step

        output = self.network_inference(raw_prompt=data, raw_queries=raw_queries)
        loss_eqn = self._loss_eqn(network_output=output)
        self.train_metrics["loss_eqn"](loss_eqn)
        self.log("train/loss_eqn", self.train_metrics["loss_eqn"], on_step=True, on_epoch=False)

        loss = loss_eqn

        self.train_metrics["loss"](loss)
        self.log("train/loss", self.train_metrics["loss"], on_step=True, on_epoch=False)
        self.train_metrics.reset()
        return loss.mean()  # pool over batch in the end

    def get_network_output_for_eval(
        self,
        prpt_mesh_list: idu.DataMeshList,
        qury_mesh_list: idu.DataMeshList,
    ) -> PyTree:
        """
        get the predicted stress on elements and nodal forces on nodes

        Args:
            mesh_list (idu.DataMeshList): list of meshes
        Returns:
            network_output (PyTree): output of network_inference
        """

        raw_queries = {}

        # get calibration scale for ALL meshes
        if len(self.cfg.loss.calibrate_modes) > 0:
            raw_queries["calibrate"] = [[boundary["XI"] for boundary in mesh["IFdata"]] for mesh in prpt_mesh_list]

        # get element/node prediction only for eval_mesh_list
        if self.cfg.loss.eval_error_element:
            raw_queries["element"] = [mesh["I"] for mesh in qury_mesh_list]

        if self.cfg.loss.eval_error_node:
            raw_queries["node"] = [mesh["XI"] for mesh in qury_mesh_list]

        # currently only support eqn loss for full_mesh_list
        if self.cfg.loss.eval_loss_eqn:
            eqn_data = idu.to_eqn(prpt_mesh_list)
            raw_queries["eqn"] = eqn_data["XI"]
        # use full mesh_list as prompt, and infer on all raw_queries
        network_output = self.network_inference(raw_prompt=prpt_mesh_list, raw_queries=raw_queries)
        return network_output

    def get_preds_from_network_output(
        self,
        network_output: PyTree,
        prpt_mesh_list: idu.DataMeshList,
        qury_mesh_list: idu.DataMeshList,
        scale_idxs: list[int] | None,
    ) -> PyTree:
        """
        get the predicted stress on elements and nodal forces on nodes

        Args:
            network_output (PyTree): output of network_inference
            mesh_list (idu.DataMeshList): list of meshes
                                        should be the same as eval_mesh_list in get_network_output_for_eval
            calibrate_modes (list[str]): the modes of calibration
        Returns:
            preds_dict (dict[str, dict[str, list[torch.Tensor]]])
            possible keys: "raw", calibrate_modes
            for "raw", value is a dict with keys "raw_scales", "stress_Ss", "nodal_forces"
            for calibrate_modes, value corresponds to the return of calibrate()
        """
        preds_dict = {"raw": {}}
        # get the raw scales for each mesh, each boundary of full_mesh_list
        if len(self.cfg.loss.calibrate_modes) > 0:
            raw_scales = [
                [
                    ipost.get_single_calibrate_scale(gpgi, boundary, mode="raw_projected")
                    for gpgi, boundary in zip(gpgi_mesh, mesh["IFdata"], strict=True)
                ]
                for gpgi_mesh, mesh in zip(network_output["grad_psi_grad_Is"]["calibrate"], prpt_mesh_list, strict=True)
            ]
            preds_dict["raw"]["raw_scales"] = raw_scales

        # get prediction on elements, including stress_Ss and grad_psi_grad_Is.
        if self.cfg.loss.eval_error_element:
            stress_Ss = []
            grad_psi_grad_Is = []  # Note: arrange by elements
            for i, mesh in enumerate(qury_mesh_list):
                grad_psi_grad_I = network_output["grad_psi_grad_Is"]["element"][i]  # (bs, N_e, 1, 2)
                stress_S = 2 * einops.einsum(
                    grad_psi_grad_I,  # (bs, N_e, 1, 2)
                    mesh["grad_I_grad_C"],  # (bs, N_e, 1, 2, 2, 2)
                    "b e t d, b e t d d1 d2 -> b e t d1 d2",  # (bs, N_e, 1, 2, 2)
                )
                stress_Ss.append(stress_S)
                grad_psi_grad_Is.append(grad_psi_grad_I)
            preds_dict["raw"]["stress_Ss"] = stress_Ss
            preds_dict["raw"]["grad_psi_grad_Is"] = grad_psi_grad_Is

        # get prediction on nodes, including nodal forces
        if self.cfg.loss.eval_error_node:
            nodal_forces = []
            for i, mesh in enumerate(qury_mesh_list):
                grad_psi_grad_I = network_output["grad_psi_grad_Is"]["node"][i]  # (bs, N_n, 8, 2)
                # must use unsliced and unscaled A when calculate the real nodal force
                # ignore mask since padded elements has A=0
                nodal_force = einops.einsum(
                    mesh["A"],  # (bs, N_n, 8, 2, 2)
                    grad_psi_grad_I,  # (bs, N_n, 8, 2)
                    "b n e d1 d2, b n e d2 -> b n d1",  # (bs, N_n, 2)
                )
                nodal_forces.append(nodal_force)
            preds_dict["raw"]["nodal_forces"] = nodal_forces

        for calibrate_mode in self.cfg.loss.calibrate_modes:
            pooled_scales, scales_ratio, scales_rel_std = ipost.pool_calibrate_scales(raw_scales, calibrate_mode)
            preds_dict[calibrate_mode] = {
                "scales": pooled_scales,  # [(bs,)] * meshes
                "scales_ratio": scales_ratio,  # [(bs,)] * meshes
                "scales_rel_std": scales_rel_std,  # [(bs,)] * meshes
            }
            if self.cfg.loss.eval_error_element:
                calibrated_stress_Ss = ipost.calibrate(pooled_scales, stress_Ss, scale_idxs)
                preds_dict[calibrate_mode]["calibrated_stress_Ss"] = calibrated_stress_Ss
            if self.cfg.loss.eval_error_node:
                calibrated_nodal_forces = ipost.calibrate(pooled_scales, nodal_forces, scale_idxs)
                preds_dict[calibrate_mode]["calibrated_nodal_forces"] = calibrated_nodal_forces
        return preds_dict

    ############ validation #############
    def validation_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0):
        return self.eval_step(batch, batch_idx, "valid", dataloader_idx)

    def eval_step(self, batch: dict, batch_idx: int, stage: Literal["valid", "test"], dataloader_idx: int = 0):
        # if "prpt" in batch:  # hacking for parellel dataset
        prpt_mesh_list: idu.DataMeshList = batch["prpt"]
        qury_mesh_list: idu.DataMeshList = batch["qury"]
        scale_idxs = batch["scale_idxs"]

        # keep autocast for eval, in consistency with training
        network_output = self.get_network_output_for_eval(prpt_mesh_list, qury_mesh_list)
        # cast to float32 to avoid overflow in post-processing
        network_output = optree.tree_map(lambda x: x.to(torch.float32), network_output)
        # disable autocast to avoid overflow in post-processing
        with torch.autocast(device_type="cuda", enabled=False):
            return self.eval_from_network_output(
                network_output,
                prpt_mesh_list,
                qury_mesh_list,
                scale_idxs,
                batch_idx,
                stage,
                dataloader_idx,
            )

    def eval_from_network_output(
        self,
        network_output: PyTree,
        prpt_mesh_list: idu.DataMeshList,
        qury_mesh_list: idu.DataMeshList,
        scale_idxs: list[int] | None,
        batch_idx: int,
        stage: Literal["valid", "test"],
        dataloader_idx: int = 0,
    ):
        """
        update metrics and log the results

        Args:
            batch (idu.DataMeshList): evaluation batch
            batch_idx (int): batch index
            stage (Literal["valid", "test"]): stage of evaluation
            dataloader_idx (int): dataloader index

        Returns:
            dict[str, torch.Tensor]:
                preds (dict[str, torch.Tensor]): predicted values
                errors (dict[str, torch.Tensor]): errors, not pooled
                metrics (dict[str, torch.Tensor]): metrics, in the similar shape of (bs, )
        """
        preds_dict = self.get_preds_from_network_output(
            network_output,
            prpt_mesh_list=prpt_mesh_list,
            qury_mesh_list=qury_mesh_list,
            scale_idxs=scale_idxs,
        )

        preds = {}
        errors = {}
        metrics = {}
        for key, value in preds_dict["raw"].items():
            preds[f"raw/{key}"] = value
        for calibrate_mode in self.cfg.loss.calibrate_modes:
            this_pred = preds_dict[calibrate_mode]
            this_error = im.get_errors(this_pred, qury_mesh_list)
            this_metrics = im.get_metrics(this_pred, this_error, qury_mesh_list)
            for key, value in this_pred.items():
                preds[f"{calibrate_mode}/{key}"] = value
            for key, value in this_error.items():
                errors[f"{calibrate_mode}/{key}"] = value
            for key, value in this_metrics.items():
                metrics[f"{calibrate_mode}/{key}"] = value

        loss = torch.tensor(0.0)

        if self.cfg.loss.eval_loss_eqn:
            loss_eqn = self._loss_eqn(network_output=network_output)
            loss += loss_eqn
            metrics["loss/eqn"] = loss_eqn

        metrics["loss/total"] = loss

        # pred and label boundary total forces
        raw_scales_flatten = optree.tree_leaves(preds_dict["raw"]["raw_scales"])
        metrics["raw_scales"] = torch.stack(raw_scales_flatten, dim=1)  # (bs, meshes * n_bc * 2)

        if stage == "valid":
            eval_metrics = self.valid_metrics
            dataset_name = cu.get_dataset_name(self.cfg.data.valid, dataloader_idx)
        elif stage == "test":
            eval_metrics = self.test_metrics
            dataset_name = cu.get_dataset_name(self.cfg.data.test, dataloader_idx)
        else:
            raise ValueError(f"Got unknown stage: {stage}")

        for metric_name in eval_metrics[dataloader_idx]:
            eval_metrics[dataloader_idx][metric_name].update(metrics[metric_name])

        for metric_name in eval_metrics[dataloader_idx]:
            self.log(
                f"{dataset_name}/{metric_name}",
                eval_metrics[dataloader_idx][metric_name],
                on_step=False,
                on_epoch=True,
                add_dataloader_idx=False,
            )

        return {"preds": preds, "errors": errors, "metrics": metrics, "query_mesh": qury_mesh_list}

    ############ test #############
    def on_test_start(self) -> None:
        """Lightning hook that is called when testing begins."""
        for metrics in self.test_metrics:
            metrics.reset()

    def test_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0):
        return self.eval_step(batch, batch_idx, "test", dataloader_idx)
