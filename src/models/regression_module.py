from typing import Any, Dict, List, Tuple

import hydra
import numpy as np
import pandas as pd
import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import JaccardIndex
from torchmetrics.classification.accuracy import Accuracy

from src.metrics.compute_metrics import compute_metrics
from src.models.components.utils import Masked_MAE, Masked_nMAE, Masked_RMSE
from src.utils import mkdir, pylogger
from src.utils.data import Window_writer_rasterio
from src.utils.geoaffine import Interpolation, reproject_to_tcell_rasterize

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def write_to_rasterio(wwr, preds_np, metas, poly_aoi, overlap=-1.0):
    """Writes the predicted scores to a raster using the rasterio library.

    Args:
        wwr (rasterio.io.DatasetWriter): The rasterio dataset writer object.
        preds_np (numpy.ndarray): The predicted scores.
        metas (list): List of metadata dictionaries.
        poly_aoi (shapely.geometry.Polygon): The area of interest polygon.

    Returns:
        None
    """
    for scores, meta in zip(preds_np, metas):
        window_icell = meta["window"]
        safft_icell_to_tcell = meta["safft_icell_to_tcell"]
        safft_world_to_icell = meta["safft_world_to_icell"]
        square = meta["square"]
        if overlap > 0.0:
            # extract the center part of the square (square is an shapely polygon)
            square_side = square.area**0.5
            square = square.buffer(-overlap * square_side / 2.0)

        goodpoly_world = poly_aoi & square
        if goodpoly_world.is_empty:
            continue
        scores_icell = np.ma.stack(
            [
                reproject_to_tcell_rasterize(
                    score,
                    window_icell,
                    safft_icell_to_tcell,
                    goodpoly_world,
                    safft_world_to_icell,
                    Interpolation.LINEAR,
                )
                for score in scores
            ]
        )
        wwr.write(scores_icell, window_icell)


class RegressionModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        compile: bool,
        aux_loss_factor,
        scheduler: torch.optim.lr_scheduler = None,
        warmup_scheduler: torch.optim.lr_scheduler = None,
        metric_monitored: str = None,
        loss="l2",
        loss_ce_weight: torch.tensor = None,
        activation="none",
        num_classes: int = 1,
        test_overlap=0.0,
        save_freq: int = 1,
        save_eval_only: bool = False,
    ) -> None:
        """Initialize a `segmentation Module`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net

        self.save_freq = save_freq
        self.save_eval_only = save_eval_only

        # loss function
        if loss == "l2":
            self.criterion = torch.nn.MSELoss(reduction="mean")
        elif loss == "l1":
            self.criterion = torch.nn.L1Loss(reduction="mean")
        elif loss == "Huber":
            self.criterion = torch.nn.HuberLoss(reduction="mean")

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.eval_train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for MAE
        self.train_MAE = Masked_MAE()
        self.eval_train_MAE = Masked_MAE()
        self.val_MAE = Masked_MAE()
        self.test_MAE = Masked_MAE()

        # for RMSE
        self.train_RMSE = Masked_RMSE()
        self.eval_train_RMSE = Masked_RMSE()
        self.val_RMSE = Masked_RMSE()
        self.test_RMSE = Masked_RMSE()

        # for nMAE
        self.train_nMAE = Masked_nMAE()
        self.eval_train_nMAE = Masked_nMAE()
        self.val_nMAE = Masked_nMAE()
        self.test_nMAE = Masked_nMAE()

        self.last_aoi_name = None

        try:
            self.output_dir = (
                hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            )
        except AttributeError:
            self.output_dir = "./logs"
        mkdir(self.output_dir + "/preds")

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            # self.net = torch.compile(self.net, backend="eager")
            # self.net = torch.compile(self.net, backend="aot_eager")
            self.net = torch.compile(self.net, backend="inductor")

    def get_mask_from_targets(self, targets, mask_type):
        if mask_type == "train":
            mask = torch.isinf(targets)
            if len(mask.shape) == 4:
                mask = mask.any(dim=1)
        elif mask_type == "eval":
            mask = torch.isinf(targets)
            if len(mask.shape) == 4:
                mask = mask.any(dim=1)
            mask = mask.logical_or(targets[:, 1, :, :] == 5)
        else:
            raise ValueError(f"mask_type={mask_type} not recognized")
        return mask.unsqueeze(1)

    def forward(self, x: torch.Tensor, metas: List = None) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x, metas=metas)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_MAE.reset()
        self.val_RMSE.reset()
        self.val_nMAE.reset()

        self.net.train()

    def model_step(self, batch, overlap=-1.0):
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        data, targets, metas = batch
        output = self.forward(data, metas=metas)

        # target is in dm
        targets = targets / 10.0

        if self.hparams.activation == "none":
            output = output["out"]
        elif self.hparams.activation == "relu":
            output = torch.relu(output["out"])
        elif self.hparams.activation == "softplus":
            output = torch.nn.functional.softplus(output["out"])
        output_full = output
        if overlap > 0.0:
            # only keep the center of the image
            h, w = data.shape[-2:]
            new_h, new_w = h - int(h * overlap), h - int(w * overlap)
            h_start, w_start = int(h * overlap) // 2, int(w * overlap) // 2
            output = output[
                ..., h_start : h_start + new_h, w_start : w_start + new_w
            ]
            targets = targets[
                ..., h_start : h_start + new_h, w_start : w_start + new_w
            ]

        # Set all not data to zero
        mask = self.get_mask_from_targets(targets, "train")
        pure_targets = targets[:, :1, :, :]
        masked_targets = torch.where(
            mask, torch.zeros_like(pure_targets), pure_targets
        )
        masked_output = torch.where(mask, torch.zeros_like(output), output)

        loss = self.criterion(masked_output, masked_targets)
        preds, preds_full = output, output_full
        if overlap >= 0.0:
            return loss, preds, targets, metas, preds_full
        else:
            return loss, preds, targets, metas

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets, _ = self.model_step(batch)
        mask = self.get_mask_from_targets(targets, "eval")
        targets = targets[:, :1, :, :]

        # update and log metrics
        self.train_loss(loss)
        self.train_MAE(preds, targets, mask)
        self.train_RMSE(preds, targets, mask)
        self.train_nMAE(preds, targets, mask)
        self.log(
            "train/loss",
            self.train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/MAE",
            self.train_MAE,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/RMSE",
            self.train_RMSE,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/nMAE",
            self.train_nMAE,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        # return loss or backpropagation will fail
        return loss

    def on_validation_start(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        # Print metrics to info
        log.info(f"epoch={self.trainer.current_epoch}")
        log.info(
            f"train/loss={self.train_loss.compute()}, train/MAE={self.train_MAE.compute()}, train/RMSE={self.train_RMSE.compute()}, train/nMAE={self.train_nMAE.compute()}"
        )

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        # common per aoi
        aoi_name = batch[2][0]["aoi_name"]
        gf_aoi = self.trainer.datamodule.gf_aois.loc[aoi_name]
        poly_aoi = gf_aoi["geometry"]
        split = gf_aoi["split"]
        raster_targets = self.trainer.datamodule.raster_targets
        epoch = self.trainer.current_epoch

        # all the righting to raster is done on rank 0
        new_aoi = self.last_aoi_name != aoi_name
        correct_split = not self.save_eval_only or split == "val"
        correct_epoch = (self.trainer.max_epochs - epoch) % self.save_freq == 0
        rank_zero = self.trainer.global_rank == 0
        multi_gpu = self.trainer.world_size > 1

        if new_aoi and correct_split and correct_epoch and rank_zero:
            raster_argmax_path = (
                self.output_dir
                + f"/preds/eval_{self.trainer.current_epoch}_{aoi_name}_argmax.tif"
            )
            if self.last_aoi_name is not None:
                self.wwr.close()
            profile = raster_targets[aoi_name]["profile"]
            profile["driver"] = "GTiff"
            profile["BIGTIFF"] = "YES"
            profile["count"] = 1
            self.wwr = Window_writer_rasterio(
                raster_argmax_path, profile, "float32", indexes=[1]
            )
            self.last_aoi_name = aoi_name

        loss, preds, targets, metas = self.model_step(batch)
        mask = self.get_mask_from_targets(targets, "eval")
        targets = targets[:, :1, :, :]

        if correct_split and correct_epoch:
            if multi_gpu:
                # Gather from all workers
                preds_all = self.all_gather(preds)
                metas_all = [_ for _ in range(self.trainer.world_size)]
                torch.distributed.all_gather_object(metas_all, metas)
                preds_all = preds_all.reshape(-1, *preds_all.shape[2:])
                metas_all = [item for sublist in metas_all for item in sublist]
            else:
                preds_all = preds
                metas_all = metas

            if rank_zero:
                write_to_rasterio(
                    self.wwr,
                    preds_all.detach().cpu().numpy(),
                    metas_all,
                    poly_aoi,
                )

        # update and log metrics
        if split == "val":
            self.val_loss(loss)
            self.val_MAE(preds, targets, mask)
            self.val_RMSE(preds, targets, mask)
            self.val_nMAE(preds, targets, mask)
        elif split == "train":
            self.eval_train_loss(loss)
            self.eval_train_MAE(preds, targets, mask)
            self.eval_train_RMSE(preds, targets, mask)
            self.eval_train_nMAE(preds, targets, mask)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        # log metrics
        self.log("val/loss", self.val_loss.compute(), sync_dist=True)
        self.log("val/MAE", self.val_MAE.compute(), sync_dist=True)
        self.log("val/RMSE", self.val_RMSE.compute(), sync_dist=True)
        self.log("val/nMAE", self.val_nMAE.compute(), sync_dist=True)
        # self.log(
        #     "eval_train/loss", self.eval_train_loss.compute(), sync_dist=True
        # )
        # self.log(
        #     "eval_train/MAE", self.eval_train_MAE.compute(), sync_dist=True
        # )
        # self.log(
        #     "eval_train/RMSE", self.eval_train_RMSE.compute(), sync_dist=True
        # )
        # self.log(
        #     "eval_train/nMAE", self.eval_train_nMAE.compute(), sync_dist=True
        # )
        # Print metrics to info
        log.info(
            f"val/loss={self.val_loss.compute()}, val/MAE={self.val_MAE.compute()}, val/RMSE={self.val_RMSE.compute()}, val/nMAE={self.val_nMAE.compute()}"
        )
        # log.info(
        #     f"eval_train/loss={self.eval_train_loss.compute()}, eval_train/MAE={self.eval_train_MAE.compute()}, eval_train/RMSE={self.eval_train_RMSE.compute()}, eval_train/nMAE={self.eval_train_nMAE.compute()}"
        # )
        # reset metrics
        self.val_loss.reset()
        self.val_MAE.reset()
        self.val_RMSE.reset()
        self.val_nMAE.reset()
        self.eval_train_loss.reset()
        self.eval_train_MAE.reset()
        self.eval_train_RMSE.reset()
        self.eval_train_nMAE.reset()
        if self.last_aoi_name is not None:
            self.wwr.close()

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        # Multi gpu
        rank_zero = self.trainer.global_rank == 0
        multi_gpu = self.trainer.world_size > 1

        # common per aoi
        aoi_name = batch[2][0]["aoi_name"]
        gf_aoi = self.trainer.datamodule.gf_aois.loc[aoi_name]
        poly_aoi = gf_aoi["geometry"]
        split = gf_aoi["split"]
        raster_targets = self.trainer.datamodule.raster_targets

        # create dir if needed
        mkdir(self.output_dir + "/test")

        raster_argmax_path = self.output_dir + f"/test/{aoi_name}.tif"
        if self.last_aoi_name != aoi_name:
            if self.last_aoi_name is not None:
                self.wwr.close()
            profile = raster_targets[aoi_name]["profile"]
            profile["count"] = 1
            profile["driver"] = "GTiff"
            profile["BIGTIFF"] = "YES"
            self.wwr = Window_writer_rasterio(
                raster_argmax_path, profile, "float32", indexes=[1]
            )
            self.last_aoi_name = aoi_name

        loss, preds, targets, metas, preds_full = self.model_step(
            batch, overlap=self.hparams.test_overlap
        )
        if multi_gpu:
            # Gather from all workers
            preds_all = self.all_gather(preds_full)
            metas_all = [_ for _ in range(self.trainer.world_size)]
            torch.distributed.all_gather_object(metas_all, metas)
            preds_all = preds_all.reshape(-1, *preds_all.shape[2:])
            metas_all = [item for sublist in metas_all for item in sublist]
        else:
            preds_all = preds_full
            metas_all = metas

        if rank_zero:
            write_to_rasterio(
                self.wwr,
                preds_all.detach().cpu().numpy(),
                metas_all,
                poly_aoi,
                overlap=self.hparams.test_overlap,
            )

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        self.wwr.close()

        compute_metrics(
            dataset_dir=self.trainer.datamodule.datadir,
            predictions_path={
                2021: self.output_dir + "/test/test_2021.tif",
                2022: self.output_dir + "/test/test_2022.tif",
                2023: self.output_dir + "/test/test_2023.tif",
            },
            save_dir=self.output_dir + "/metrics",
        )

    def predict_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single predict step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        # Multi gpu
        rank_zero = self.trainer.global_rank == 0
        multi_gpu = self.trainer.world_size > 1
        # common per aoi
        aoi_name = batch[2][0]["row_square"]["aoi_name"]
        gf_aoi = self.trainer.datamodule.gf_aois.loc[aoi_name]
        poly_aoi = gf_aoi["geometry"]
        split = gf_aoi["split"]
        raster_targets = self.trainer.datamodule.raster_targets

        raster_argmax_path = (
            self.output_dir + f"/preds/prediction_{aoi_name}_argmax.tif"
        )
        if self.last_aoi_name != aoi_name:
            if self.last_aoi_name is not None:
                self.wwr.close()
            profile = raster_targets[aoi_name]["profile"]
            profile["driver"] = "GTiff"
            profile["BIGTIFF"] = "YES"
            profile["count"] = 1
            self.wwr = Window_writer_rasterio(
                raster_argmax_path, profile, "float32", indexes=[1]
            )
            self.last_aoi_name = aoi_name

        loss, preds, targets, metas, preds_full = self.model_step(
            batch, overlap=self.hparams.test_overlap
        )
        preds_int = preds.argmax(axis=1)

        if multi_gpu:
            # Gather from all workers
            preds_all = self.all_gather(preds)
            metas_all = [_ for _ in range(self.trainer.world_size)]
            torch.distributed.all_gather_object(metas_all, metas)
            preds_all = preds_all.reshape(-1, *preds_all.shape[2:])
            metas_all = [item for sublist in metas_all for item in sublist]
        else:
            preds_all = preds
            metas_all = metas

        if rank_zero:
            write_to_rasterio(
                self.wwr,
                preds_all.detach().cpu().numpy(),
                metas_all,
                poly_aoi,
            )

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(
            params=self.trainer.model.parameters()
        )
        list_schedulers = []
        if self.hparams.scheduler is not None:
            assert (
                self.hparams.metric_monitored is not None
            ), "A metric must be monitored to use a scheduler."
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            list_schedulers.append(
                {
                    "scheduler": scheduler,
                    "monitor": self.hparams.metric_monitored,
                    "interval": "epoch",
                    "frequency": 1,
                }
            )
        if self.hparams.warmup_scheduler is not None:
            warmup = self.hparams.warmup_scheduler(
                optimizer=optimizer,
                total_step=self.trainer.estimated_stepping_batches,
            )
            list_schedulers.append(
                {
                    "scheduler": warmup,
                    "interval": "step",
                    "frequency": 1,
                }
            )
        print(f"{list_schedulers=}")
        return [optimizer], list_schedulers


if __name__ == "__main__":
    _ = RegressionModule(None, None, None, None)
