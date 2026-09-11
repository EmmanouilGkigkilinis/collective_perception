import torch
import torch.nn as nn
import pytorch_lightning as pl
from src.utils.architecture.bevfusion import BEVFusionV2X
from src.utils.util_fn import read_jpg, read_pcd
import logging

logging.getLogger(__file__)

class BEVFusionLightningModule(pl.LightningModule):
    def __init__(
        self,
        lr=1e-4,
        weight_decay=1e-4,
    ):
        super().__init__()

        self.model = BEVFusionV2X()

        self.lr = lr
        self.weight_decay = weight_decay

        self.save_hyperparameters(ignore=["model"])

    def on_train_epoch_start(self):
            print(f"\nEpoch {self.current_epoch + 1}/{self.trainer.max_epochs}")
            logging.info(f"\nEpoch {self.current_epoch + 1}/{self.trainer.max_epochs}")
            
  
    def training_step(self, batch, batch_idx):
        """
        training step batch
        """
        logging.info("Forward pass ")
        img = batch["image"]  #fix this for batches ... 
        lid = batch["points"]

        targets = batch["gt_boxes"]

        outputs = self.model(
            img, lid
        )

        # Example:
        # outputs = {
        #     "loss_cls": ...,
        #     "loss_bbox": ...,
        #     "loss_heatmap": ...
        # }

        loss = loss_fn(outputs, targets)

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=self._get_batch_size(batch),
        )

        for name, value in outputs.items():
            self.log(
                f"train/{name}",
                value,
                on_step=False,
                on_epoch=True,
                batch_size=self._get_batch_size(batch),
            )

        return loss


    def validation_step(self, batch, batch_idx):

        outputs = self.model(
            batch,
            return_loss=True,
        )

        loss = sum(outputs.values())

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=self._get_batch_size(batch),
        )

        for name, value in outputs.items():
            self.log(
                f"val/{name}",
                value,
                on_step=False,
                on_epoch=True,
                batch_size=self._get_batch_size(batch),
            )

        return outputs

    def training_epoch_end(self, outputs):
        print(f"\nEpoch {self.current_epoch} train metrics:")
        for k, v in self.trainer.callback_metrics.items():
            if "train" in k:
                print(f"{k}: {float(v):.6f}")

        optimizer = self.trainer.optimizers[0]
        current_lr = optimizer.param_groups[0]["lr"]

        self.log("lr_epoch", current_lr, prog_bar=True, logger=True)


    def test_step(self, batch, batch_idx):

        predictions = self.model(
            batch,
            return_loss=False,
        )

        return predictions


    def predict_step(self, batch, batch_idx):

        return self.model(
            batch,
            return_loss=False,
        )


    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


    @staticmethod
    def _get_batch_size(batch):

        if "image" in batch:
            return batch["image"].shape[0]

        if "points" in batch:
            return len(batch["points"])

        return 1
    

