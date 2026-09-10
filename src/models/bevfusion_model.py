import torch
import torch.nn as nn
import pytorch_lightning as pl


class BEVFusionLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr=1e-4,
        weight_decay=1e-4,
    ):
        super().__init__()

        self.model = model

        self.lr = lr
        self.weight_decay = weight_decay

        self.save_hyperparameters(ignore=["model"])


    def forward(self, batch):
        """
        Inference forward pass.

        batch can contain for example:
            batch["image"]
            batch["points"]
            batch["camera_intrinsics"]
            batch["camera_extrinsics"]
            batch["lidar_extrinsics"]
            batch["gt_boxes"]
            batch["gt_labels"]
        """
        return self.model(batch)


    def training_step(self, batch, batch_idx):

        outputs = self.model(
            batch,
            return_loss=True,
        )

        # Example:
        # outputs = {
        #     "loss_cls": ...,
        #     "loss_bbox": ...,
        #     "loss_heatmap": ...
        # }

        loss = sum(outputs.values())

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