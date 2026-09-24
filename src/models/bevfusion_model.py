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
        in_channels,
        out_channels,
        image_size,
        feature_size,
        xbound,
        ybound,
        zbound,
        dbound,
        downsample,
        calib_path,
        device,
        lr=1e-4,
        weight_decay=1e-4,
    ):
        super().__init__()

        # self.device=device

        self.model = BEVFusionV2X(in_channels,
                                    out_channels,
                                    image_size,
                                    feature_size,
                                    xbound,
                                    ybound,
                                    zbound,
                                    dbound,
                                    downsample,
                                    calib_path,
                                    device=self.device )
        
        # model=model.to(self.device)

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
        batch=batch.to(self.device)
        


        logging.info("Forward pass ")
        img = batch["image"]  #fix this for batches ... 
        points = batch["points"]
        targets = batch["gt_boxes"]
        frame_idx = batch["frame_idx"]


        #calibration parameters
        camera2lidar=batch["camera2lidar"]
        camera2ego=batch["camera2ego"]
        lidar2ego=batch["lidar2ego"]
        img_aug_matrix=batch["img_aug_matrix"]
        lidar_aug_matrix=batch["lidar_aug_matrix"]
        camera_intrinsics = batch["camera_intrinsics"]
        # frame_idx = batch["frame_idx"]

        import inspect

        print("\n===== MODEL DEBUG =====")
        print("model object:", self.model)
        print("model class:", type(self.model))
        print("MRO:", type(self.model).__mro__)
        print("forward:", self.model.forward)
        print("forward signature:", inspect.signature(self.model.forward))
        print("class file:", inspect.getfile(type(self.model)))
        print("forward file:", inspect.getfile(self.model.forward))
        print("=======================\n")




        outputs = self.model(
            img=img,
            points=points , 
            camera_intrinsics = camera_intrinsics,
            camera2lidar=camera2lidar , 
            camera2ego=camera2ego, 
            lidar2ego=lidar2ego, 
            img_aug_matrix=img_aug_matrix, 
            lidar_aug_matrix = lidar_aug_matrix
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
        """
        validation 
        """
        logging.info("Forward validation pass ")
        img = batch["image"]  #fix this for batches ... 
        points = batch["points"]
        targets = batch["gt_boxes"]
        

        #calibration parameters
        camera_intrinsics = batch["camera_intrinsics"]
        camera2lidar=batch["camera2lidar"]
        camera2ego=batch["camera2ego"]
        lidar2ego=batch["lidar2ego"]
        img_aug_matrix=batch["img_aug_matrix"]
        lidar_aug_matrix=batch["lidar_aug_matrix"]
        # frame_idx = batch["frame_idx"]

        outputs = self.model(  #lss model forward
            img=img,
            points=points , 
            camera_intrinsics = camera_intrinsics, 
            camera2lidar=camera2lidar , 
            camera2ego=camera2ego, 
            lidar2ego=lidar2ego, 
            img_aug_matrix=img_aug_matrix, 
            lidar_aug_matrix = lidar_aug_matrix
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
            self.model.parameters(),
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
    

