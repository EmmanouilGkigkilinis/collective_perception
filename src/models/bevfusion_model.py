import torch
import torch.nn as nn
import pytorch_lightning as pl
from src.utils.architecture.bevfusion import BEVFusionV2X
from src.utils.util_fn import read_jpg, read_pcd
import logging

from src.utils.centerpoint_utils import encode_gt_batches
from src.utils.losses.losses import detection_3d_loss
from src.utils.metrics.metrics_3d_detection import Detection3DMetrics

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
        output_bev_range_H,
        output_bev_range_W,
        output_bev_range_xmin,
        output_bev_range_ymin,
        metres_per_cell , 
        num_classes, 
        lr=1e-4,
        weight_decay=1e-4,
    ):
        super().__init__()

        self.output_bev_range_H = output_bev_range_H
        self.output_bev_range_W = output_bev_range_W
        self.output_bev_range_xmin = output_bev_range_xmin
        self.output_bev_range_ymin = output_bev_range_ymin
        self.meters_per_cell = metres_per_cell
        self.num_classes = num_classes 
        

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
                                    device=self.device,
                                    num_classes=num_classes )
        
        self.model=self.model.to(self.device)

        self.lr = lr
        self.weight_decay = weight_decay

        self.save_hyperparameters(ignore=["model"])

        self.loss=detection_3d_loss(device = self.device )

        # Shared configuration; each instance maintains its own separate state.
        metric_kwargs = dict(
            x_min=self.output_bev_range_xmin,  # BEV minimum x in metres.
            y_min=self.output_bev_range_ymin,  # BEV minimum y in metres.
            metres_per_cell=self.meters_per_cell,  # Final detection-grid resolution.
            score_threshold=0.1,              # Minimum prediction confidence.
            thresh_iou=0.5,                   # Minimum 3D IoU for a correct match.
            num_classes=num_classes,     # Number of heatmap classes.
        )
        self.val_metrics = Detection3DMetrics(**metric_kwargs)

        self.train_metrics = Detection3DMetrics(**metric_kwargs)

    def on_train_epoch_start(self):
            print(f"\nEpoch {self.current_epoch + 1}/{self.trainer.max_epochs}")
            logging.info(f"\nEpoch {self.current_epoch + 1}/{self.trainer.max_epochs}")
            
  
    def training_step(self, batch, batch_idx):
        """
        training step batch
        """
        # batch=batch.to(self.device)
        
        logging.info("Forward pass ")
        img = batch["image"]  #fix this for batches ... 
        points = batch["points"]
        targets_metric = batch["gt_boxes"]

        targets_centerpoint = encode_gt_batches(targets_metric=targets_metric ,   #get BEV-coded targets 
                                        BEV_RANGE_HORZ=self.output_bev_range_W,
                                        BEV_RANGE_HORZ_MIN=self.output_bev_range_xmin,
                                        BEV_RANGE_VERT=self.output_bev_range_H,
                                        BEV_RANGE_VERT_MIN=self.output_bev_range_ymin,
                                        metres_per_cell=self.meters_per_cell,
                                        NUM_CLASSES=self.num_classes,
                                        device=self.device)


        #calibration parameters
        camera2lidar=batch["camera2lidar"]
        camera2ego=batch["camera2ego"]
        lidar2ego=batch["lidar2ego"]
        img_aug_matrix=batch["img_aug_matrix"]
        lidar_aug_matrix=batch["lidar_aug_matrix"]
        camera_intrinsics = batch["camera_intrinsics"]
        # frame_idx = batch["frame_idx"]


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

        final_loss,\
        heatmap_loss,\
        offset_loss, \
        height_loss, \
        size_loss, \
        rots_loss, = self.loss(outputs, targets_centerpoint)

        self.log(
            "total_train_loss",
            final_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=self._get_batch_size(batch),
        )

        self.log(
            "heatmap_train_loss",
            final_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=self._get_batch_size(batch),
        )

        self.log(
            "offset_train)_loss",
            final_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=self._get_batch_size(batch),
        )

        self.log(
            "height__train_loss",
            final_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=self._get_batch_size(batch),
        )

        self.log(
            "size__train_loss",
            final_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=self._get_batch_size(batch),
        )

        self.log(
            "rotation_train_loss",
            final_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=self._get_batch_size(batch),
        )

        for name, value in outputs.items():   #logs individual losses, step and epoch losses 
            self.log(
                f"train/{name}",
                value,
                on_step=False,
                on_epoch=True,
                batch_size=self._get_batch_size(batch),
            )

        self.train_metrics.reset()
        self.train_metrics.update(outputs, targets_metric)
        train_results = self.train_metrics.compute()
        for name, value in train_results.items():
            self.log(
                f"train_batch/{name}",
                value,
                on_step=True,
                on_epoch=False,
                prog_bar=(name == "mAP"),
                logger=True,
                batch_size=self._get_batch_size(batch),
            )

        # Release the accumulated prediction records.
        self.train_metrics.reset()


        return final_loss


    def validation_step(self, batch, batch_idx):
        """
        validation 
        """
        logging.info("Forward validation pass ")
        img = batch["image"]  #fix this for batches ... 
        points = batch["points"]
        targets = batch["gt_boxes"]

        targets_centerpoint = encode_gt_batches(targets_metric=targets ,
                                                BEV_RANGE_HORZ=self.output_bev_range_W,
                                                BEV_RANGE_HORZ_MIN=self.output_bev_range_xmin,
                                                BEV_RANGE_VERT=self.output_bev_range_H,
                                                BEV_RANGE_VERT_MIN=self.output_bev_range_ymin,
                                                metres_per_cell=self.meters_per_cell,
                                                NUM_CLASSES=self.num_classes,
                                                device=self.device)
        

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

        # loss = sum(outputs.values())
        final_loss,\
        heatmap_loss,\
        offset_loss, \
        height_loss, \
        size_loss, \
        rots_loss, = self.loss(outputs, targets_centerpoint)

        logging.info("Epoch {} , final total loss after validation is {}".format(self.current_epoch,final_loss))

        self.val_metrics.update(outputs, targets)

        self.log(
            "total_val_loss",
            final_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=self._get_batch_size(batch),
        )

        self.log(
            "heatmap_val_loss",
            heatmap_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=self._get_batch_size(batch),
        )

        self.log(
            "offset_val_loss",
            offset_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=self._get_batch_size(batch),
        )

        self.log(
            "height_val_loss",
            height_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=self._get_batch_size(batch),
        )

        self.log(
            "size_val_loss",
            size_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=self._get_batch_size(batch),
        )

        self.log(
            "rotation_val_loss",
            rots_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=self._get_batch_size(batch),
        )
 

        # x = x_min + (cx + dx) * metres_per_cell
        # y = y_min + (cy + dy) * metres_per_cell

        # # Your size targets were ordered [w, l, h].
        # width, length, height = np.exp(log_sizes)

        # yaw = np.arctan2(sin_yaw, cos_yaw)
        # draw_bboxes(
        #     gt_boxes,
        #     pred_boxes,
        #     xlim=(-20, 80),
        #     ylim=(-40, 40),
        # )

        return outputs
    
    def on_validation_epoch_start(self) -> None:
        # Clear records and GT counts, including those from sanity validation.
        self.val_metrics.reset()

    def on_validation_epoch_end(self):
        # Calculate AP using every validation batch accumulated this epoch.
        metrics = self.val_metrics.compute()

        # "AP50" assumes thresh_iou=0.5; change the name for other thresholds.
        self.log_dict({
            f"val_3d_AP50/{name}": value
            for name, value in metrics.items()
        })

        # Release accumulated records and reset GT counts.
        self.val_metrics.reset()

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
    

