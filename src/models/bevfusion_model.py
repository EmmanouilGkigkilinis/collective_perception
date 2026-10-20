import torch
import torch.nn as nn
import pytorch_lightning as pl
from src.utils.architecture.bevfusion import BEVFusionV2X
from src.utils.util_fn import read_jpg, read_pcd
import logging
import time
from src.utils.centerpoint_utils import encode_gt_batches
from src.utils.losses.losses import detection_3d_loss
from src.utils.metrics.metrics_3d_detection import Detection3DMetrics
from src.tests.viz_boxes import viz_boxes
from src.tests.viz_boxes_2 import viz_boxes as viz_boxes_2

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
        flag,
        epoch_viz,
        score_threshold,
        thresh_iou,
        epoch_viz_step,
        lr=1e-4,
        weight_decay=1e-4,
    ):
        super().__init__()

        self.output_bev_range_H = output_bev_range_H
        self.output_bev_range_W = output_bev_range_W
        self.output_bev_range_xmin = output_bev_range_xmin
        self.output_bev_range_ymin = output_bev_range_ymin
        self.meters_per_cell = metres_per_cell
        self.score_threshold=score_threshold
        self.num_classes = num_classes 
        self.epoch_viz_step=epoch_viz_step
        self.flag=flag
        self.epoch_viz=epoch_viz
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
        self.val_batch_metrics=Detection3DMetrics(**metric_kwargs)
        self.train_metrics = Detection3DMetrics(**metric_kwargs)

    def on_train_epoch_start(self):
            print(f"\nEpoch {self.current_epoch + 1}/{self.trainer.max_epochs}")
            logging.info(f"\nEpoch {self.current_epoch + 1}/{self.trainer.max_epochs}")
            
  
    def training_step(self, batch, batch_idx):
        """
        training step batch
        """
        # batch=batch.to(self.device)
        start = time.perf_counter()


        logging.info("Forward pass ")
        img = batch["image"]  #fix this for batches ... 
        points = batch["points"]
        targets_metric = batch["gt_boxes"]

        start3=time.perf_counter()
        targets_centerpoint = encode_gt_batches(targets_metric=targets_metric ,   #get BEV-coded targets 
                                        BEV_RANGE_HORZ=self.output_bev_range_W,
                                        BEV_RANGE_HORZ_MIN=self.output_bev_range_xmin,
                                        BEV_RANGE_VERT=self.output_bev_range_H,
                                        BEV_RANGE_VERT_MIN=self.output_bev_range_ymin,
                                        metres_per_cell=self.meters_per_cell,
                                        NUM_CLASSES=self.num_classes,
                                        device=self.device)
        elapsed3 = time.perf_counter() - start3

        logging.info(f"Elapsed time in encode_gt_batches of training step {elapsed3:.3f} at epoch {self.current_epoch}")
        print(f"Elapsed time in encode_gt_batches training step {elapsed3:.3f} at epoch {self.current_epoch} and batch idx {batch_idx}")


        #calibration parameters
        camera2lidar=batch["camera2lidar"]
        camera2ego=batch["camera2ego"]
        lidar2ego=batch["lidar2ego"]
        img_aug_matrix=batch["img_aug_matrix"]
        lidar_aug_matrix=batch["lidar_aug_matrix"]
        camera_intrinsics = batch["camera_intrinsics"]
        # frame_idx = batch["frame_idx"]


        start2=time.perf_counter()
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
        elapsed2 = time.perf_counter() - start2

        logging.info(f"Elapsed time in self.model() of training step {elapsed2:.3f} at epoch {self.current_epoch}")
        print(f"Elapsed time in self.model() training step {elapsed2:.3f} at epoch {self.current_epoch} and batch idx {batch_idx}")

        # if self.flag and self.current_epoch==self.epoch_viz and self.epoch_vize_step==None:
        #     viz_boxes(predictions = outputs , 
        #                 targets = batch,
        #                 pc = points,
        #                 score_threshold = self.score_threshold,
        #                 metres_per_cell = self.meters_per_cell,
        #                 x_min = self.output_bev_range_xmin,
        #                 y_min = self.output_bev_range_ymin,desc="{}".format(self.current_epoch),save_path=None)  #vizualize pred
        # elif self.flag and (self.epoch_viz_step is not None) and (self.current_epoch %  self.epoch_viz_step  )==0 :
        #     print("printing boxes")
        #     viz_boxes(predictions = outputs ,           
        #                 targets = batch,
        #                 pc = points,
        #                 score_threshold = self.score_threshold,
        #                 metres_per_cell = self.meters_per_cell,
        #                 x_min = self.output_bev_range_xmin,
        #                 y_min = self.output_bev_range_ymin,
        #                 desc="{}".format(self.current_epoch),
        #                 save_path="/mnt/e/dev/hidden/projects/detection_dairv2x/logging/io/"
        #                 )  #vizualize pred
        

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
            "heatmap_train__step_loss",
            heatmap_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=self._get_batch_size(batch),
        )

        self.log(
            "offset_train_step_loss",
            offset_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=self._get_batch_size(batch),
        )

        self.log(
            "height__train_step_loss",
            height_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=self._get_batch_size(batch),
        )

        self.log(
            "size__train_step_loss",
            size_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=self._get_batch_size(batch),
        )

        self.log(
            "rotation_train_step_loss",
            rots_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=self._get_batch_size(batch),
        )

        start4=time.perf_counter()

        # for name, value in outputs.items():   #logs individual losses, step and epoch losses 
        #     self.log(
        #         f"train/{name}",
        #         value,
        #         on_step=False,
        #         on_epoch=True,
        #         batch_size=self._get_batch_size(batch),
        #     )

        self.train_metrics.reset()
        self.train_metrics.update(outputs, targets_metric)
        train_metrics_values, train_metrics_values_2 = self.train_metrics.compute()
        for name, value in train_metrics_values.items():
            self.log(
                f"train_batch_AP/{name}",
                value,
                on_step=True,
                on_epoch=False,
                prog_bar=(name == "mAP"),
                logger=True,
                batch_size=self._get_batch_size(batch),
            )
        # for name, value in train_metrics_values_2.items():
        #     self.log(
        #         f"train_batch/{name}",
        #         value,
        #         on_step=True,
        #         on_epoch=False,
        #         prog_bar=(name == "mAP"),
        #         logger=True,
        #         batch_size=self._get_batch_size(batch),
        #     )
        

        # Release the accumulated prediction records.
        self.train_metrics.reset()
        elapsed4 = time.perf_counter() - start4
        logging.info(f"Elapsed time in logging  training step {elapsed4:.3f} at epoch {self.current_epoch}")
        print(f"Elapsed time in logging training step {elapsed4:.3f} at epoch {self.current_epoch} and batch idx {batch_idx}")


        elapsed = time.perf_counter() - start

        logging.info(f"Elapsed time in forward call of training step {elapsed:.3f} at epoch {self.current_epoch}")
        print(f"Elapsed time in forward call training step {elapsed:.3f} at epoch {self.current_epoch} and batch idx {batch_idx}")

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
            on_step=True,
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

        if self.flag and self.current_epoch==self.epoch_viz and self.epoch_vize_step==None:
            viz_boxes(predictions = outputs , 
                        targets = batch,
                        pc = points,
                        score_threshold = self.score_threshold,
                        metres_per_cell = self.meters_per_cell,
                        x_min = self.output_bev_range_xmin,
                        y_min = self.output_bev_range_ymin,desc="{}".format(self.current_epoch),save_path=None)  #vizualize pred
        elif self.flag and (self.epoch_viz_step is not None) and (self.current_epoch %  self.epoch_viz_step  )==0 :
            print("printing validation boxes")
            viz_boxes(predictions = outputs ,           
                        targets = batch,
                        pc = points,
                        score_threshold = self.score_threshold,
                        metres_per_cell = self.meters_per_cell,
                        x_min = self.output_bev_range_xmin,
                        y_min = self.output_bev_range_ymin,
                        desc="validation-{}".format(self.current_epoch),
                        save_path="/mnt/e/dev/hidden/projects/detection_dairv2x/logging/io/"
                        )  #vizualize pred

        # Diagnostic: evaluate exactly these validation predictions independently.
        self.val_batch_metrics.reset()
        self.val_batch_metrics.update(outputs, targets)
        batch_metrics , batch_metrics_2 = self.val_batch_metrics.compute()


        
        
        # print(
        #     f"Validation batch {batch_idx}: "
        #     f"mAP={batch_metrics['mAP'].item():.8f}"
        # )
        self.log_dict({
            f"val_3d_AP50_batch/{name}": value
            for name, value in batch_metrics.items()
        })

        self.val_batch_metrics.reset()


        return outputs
    
    def on_validation_epoch_start(self) -> None:
        # Clear records and GT counts, including those from sanity validation.
        self.val_metrics.reset()

    def on_validation_epoch_end(self):
        # Calculate AP using every validation batch accumulated this epoch.
        metrics,_ = self.val_metrics.compute()

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
    

