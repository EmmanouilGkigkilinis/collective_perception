
from src.utils.architecture.encoders import LSSCameraEncoder,PointPillarsEncoder
from src.utils.architecture.util_models import BEVFusionNeck,CenterPointHead
import torch.nn as nn

class BEVFusionV2X():
    """
    DUAL BRANCH ARCHITECTURE INF LIDAR, VEH CAM
    """
    def __init__(self, pipe=None):
        super().__init__()

        self.camera_encoder = LSSCameraEncoder()
        self.lidar_encoder = PointPillarsEncoder()
        self.fusion = BEVFusionNeck()
        self.head = CenterPointHead()  # or TransFusion/Anchor3DHead

    def forward(self, img, points):
        # veh_frame = vic_frame.vehicle_frame()
   
        cam_bev = self.camera_encoder(img)
        lidar_bev = self.lidar_encoder(points)

        # trans = vic_frame.transform(
        #     from_coord="Infrastructure_lidar",
        #     to_coord="Vehicle_lidar"
        # )

        # lidar_bev = warp_infra_bev_to_vehicle(lidar_bev, trans)

        fused_bev = self.fusion(cam_bev, lidar_bev)

        pred = self.head(fused_bev)

        return pred
    

class V2XBEVFusion(nn.Module):
    def __init__(self, num_classes=3, bev_h=200, bev_w=200):
        super().__init__()

        self.camera_encoder = LSSCameraEncoder(
            in_channels=3,
            bev_channels=64,
            bev_h=bev_h,
            bev_w=bev_w,
        )

        self.lidar_encoder = PointPillarsEncoder(
            in_channels=9,
            bev_channels=64,
        )

        self.fusion = BEVFusionNeck(
            cam_channels=64,
            lidar_channels=64,
            out_channels=128,
        )

        self.head = CenterPointHead(
            in_channels=128,
            num_classes=num_classes,
        )

    def forward(self, vehicle_img, infra_lidar_pseudo_img):
        cam_bev = self.camera_encoder(vehicle_img)
        lidar_bev = self.lidar_encoder(infra_lidar_pseudo_img)

        fused_bev = self.fusion(cam_bev, lidar_bev)

        preds = self.head(fused_bev)

        return preds

