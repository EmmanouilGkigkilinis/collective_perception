
import json
import torch
from src.utils.architecture.encoders import LSSCameraEncoder,PointPillarsEncoder
from src.utils.architecture.util_models import BEVFusionNeck,CenterPointHead
import torch.nn as nn

from os import path as osp

from src.utils.warping import warp_infra_bev_to_vehicle
from src.utils.architecture.lss.lss import LSSTransform

class BEVFusionV2X(nn.Module):
    """
    DUAL BRANCH ARCHITECTURE INF LIDAR, VEH CAM
    CAM ENCODER LSS from bevfusion.py (vtransforms)

    LIDAR ENCODER pointpillars

    HEAD centerpoint
    """
    def __init__(self, 
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
                 pipe=None ):
        super().__init__()

        # self.camera_encoder = LSSCameraEncoder(bev_h,bev_w)
        self.camera_encoder = LSSTransform(in_channels,
                                            out_channels,
                                            image_size,
                                            feature_size,
                                            xbound,
                                            ybound,
                                            zbound,
                                            dbound,
                                            downsample,
                                            )
        self.lidar_encoder = PointPillarsEncoder()
        self.fusion = BEVFusionNeck()
        self.head = CenterPointHead()  # or TransFusion/Anchor3DHead

        # self.x_bound=x_bound
        # self.y_bound=y_bound


        # self.img_aug_matrix = torch.eye(
        #         4,
        #         dtype=torch.float32,
        #     )
        # self.lidar_aug_matrix = torch.eye(
        #         4,
        #         dtype=torch.float32,
        # )




    def read_calib_path(self , path:str):
        """
        str name of calibration json file , acc. to timestamp of current training sample 
        """
        with open(osp.join(path,"camera_instrinsic",path) , "r") as f:
            calib = json.load(f) 
        camera_intinsics = [calib["cam_D"] , calib["cam_K"]]

        with open(osp.join(path , "lidar_to_camera") , "r") as f:
            calib=json.load(f)
        camera2lidar=[calib["R"] , calib["t"]]
        

        return camera2lidar , camera_intinsics

    def forward(self, img, points , frame_idx):
        """
        bevfusion fwd call -
        LSS
        POINTPILLARS
        NECK
        FUSE
        CENTERPOINT
        """
        camera2lidar,camera_intrinsic = self.read_calib_path(self.calib_path , frame_idx) #read calib files
        camera2ego = camera2lidar.clone()
        lidar2ego=torch.eye(4)
        img_aug_matrix = torch.eye(
                4,
                dtype=torch.float32,
            )
        lidar_aug_matrix = torch.eye(
                4,
                dtype=torch.float32,
        )

        # veh_frame = vic_frame.vehicle_frame()
   
        #img
        # camera2ego,
        # lidar2ego,
        # camera_intrinsics,
        # camera2lidar,
        # img_aug_matrix,
        # lidar_aug_matrix,


        cam_features_bev = self.camera_encoder(img , 
                                               camera2lidar,
                                               camera2ego,
                                               camera_intrinsic,
                                               lidar2ego,
                                               lidar_aug_matrix,
                                               img_aug_matrix)

        #apply_transform(points)
        #infra_features_veh_bev = self.lidar_encoder(points)

        infra_features_bev = self.lidar_encoder(points)

        # ----------------------------------------
        # Infrastructure -> Vehicle transform
        # ----------------------------------------

        trans = transform(
            from_coord="Infrastructure_lidar",
            to_coord="Vehicle_lidar",
        )

        R, t = trans.get_rot_trans()

        # ----------------------------------------
        # Spatial alignment
        # ----------------------------------------

        infra_bev_aligned = warp_infra_bev_to_vehicle(
            infra_bev=infra_bev,
            R_inf_to_veh=R,
            t_inf_to_veh=t,
            xbound=self.xbound,
            ybound=self.ybound,
        )

        # lidar_bev = warp_infra_bev_to_vehicle(lidar_bev, trans)

        fused_bev = self.fusion(cam_bev, infra_bev_aligned)

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

