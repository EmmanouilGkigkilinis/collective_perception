import logging
import json
import logging
import sys
import torch
from src.utils.architecture.encoders import LSSCameraEncoder,PointPillarsEncoder
from src.utils.architecture.util_models import BEVFusionNeck,CenterPointHead
import torch.nn as nn

from os import path as osp
# from src.utils.architecture.detector.centerpoint_head import CenterPointHead
from src.utils.architecture.detector.centerpoint_head import CenterPointHead
# from src.utils.warping import warp_infra_bev_to_vehicle
from src.utils.architecture.lss.lss import LSSTransform
from src.utils.architecture.cam_feat_extraction.model_builder import mmdet3d_model_builder
from src.utils.architecture.pointpillars.pointpillars_utils import PillarsToBEV
from src.utils.warping import warp_infra_bev_to_vehicle

logger= logging.getLogger(__file__)

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
                device,
                num_classes,
                rsu2veh,
                pipe=None ):
        super().__init__()


        self.calib_path = calib_path
        # self.camera_encoder = LSSCameraEncoder(bev_h,bev_w)
        # self.camera_feature_extractor = CameraFeaturizer()
        self.model_builder = mmdet3d_model_builder() #buidl camera backbones and necks 
        self.camera_backbone = self.model_builder.get_camera_backbone()
        self.camera_neck = self.model_builder.get_camera_neck().to(device)
        
        self.camera_encoder = LSSTransform(in_channels,  #view transform
                                            out_channels,
                                            image_size,
                                            feature_size,
                                            xbound,
                                            ybound,
                                            zbound,
                                            dbound,
                                            downsample,
                                            )
        self.pillars_to_bev = PillarsToBEV()  #lidar vehicle
        self.lidar_encoder = PointPillarsEncoder(in_channels=64) #lidar vehicle
        self.fusion = BEVFusionNeck() 
        self.head = CenterPointHead(in_channels=128,
                                    num_classes=num_classes)  # or TransFusion/Anchor3DHead
       
        self.lidar_rsu_transformation = warp_rsu()

    def forward(self, 
                img, 
                points , 
                camera_intrinsics,
                camera2lidar,
                camera2ego,
                lidar2ego,
                img_aug_matrix,
                lidar_aug_matrix):
        """
        bevfusion fwd call - 
        LSS
        POINTPILLARS
        NECK
        FUSE
        CENTERPOINT

        INPUTS
            img: [b,3,h,w]
            points: [x,y,z,intensity]


        """
        # print("CUDA available:", torch.cuda.is_available())
        # print("Model device:", next(self.head.parameters()).device)
        # print("Input device:", img.device)
        # sys.exit()

        logging.info("In bevfusion v2x got fwd img input of shape {}".format(img.shape))

        cam_features = self.camera_backbone(img)

        logging.info("In bevfusion v2x got cam features of shape {} and dim {}".format(len(cam_features),cam_features[0].shape))

        cam_features = self.camera_neck.forward(cam_features) 

        logging.info("In bevfusion v2x got cam features  NECK of len {} and shape {}".format(len(cam_features),cam_features[0].shape))

        logging.info("after gen lss fpn 1st output {}".format(cam_features[0].shape))
        logging.info("after gen lss fpn 2nd output {}".format(cam_features[1].shape))

        cam_features_bev = self.camera_encoder(  #lss model forward
            img=cam_features,
            points=points , 
            camera_intrinsics = camera_intrinsics, 
            camera2lidar = camera2lidar , 
            camera2ego=camera2ego, 
            lidar2ego=lidar2ego, 
            img_aug_matrix=img_aug_matrix, 
            lidar_aug_matrix = lidar_aug_matrix
        )

        logging.info("cam_features_bev output of encoder shape {}".format(cam_features_bev.shape))

        #apply_transform(points)
        #infra_features_veh_bev = self.lidar_encoder(points)

        logging.info("point cloud input to encoder of type {}".format(type(points)))

        pseudo_lidar_img = self.pillars_to_bev(points)
        infra_bev = self.lidar_encoder(pseudo_lidar_img)

        # ----------------------------------------
        # Infrastructure -> Vehicle transform               #apply in relevant branch only
        # ----------------------------------------

        # trans = transform(
        #     from_coord="Infrastructure_lidar",
        #     to_coord="Vehicle_lidar",
        # )

        # R, t = trans.get_rot_trans()

        # ----------------------------------------
        # Spatial alignment
        # ----------------------------------------

        R_inf_to_veh , t_inf_to_veh = decouple(rsu2veh)  #implemnet

        infra_bev_aligned = warp_infra_bev_to_vehicle(
            infra_bev=infra_bev,
            R_inf_to_veh=R_inf_to_veh,
            t_inf_to_veh=t_inf_to_veh,
            xbound=self.xbound,
            ybound=self.ybound,
        )

        # lidar_bev = warp_infra_bev_to_vehicle(lidar_bev, trans)

        fused_bev = self.fusion(cam_features_bev, infra_bev)

        pred = self.head(fused_bev)

        return pred
    
    

