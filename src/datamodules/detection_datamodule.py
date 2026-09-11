import json
import os
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import pytorch_lightning as pl
import logging
from pathlib import Path
import pandas as pd
from os import path as osp

from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


import logging

from src.utils.util_fn import read_jpg, read_pcd
logging.getLogger(__file__)

class DAIRV2X_DATASET(Dataset):
    """
    Dataset for detection in DAIRV2XC,currently for veh-side cam+lid detection
    Interface with DAIRV2XDataModule
    """
    def __init__(self, 
                 split:list,
                 desc:str,
                 data_path_veh_cam:str,
                 data_path_veh_lid:str,
                 path_to_data_info:str,
                 path_to_gt_labels:str,
                 path_to_veh_calib:str
                 ):
        """
        data_path_veh_camera = DAIRV2X-C vehicle-side images 
        data_path_veh_lidar = DAIRV2X-C vehicle-side lidar
        path_to_data_info:str,=data_info.json, frame-label correspondsance
        path_to_gt_labels:str,  = label information (bbox) 
        path_to_veh_calib:str = calibration for vehicle frame (intrinsic, extrinsics)
        
        Read dataset for vehicle only detection
        """
        super().__init__()
        #------ 
        self.path_to_data_info=Path(path_to_data_info) #data anotations/metadata
        self.path_to_gt_labels=Path(path_to_gt_labels)
        self.path_to_veh_calib=Path(path_to_veh_calib)

        #---
        self.data_path_veh_cam = Path(data_path_veh_cam) #data paths 
        self.data_path_veh_lid = Path(data_path_veh_lid)

      
        #-----
        self.database = self.get_veh_cam_lid_frame_id(split , desc)  #construct database list from data_info of DAIRV2X 


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """
        fetcher
        """
        return self.database[idx]
    
    def parse_calibration_files(self, frame_idx):
        """
        read vehicle calibration instrinsics and extrinsics 
        """

        #instrinics
        with open(self.path_to_veh_calib / "camera_intrinsic" /frame_idx+"json","r") as f:
            data=json.load(f)
        d=data["cam_D"] #distortion
        k=data["cam_K"] #camera coeffs

        #extrinsics 
        data=[]
        with open(self.path_to_veh_calib / "lidar_to_camera" /frame_idx+"json","r") as f:
            data=json.load(f)

        t_l_c = data["translation"]
        r_l_c = data["rotation"]

        with open(self.path_to_veh_calib / "lidar_to_novatel" /frame_idx+"json","r") as f:
            data=json.load(f)

        t_l_n = data["translation"]
        r_l_n = data["rotation"]

        with open(self.path_to_veh_calib / "lidar_to_novatel" /frame_idx+"json","r") as f:
            data=json.load(f)

        t_n_w = data["translation"]
        r_n_w = data["rotation"]



    def parse_camera_label(self,label_camera):
        """
        parse object detection ground truth for vehicle camera, at one timestamp
        Read Labels file that contains all labels seen by vehilce at current timestamp 
        accorind to data_info.json and append it to output 

        returns 
        ground_truth_data: list of 2d_bboxes at current timestamp
        """
        with open(osp.join(self.path_to_gt_labels , label_camera), "r") as f:
            ground_truth_file=json.load(f)
        ground_truth_data=[]
        for obj_dict in ground_truth_file:
            ground_truth_data.append(obj_dict["2d_box"])

        return ground_truth_data
    
    def get_veh_cam_lid_frame_id(self, split:list , split_desc:str):
        """ 
        generator for getting sequential dairv2x cam/lid corresponding inputs from data_info.json
        
        constructor for the detection database. Data_info reports sequentialy all frames and
        associated labels, pcd files at each teimestamp. Cross check with labels files to find labels
        of that frame.

        inputs
            split: list of frames to be used by current split
        
        Returns
        database: dict of inputs,ground truth imgs, pcd files , bboxes respectively
        """
        with open(self.path_to_data_info, "r") as f:
            data = json.load(f)
        database = [] 
        for elem in data:
            frame_idx = elem["image_path"].split("/")[-1].replace(".jpg", "")

            if frame_idx not in split:
                continue   #keep only frame idx in current split

            label_camera = elem["label_camera_std_path"]    
            label_lidar = elem["label_lidar_std_path"]
            img_path = self.data_path_veh_cam / elem["image_path"]                  #paths
            lidar_path = self.data_path_veh_lid / elem["pointcloud_path"]

            ground_truth_veh_cam = self.parse_camera_label(label_camera)   #ground truth labels(2dbbox)

            # ------------------------
            # Camera                |
            # ------------------------
            img = read_jpg(img_path)
            # numpy: [H, W, 3], uint8, BGR
            img = torch.from_numpy(img).permute(2,0,1).float() # [3, H, W]

            # ------------------------
            # LiDAR                 |
            # ------------------------
            points = read_pcd(lidar_path)
            points = torch.from_numpy(points).float()
            # [N, 4]

            # ------------------------
            # GT
            # ------------------------


            database.append({                               #training sample 
                             "img":img , 
                             "points":points , 
                             "ground_truth_veh_cam":ground_truth_veh_cam  #2d bbox
                                                                         })  
            
        logging.info("Loaded {} data for current {} split".format(len(database[img])))

        return database


class DAIRV2XDataModule(pl.LightningDataModule):
    """
    datamodule for dairv2x detection 
    """
    def __init__(
        self,
        data_path_veh_cam:str,
        data_path_veh_lid:str,
        path_to_data_info:str,
        path_to_gt_labels:str,
        path_to_data_splits:str,
        path_to_veh_calib:str,
        batch_size=128,
        num_workers=4,
        pin_memory=True,
    ):
        super().__init__()


        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        #requirements
        self.data_path_veh_cam=data_path_veh_cam    #input paths
        self.data_path_veh_lid=data_path_veh_lid
        self.path_to_data_info=path_to_data_info    #data (paths,labels) info
        self.path_to_gt_labels=path_to_gt_labels    #gt info
        self.path_to_data_splits=path_to_data_splits #train-val-test splits info
        self.path_to_veh_calib= path_to_veh_calib
        #splitting
        self.train_splits,self.val_splits,self.test_splits = self.parse_data_splits(self.path_to_data_splits) #get splits

    def parse_data_splits(self,path_to_data_splits):
        """
        return frame idxs lists as splits 
        """
        with open(path_to_data_splits, "r") as f:
            splits = json.load(f)["vehicle_split"]

        train_splits = [x for x in splits["train"]]
        val_splits = [x for x in splits["val"]]
        test_splits = [x for x in splits["test"]]

        return train_splits,val_splits,test_splits

    def setup(self, stage=None):
        """
        setups current training stage. 
        fit:train and val
        test: test
        """
        if stage == "fit" or stage is None:
            self.train_dataset = DAIRV2X_DATASET(
                split=self.train_splits, 
                data_path_veh_cam=self.data_path_veh_cam,
                data_path_veh_lid=self.data_path_veh_lid,
                path_to_data_info=self.path_to_data_info,
                path_to_gt_labels=self.path_to_gt_labels,
                path_to_veh_calib=self.path_to_veh_calib,
                desc="train",
            )

            self.val_dataset = DAIRV2X_DATASET(
                split=self.val_splits, 
                data_path_veh_cam=self.data_path_veh_cam,
                data_path_veh_lid=self.data_path_veh_lid,
                path_to_data_info=self.path_to_data_info,
                path_to_gt_labels=self.path_to_gt_labels,
                path_to_veh_calib=self.path_to_veh_calib,
                desc="val",
            )

        #testing
        if stage == "test" or stage is None:
            self.test_dataset = DAIRV2X_DATASET(
                self.test_file
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            collate_fn=self.bevfusion_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            collate_fn=self.bevfusion_collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            collate_fn=self.bevfusion_collate_fn,
        )
    
    def bevfusion_collate_fn(batch):
            """
            simple dict based collator
            
            TODO handle pcd? 
            """
            images = torch.stack(
                [sample["image"] for sample in batch]
            )

            points = [
                sample["points"]
                for sample in batch
            ]

            gt_boxes = [
                sample["ground_truth_veh_cam"]
                for sample in batch
            ]

           
            return {
                "image": images,          # [B, 3, H, W]

                # lists because number differs per sample
                "points": points,        # list of [Ni, 4]
                "gt_boxes": gt_boxes,    # list of [Mi, box_dim]

                # "camera_intrinsics": torch.stack(
                #     [sample["camera_intrinsics"] for sample in batch]
                # ),

                # "camera_extrinsics": torch.stack(
                #     [sample["camera_extrinsics"] for sample in batch]
                # ),
            }
