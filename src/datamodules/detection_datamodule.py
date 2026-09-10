import json
import os
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import pytorch_lightning as pl
import logging
from pathlib import Path
import pandas as pd

from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class DAIRV2X_DATASET(Dataset):
    """
    Dataset for detection in DAIRV2XC,currently for veh-side cam+lid detection
    Interface with DAIRV2XDataModule
    """
    def __init__(self, 
                 data_path_veh_cam:str,
                 data_path_veh_lid:str,
                 path_to_data_info:str,
                 path_to_gt_labels:str,
                 ):
        """
        data_path_veh_camera = DAIRV2X-C vehicle-side images 
        data_path_veh_lidar = DAIRV2X-C vehicle-side lidar

        Read dataset for vehicle only detection
        """
        super().__init__()
        self.path_to_data_info=Path(path_to_data_info)
        self.path_to_gt_labels=Path(path_to_gt_labels)


        images_list=[]
        pcd_list=[]
        data_path_veh_cam = Path(data_path_veh_cam)
        data_path_veh_lid = Path(data_path_veh_lid)

        # for image in data_path_veh_cam.rglob("*.jpg"):
        #     images_list.append(image)

        # for pcd in data_path_veh_lid.rglob("*.pcd"):
        #     pcd_list.append(pcd)

        # self.database = images_list
        # self.database_pcd = pcd_list

        self.database = self.get_veh_cam_lid_frame_id()  #construct database list from data_info of DAIRV2X 

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]   # [Tin, 2]
        y = self.Y[idx]   # [Tout, 2]

        return x, y

    def get_veh_cam_lid_frame_id(self, ):
        """ 
        generator for getting sequential dairv2x cam/lid corresponding inputs from data_info.json
        
        constructor for the detection database
        """
        data = json.load(self.path_to_data_info)
        database = [] 
        for elem in data:
            label_camera = elem["label_camera_std_path"]
            label_lidar = elem["label_lidar_std_path"]
            img_path = elem["image_path"]
            lidar_path = elem["pointcloud_path"]

            database.append((label_camera,label_lidar , img_path , lidar_path))

        return database

class DAIRV2XDataModule(pl.LightningDataModule):
    """
    datamodule for dairv2x detection 
    """
    def __init__(
        self,
        train_file,
        val_file,
        test_file,
        batch_size=128,
        num_workers=4,
        pin_memory=True,
    ):
        super().__init__()

        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):

        if stage == "fit" or stage is None:
            self.train_dataset = DAIRV2X_DATASET(
                self.train_file
            )

            self.val_dataset = DAIRV2X_DATASET(
                self.val_file
            )

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
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )