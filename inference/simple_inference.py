from pathlib import Path
import sys
import torch
from tqdm import tqdm
from src.models.bevfusion_model import BEVFusionV2X
from src.utils.centerpoint_utils import get_decoded_bbox_from_pred, nms_boxes
from src.models.bevfusion_model import BEVFusionLightningModule
from src.tests.viz_boxes import viz_boxes


def inference_loader():
    """
    generator

    #model ingested by getitem:
        points,imaage,camera_intrinsics,camera2lidar,camera2ego,lidar2ego,img_aug_matrix
    """

    image_root=Path("/mnt/e/dev/hidden/data/iccs_collaborative_dataset/data__livox_new/reconstructed/recording_0000/images/")
    pcd_root=Path("/mnt/e/dev/hidden/data/iccs_collaborative_dataset/data__livox_new/reconstructed/recording_0000/velo/")

    for img_path,pcd_path in zip(image_root.rglob("*.npy"), pcd_root.rglob("*.npy")):

        print(f"Reading image at path {img_path}")
        print(f"Reading pcd at path {pcd_path}")

        pcd=read_npy(pcd_path)
        img=read_npy_image(img_path)
        camera_intrinsics,camera2lidar=get_calib()

        K = torch.tensor(
            camera_intrinsics,
            dtype=torch.float32,
        ).reshape(3, 3)

        # BEVFusion uses homogeneous 4x4 calibration matrices
        K4 = torch.eye(4, dtype=torch.float32)
        K4[:3, :3] = K

        # N = 1 camera
        camera_intrinsics = K4.unsqueeze(0).to("cuda:0")

        camera2lidar = camera2lidar.unsqueeze(0).unsqueeze(0).to("cuda:0")

        camera2ego=camera2lidar.clone()
        lidar2ego=torch.eye(4).to("cuda:0")
        img_aug_matrix = torch.eye(
                4,
                dtype=torch.float32,
            ).to("cuda:0")
        lidar_aug_matrix = torch.eye(
                4,
                dtype=torch.float32,
        ).to("cuda:0")

        batch = {"image":img,
                 "points":pcd,
                 "camera2lidar":camera2lidar,
                 "camera_intrinsics":camera_intrinsics,
                 "camera2ego":camera2ego,
                 "lidar2ego":lidar2ego,
                 "img_aug_matrix":img_aug_matrix,
                 "lidar_aug_matrix":lidar_aug_matrix}

        yield batch


import numpy as np
import torch

def read_npy(path_npy):
    """Read point cloud as [N, 4]: x, y, z, intensity."""
    pc = np.load(path_npy, allow_pickle=False).reshape(-1)

    points = np.stack(
        [
            pc["x"],
            pc["y"],
            pc["z"],
            pc["intensity"] / 256.0,
        ],
        axis=1,
    ).astype(np.float32)

    # Remove points containing NaN or infinity.
    valid = np.isfinite(points).all(axis=1)

    points = points[valid]

    # point_times = pc["time"][valid].copy()

    # Example: 000000001_1791465486989885208.npy
    # timestamp_ns = int(Path(path_npy).stem.rsplit("_", 1)[1])

    print(f"point cloud is {pc}")

    return torch.from_numpy(points).to("cuda:0")


from pathlib import Path
import numpy as np

def read_npy_image(path_npy):
    image = np.load(path_npy, allow_pickle=False)
    # timestamp_ns = int(Path(path_npy).stem.rsplit("_", 1)[1])

    print(f"Image shape is {image.shape}")

    return torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0).to("cuda:0")

import json
import numpy as np
from scipy.spatial.transform import Rotation

def get_calib():
    camera_distortion = []
    calibration_file = "/mnt/e/dev/hidden/data/iccs_collaborative_dataset/dataset/reconstructed/calib.json"
    with open(calibration_file, "r") as f:
        calib = json.load(f)

    fx, fy, cx, cy = calib["camera"]["intrinsics"]

    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1],
    ], dtype=np.float64)  

    distortion = np.array(
        calib["camera"]["distortion_coeffs"],
        dtype=np.float64,
    )

    values = np.array(calib["results"]["T_lidar_camera"])

    translation = values[:3]
    quaternion = values[3:]  # scipy expects [qx, qy, qz, qw]

    camera_to_lidar = np.eye(4, dtype=np.float64)
    camera_to_lidar[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    camera_to_lidar[:3, 3] = translation

    lidar_to_camera=camera_to_lidar

    camera2lidar = np.linalg.inv(lidar_to_camera)

    print("Camera intrinsic:\n", K)
    print("Camera to LiDAR:\n", camera_to_lidar)
    print("LiDAR to camera:\n", lidar_to_camera)

    return K , torch.tensor(camera2lidar,dtype=torch.float32,)

if __name__=="__main__":
    class_names=["Car","Truck","Van","Bus","Pedestrian","Cyclist","Tricyclist",     "Motorcyclist","Barrowlist","Trafficcone"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # mylighnting_module=BEVFusionLightningModule()

    ckpt="/mnt/e/dev/hidden/projects/detection_dairv2x/checkpoints/epoch=epoch=29-val=val=0.0000-name=full_train_10.ckpt"

    model = BEVFusionLightningModule.load_from_checkpoint(
        ckpt,
        map_location="cpu",
        # Add constructor arguments here if not saved in the checkpoint.
        flag=False,
        epoch_viz=None,
        score_threshold=0.1,
        thresh_iou=0.5,
        epoch_viz_step=None,
    )
    model = model.to(device)
    model.eval()

    metres_per_cell=1
    x_min=-150
    y_min=-150

    with torch.no_grad():
        for idx,batch in tqdm(enumerate(inference_loader())):

            images = batch["image"].to(device)
            import torch.nn.functional as F

            # images: [B, 3, H, W]
            original_h, original_w = images.shape[-2:]

            # Use the exact image_size from your training configuration.
            train_h, train_w = (1080,1920)

            images = F.interpolate(
                images,
                size=(train_h, train_w),
                mode="bilinear",
                align_corners=False,
            )

            # Assumes one point-cloud tensor per sample.
            # points = [pc.to(device) for pc in batch["points"]]
            points = batch["points"].unsqueeze(0).to(device)

            camera2lidar=batch["camera2lidar"]
            camera2ego=batch["camera2ego"]
            camera_intrinsics=batch["camera_intrinsics"]
            lidar2ego=batch["lidar2ego"]
            img_aug_matrix=batch["img_aug_matrix"]
            lidar_aug_matrix=batch["lidar_aug_matrix"]

            # Match this call to your model's forward() signature.
            predictions = model.model(img=images,points=points,
                                      camera_intrinsics=camera_intrinsics,
                                      camera2lidar=camera2lidar,
                                      camera2ego=camera2ego,
                                      lidar2ego=lidar2ego,
                                      img_aug_matrix=img_aug_matrix,
                                      lidar_aug_matrix=lidar_aug_matrix,)

            results = get_decoded_bbox_from_pred(
                predictions,
                score_threshold=0.1,
                metres_per_cell=metres_per_cell,
                x_min=x_min,
                y_min=y_min,
            )

            # Apply your existing NMS here if decoding does not include it.

            nms_out = nms_boxes(results)

            # print(f"total results bboxc output is: {nms_out['bboxes'].shape}")

            # for batch_idx, result in enumerate(results):
            #         print(f"Batch item: {batch_idx}")

            #         for box, score, label in zip(
            #             result["boxes"],
            #             result["scores"],
            #             result["labels"],
            #         ):
            #             print(
            #                 f"class={label.item()} score={score.item():.3f} "
            #                 f"xy={box[:2].tolist()} "
            #                 f"lwh={box[3:6].tolist()} "
            #                 f"yaw={box[6].item():.3f}"
            #             )
            #             sys.exit(0)
                    


            labels_tot=[]
            for i, result in enumerate(nms_out):    
                boxes = result["boxes"].detach().cpu().numpy()
                scores = result["scores"].detach().cpu().numpy()
                labels = result["labels"].detach().cpu().numpy()


                labels_tot.append(labels)
                print(f"num boxes results {i}:{boxes.shape}")
                for j in range(len(class_names)):
                    print(f"class results {j}:{class_names[labels[i]]}")

            pedestrians=[]
            motorcyclists=[]
            vehicles=[]
            # for i in labels_tot:
            #     if i==


            viz_boxes(  predictions ,
                        targets=None,
                        pc=points,
                        # points=points,
                        score_threshold=0.1,
                        metres_per_cell=metres_per_cell,
                        x_min=x_min,
                        y_min=y_min,
                        save_path="/mnt/e/dev/hidden/projects/detection_dairv2x/logging/io_iccs/",
                        desc="iccs_inference")
                                # Project boxes into the original camera image and save.