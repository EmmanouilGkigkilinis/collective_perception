import torch
import torch.nn as nn
from mmdet3d.registry import MODELS
from mmdet3d.models.data_preprocessors import Det3DDataPreprocessor

class PillarsToBEV(nn.Module):
    def __init__(self):
        super().__init__()

        voxel_size = [0.5, 0.5, 20.0]
        point_cloud_range = [-150, -150, -10, 150, 150, 10]

        self.preprocessor = Det3DDataPreprocessor(
            voxel=True,
            voxel_layer=dict(
                max_num_points=32,
                point_cloud_range=point_cloud_range,
                voxel_size=voxel_size,
                max_voxels=(16000, 16000),
            ),
        )
        self.voxel_encoder = MODELS.build(dict(
            type="PillarFeatureNet",
            in_channels=4,
            feat_channels=[64],
            with_distance=False,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
        ))
        self.scatter = MODELS.build(dict(
            type="PointPillarsScatter",
            in_channels=64,
            output_shape=[600, 600],  # [y cells, x cells]
        ))

    def forward(self, points):
        # Your current batch is [B, N, 4]. MMDetection3D expects
        # one [Ni, 4] tensor per sample.
        if isinstance(points, torch.Tensor):
            points = list(points.unbind(0))

        # #--
        #  # Inspect the actual voxelizer configuration.
        # layer = self.preprocessor.voxel_layer

        # print("Actual point_cloud_range:", layer.point_cloud_range)
        # print("Actual voxel_size:", layer.voxel_size)
        # print("Actual max_voxels:", layer.max_voxels)
        # print("Training mode:", self.training)

        # for i, pc in enumerate(points):
        #     print(f"Sample {i}: shape={tuple(pc.shape)}, device={pc.device}")

        #     if pc.shape[0] == 0:
        #         print("INPUT CLOUD IS EMPTY")
        #         continue

        #     xyz = pc[:, :3]
        #     finite = torch.isfinite(xyz).all(dim=1)

        #     bounds = xyz.new_tensor(layer.point_cloud_range)
        #     inside = (
        #         finite
        #         & (xyz >= bounds[:3]).all(dim=1)
        #         & (xyz < bounds[3:]).all(dim=1)
        #     )

        #     print("Finite points:", finite.sum().item())
        #     print("Points inside voxel range:", inside.sum().item())

        #     if finite.any():
        #         print("XYZ min:", xyz[finite].min(dim=0).values.tolist())
        #         print("XYZ max:", xyz[finite].max(dim=0).values.tolist())

        # #--


        vox = self.preprocessor.voxelize(points, data_samples=None)

        # #----------------------
        # for i, pc in enumerate(points):
        #     print(f"Sample {i}: points shape={tuple(pc.shape)}")
        #     if pc.numel() > 0:
        #         print("XYZ min:", pc[:, :3].amin(dim=0))
        #         print("XYZ max:", pc[:, :3].amax(dim=0))

        # vox = self.preprocessor.voxelize(points, data_samples=None)

        # print("voxels:", vox["voxels"].shape)
        # print("num_points:", vox["num_points"].shape)
        # print("coors:", vox["coors"].shape)

        # if vox["voxels"].numel() == 0:
        #     raise RuntimeError("Voxelization returned no elements")
        # #-------------------------------

        features = self.voxel_encoder(
            vox["voxels"], vox["num_points"], vox["coors"]
        )
        return self.scatter(features, vox["coors"], len(points))
        # [B, 64, 200, 200]