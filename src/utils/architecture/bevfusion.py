
from utils.architecture.encoders import LSSCameraEncoder,PointPillarsEncoder
from utils.architecture.util_models import BEVFusionNeck,CenterPointHead

class BEVFusionV2X(BaseModel):
    """
    DUAL BRANCH ARCHITECTURE INF LIDAR, VEH CAM
    """
    def __init__(self, args, pipe=None):
        super().__init__()
        self.args = args

        self.camera_encoder = LSSCameraEncoder()
        self.lidar_encoder = PointPillarsEncoder()
        self.fusion = BEVFusionNeck()
        self.head = CenterPointHead()  # or TransFusion/Anchor3DHead

    def forward(self, vic_frame, filt, prev_inf_frame_func=None):
        veh_frame = vic_frame.vehicle_frame()
        inf_frame = vic_frame.infrastructure_frame()

        img = veh_frame.image(data_format="array")
        points = inf_frame.point_cloud(data_format="array")

        cam_bev = self.camera_encoder(img)
        lidar_bev = self.lidar_encoder(points)

        trans = vic_frame.transform(
            from_coord="Infrastructure_lidar",
            to_coord="Vehicle_lidar"
        )

        lidar_bev = warp_infra_bev_to_vehicle(lidar_bev, trans)

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

import torch

def make_grid_cells(xbound, ybound, device=None, dtype=torch.float32):
    """
    Creates BEV cell centers in metric coordinates.

    xbound = [xmin, xmax, xstep]
    ybound = [ymin, ymax, ystep]

    Returns:
        grid: [H, W, 2]
              grid[h, w] = [x, y]
    """

    xmin, xmax, xstep = xbound
    ymin, ymax, ystep = ybound

    xs = torch.arange(xmin + xstep / 2, xmax, xstep, device=device, dtype=dtype)
    ys = torch.arange(ymin + ystep / 2, ymax, ystep, device=device, dtype=dtype)

    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    grid = torch.stack([xx, yy], dim=-1)

    return grid

class BEVWarp(nn.Module):

    def forward(self, feat, T):

        B,C,H,W = feat.shape

        grid = build_metric_grid(H,W)

        grid = transform_grid(grid,T)

        grid = normalize_grid(grid,H,W)

        return F.grid_sample(
            feat,
            grid,
            mode='bilinear',
            align_corners=True
        )
    
def warp_infra_bev_to_vehicle(
    infra_bev,
    R_inf_to_veh,
    t_inf_to_veh,
    xbound,
    ybound,
):
    """
    infra_bev: [B, C, H, W]
    R_inf_to_veh, t_inf_to_veh:
        transform from Infrastructure_lidar to Vehicle_lidar

    Returns:
        aligned infra BEV in vehicle coordinates: [B, C, H, W]
    """

    B, C, H, W = infra_bev.shape
    device = infra_bev.device
    dtype = infra_bev.dtype

    # output grid: vehicle BEV cell centers
    grid_vehicle = make_grid_cells(
        xbound=xbound,
        ybound=ybound,
        device=device,
        dtype=dtype,
    )  # [H, W, 2]

    # where to sample from in infra BEV
    grid_infra = transform_vehicle_grid_to_infra(
        grid_vehicle,
        R_inf_to_veh,
        t_inf_to_veh,
    )  # [H, W, 2]

    # convert infra metric coords to normalized coords for grid_sample
    sample_grid = metric_to_grid_sample_coords(
        grid_infra,
        xbound=xbound,
        ybound=ybound,
    )  # [H, W, 2]

    sample_grid = sample_grid.unsqueeze(0).repeat(B, 1, 1, 1)

    warped = F.grid_sample(
        infra_bev,
        sample_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )

    return warped

# warped = F.grid_sample(  use this to warp features from infra to veh
#     infra_bev,
#     grid,
#     mode="bilinear",
#     padding_mode="zeros",
#     align_corners=True
# )

###example use

# trans = vic_frame.transform(
#     from_coord="Infrastructure_lidar",
#     to_coord="Vehicle_lidar",
# )

# R, t = trans.get_rot_trans()

# infra_bev_aligned = warp_infra_bev_to_vehicle(
#     infra_bev,
#     R,
#     t,
#     xbound=[-50, 50, 0.5],
#     ybound=[-50, 50, 0.5],
# )