import torch
from src.utils.veh_to_infra import transform_vehicle_grid_to_infra
import torch.functional as F

def make_grid_cells(xbound, ybound, device=None, dtype=torch.float32):
    """
    Creates BEV cell centers in metric coordinates.
    

    xbound = [xmin, xmax, xstep]
    ybound = [ymin, ymax, ystep]

    300,300,2 output tensor hsape that is the centre x,y 
    of each bev grid cell

    Returns:
        grid: [H, W, 2]
              grid[h, w] = [x, y]
    """

    xmin, xmax, xstep = xbound
    ymin, ymax, ystep = ybound

    xs = torch.arange(xmin + xstep / 2, xmax, xstep, device=device, dtype=dtype)
    ys = torch.arange(ymin + ystep / 2, ymax, ystep, device=device, dtype=dtype)

    yy, xx = torch.meshgrid(ys, xs)

    grid = torch.stack([xx, yy], dim=-1) #[300,300,2]

    return grid

def metric_to_grid_sample_coords(
    grid_metric,
    xbound,
    ybound,
):
    """
    Convert metric BEV coordinates into grid_sample coordinates [-1, 1].

    Args:
        grid_metric: [H, W, 2]

    Returns:
        grid: [H, W, 2]

    grid[..., 0] -> horizontal/W coordinate
    grid[..., 1] -> vertical/H coordinate
    """

    xmin, xmax, xstep = xbound
    ymin, ymax, ystep = ybound

    x = grid_metric[..., 0]
    y = grid_metric[..., 1]

    # Number of cells
    W = int(round((xmax - xmin) / xstep))
    H = int(round((ymax - ymin) / ystep))

    # Metric position -> pixel index corresponding to cell centers
    ix = (x - (xmin + xstep / 2)) / xstep
    iy = (y - (ymin + ystep / 2)) / ystep

    # Because align_corners=True:
    # pixel 0     -> -1
    # pixel W - 1 -> +1

    gx = 2.0 * ix / (W - 1) - 1.0
    gy = 2.0 * iy / (H - 1) - 1.0

    return torch.stack(
        [gx, gy],
        dim=-1,
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


def warp_infra_bev_to_veh(veh2infra,Hv,Wv,dx,x_min,y_min,f_i):
    Hi,Wi,fi = f_i.shape

    f_veh = torch.zeros(Hv,Wv,fi)

    for x_v_d,y_v_d in zip(range(Wv),range(Hv)):
         
         x_v_r = x_min + dx * x_v_d
         y_v_r = y_min + dx * y_v_d

         x_i_r, y_i_r = [x_v_r , y_v_r]*veh2infra

         x_i_d = (x_i_r - x_min)/dx
         y_i_d = (y_i_r - y_min)/dx

         f_veh[x_v_d,y_v_d ] = f_i[x_i_d,y_i_d]

    return

def warp_infra_bev_to_veh_2(veh2infra,Hv,Wv,dx,x_min,y_min,f_i):
    for row in range(Hv):
        for col in range(Wv):
            # Vehicle cell center in metres
            x_v = x_min_v + (col + 0.5) * dx_v
            y_v = y_min_v + (row + 0.5) * dy_v

            # Transform vehicle position into infrastructure coordinates
            # Apply rotation AND translation here.
            x_i, y_i = transform_vehicle_to_infra(x_v, y_v)

            # Infrastructure cell indices (can be fractional)
            col_i = (x_i - x_min_i) / dx_i - 0.5
            row_i = (y_i - y_min_i) / dy_i - 0.5

            # Interpolate infrastructure features at (row_i, col_i)
            # and store them at f_veh[row, col].