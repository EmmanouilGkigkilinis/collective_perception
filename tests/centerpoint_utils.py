import torch

from typing import List

def encode_gt_batches(targets_metric:List, 
                    BEV_RANGE_HORZ: int,
                    BEV_RANGE_VERT: int,
                    BEV_RANGE_HORZ_MIN:int,
                    BEV_RANGE_VERT_MIN:int, 
                    metres_per_cell:float,
                    NUM_CLASSES:int  ):
    """
    GT ENCODING SCHEME FOR CENTERPOINT DETECTION

    encode heatmap head: return gaussian targets on the radius around class. In the grid you get zeros everywhere apart from obj centers where maximum is assigned and gaussian decay around it at radius = r.

    encode offset head: return zeros everywhere apart from centre cx,cy 

    encode height,


    encode dim: logarithm

    encode rot: sine 

    INputs
        targets_metric: targets in metric coords
            list of gt dictionaries of loc,dim,rot
        BEV_RANGE_HORZ: horizontal bev_range max
        BEV_RANGE_VERRT; vert bev_range max 

    RETURN:
        target: heatmap targets 
    """

    heatmap = targets_metric

    #make bbox
    boxes_batch = [(batch["3d_location"]["x"],   
                batch["3d_location"]["y"])
                for batch  in targets_metric ]  
    

    classes_batch = [batch["type"] for batch in targets_metric]

    pred_shape = (len(targets_metric) , NUM_CLASSES , BEV_RANGE_HORZ , BEV_RANGE_VERT )

    x_min = BEV_RANGE_HORZ_MIN
    y_min = BEV_RANGE_VERT_MIN
    
    ###==================encode targets =======================
    target_heatmap = make_heatmap_targets(boxes_batch=targets_metric["3d_location"],
                                  classes_batch=classes_batch,
                                  pred_shape=pred_shape,
                                  x_min=x_min,
                                  y_min=y_min,
                                  metres_per_cell=metres_per_cell)



    target_offsets,target_heights,target_sizes,targets_rots = make_offset_targets(bboxes_batch=targets_metric,
                                                                                  
                                                    heights_batch = 
                                                        [batch["3d_location"]["z"] for batch in targets_metric],
                                                        
                                                    rot_batch = [batch["rotation"] for batch in targets_metric],

                                                    size_batch = [batch["3d_dimensions"] for batch in targets_metric],

                                                    x_min=x_min,
                                                    y_min=y_min, 
                                                    metres_per_cell=metres_per_cell
                
                                                    )



    return target_heatmap , target_offsets, target_heights, target_sizes, targets_rots


# def make_height_targets(heights_batch , H ,W ):
#     B = len(heights_batch)
#     targets = torch.zeros(B, 1 , H , W)
#     for b in range(B):
#         for heights in heights_batch(b):
            


def make_offset_targets(bboxes_batch, 
                        heights_batch,
                        rots_batch,
                        size_batch , 
                        x_min , 
                        y_min , 
                        meters_per_cell,
                        H,
                        W,):
    """
    make targets for offset head of centerpoint

    return the difference of actual prediction in cell vs the cell of ground truth , which can be different due to discretization error.

    """

    B = len(bboxes_batch) 
    target_offsets = torch.zeros(B ,2 , H , W)
    targets_heights= torch.zeros(B, 1, H ,W )
    targets_size = torch.zeros(B , 3 , H ,W )
    targets_rots = torch.zeros (B, 2 , H ,W )
    for b in range(B):
        for bbox,height,rots,size in zip(bboxes_batch[b],heights_batch[b],rots_batch[b],size_batch[b]):
            x,y=bbox[0],bbox[1]

            ux = (x - x_min) / meters_per_cell
            uy = (y - y_min) / meters_per_cell

            cx = int(torch.floor(ux))
            cy = int(torch.floor(uy))

            target_offsets[b , : , cx,cy] =  torch.stack([ux - cx, uy - cy])
            targets_heights[b, 0, cy, cx] = height
            targets_size[b, :, cy, cx] = torch.log([i.values() for i in size.keys()])
            targets_rots[b, :, cy, cx] = torch.stack([
                torch.sin(rots),
                torch.cos(rots),
            ])

            if not (0 <= cx < W and 0 <= cy < H):
                continue

    
    return target_offsets,targets_heights,targets_size,targets_rots



def make_heatmap_targets(boxes_batch, 
                         classes_batch, 
                         pred_shape,
                         x_min, 
                         y_min, 
                         metres_per_cell):
    """
    Here we have a gaussian normal centered on the keypoint. We set the target at keypoint equal to the gaussian
    at the centre (maximal =1 ) and exp decay around the keyframe to get 2radius+1 rectanggle.
    The reason is purely better training convergence instead of abrupt change from 0 to 1 in non-keypoint,keypoint 
    BEV cells. 

    the patch is a 4d tensor centered around keypoint, with b batch |C| classes and H,W metric space. the class
    constitues an ascending discrete metric predefined. All targets are assigned ranges in the [0,1] range

    Then perform training of the heatmap head with the focal loss.

    x_min: min x-coord BEV
    y_min min y-coord BEV
    metres_per_cell: metric length of cell in BEV 
    
    """

    B, num_classes, H, W = pred_shape

    #construct target tensor , this is for heatmap head of centerpoint
    target = torch.zeros(pred_shape, device=boxes_batch[0].device)

    radius = 2
    yy, xx = torch.meshgrid(
        torch.arange(-radius, radius + 1, device=target.device),
        torch.arange(-radius, radius + 1, device=target.device),
        indexing="ij",
    )

    #gaussian around keybpoint
    gaussian = torch.exp(-(xx**2 + yy**2) / (2 * 1.0**2))

    for b in range(B):
        #iterate in given ground truth database 
        for box,  class_name in zip(boxes_batch[b], classes_batch[b]):
            # box centre is in vehicle LiDAR metres
            x, y = box[0], box[1]  
            #box center in BEV cell discrete coordinates
            cx = int(torch.floor((x - x_min) / metres_per_cell))
            cy = int(torch.floor((y - y_min) / metres_per_cell))
            class_name = int(class_name)

            if not (0 <= cx < W and 0 <= cy < H):
                continue

            #take the coordinates which lie di rectly inside keypoint of heatmap only
            x0, x1 = max(0, cx - radius), min(W, cx + radius + 1) #max and min in case the boxes lie outside of the 2d range
            y0, y1 = max(0, cy - radius), min(H, cy + radius + 1)

            #target is initially zeros
            patch = target[b, class_name, y0:y1, x0:x1]


            g = gaussian[
                y0 - (cy - radius): y1 - (cy - radius),
                x0 - (cx - radius): x1 - (cx - radius),
            ]

            target[b, class_name, y0:y1, x0:x1] = torch.maximum(patch, g)

    return target


# if __name__ == "__main__":
    
