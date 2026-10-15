import os
import sys
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from matplotlib.transforms import Affine2D
from src.utils.centerpoint_utils import get_decoded_bbox_from_pred


def viz_boxes(predictions ,
              targets,
              pc,
              score_threshold,
              metres_per_cell,x_min,y_min,save_path=None,desc=None)->None:
    """
    visualize 2d boxes from prediction and targert
    """
    B = predictions["heatmap"].shape[0]
    results = get_decoded_bbox_from_pred(predictions,score_threshold=score_threshold,metres_per_cell=metres_per_cell,x_min=x_min,y_min=y_min)
    
    
    print("pc shape:", pc.shape)

    # pc=pc.detach().cpu().tolist()
    # pc=np.asarray(pc)[:,:3]
    # print(pc)
    # sys.exit()

    if save_path is None:
                print("skipping saving plt")

    if save_path is not None and not os.path.exists(save_path + desc+".png"):
        fig,ax=plt.subplots()

        for batch in range(B):
            print("batcvh is {}".format(batch))
            pc = pc[batch, :, :3].detach().cpu().numpy()    
            boxes = results[batch]["boxes"].detach().cpu().tolist()
            for box in boxes:  #multiple box
                x_p, y_p, z_p , l_p , w_p, h_p, yaw_p = box
                
                rectangle=Rectangle(  #prediction
                        (x_p - w_p / 2, y_p - l_p / 2),
                        w_p,
                        l_p,
                        fill=False,
                        edgecolor="red",
                        linewidth=2,
                        label="predict",
                    )
                rectangle.set_transform(
                    Affine2D().rotate_around(x_p, y_p, yaw_p) + ax.transData
                )
                ax.add_patch(rectangle)
            
            for gt_box in targets["gt_boxes"][batch]:
                x, y =  gt_box["bbox"]#targets
                l,w= gt_box["3d_dimensions"]["l"],gt_box["3d_dimensions"]["w"]
                yaw = gt_box["rotation"]
                # print(x,y,l,w)
                # sys.exit()
                rectangle = Rectangle(  #target
                            (x - w/2 , y - l/ 2),
                            w,
                            l,
                            fill=False,
                            edgecolor="green",
                            linewidth=2,
                            label="target",
                        )
                rectangle.set_transform(
                    Affine2D().rotate_around(x, y, yaw) + ax.transData
                )
                ax.add_patch(rectangle)
                ax.set_aspect("equal", adjustable="box")
            # size_target=targets[batch]["size"]
            # width=size_target["w"]
            # length=size_target["l"]

            # x_p,y_p = boxes #prediction

            ax.set_title("object predictions vs targets")
            # ax.scatter([x,y],
            #            [x_p,y_p]) #plot predictions

            print("XYZ min:", pc[:, :3].min(axis=0))
            print("XYZ max:", pc[:, :3].max(axis=0))
            print("All finite:", np.isfinite(pc[:, :3]).all())
            
            ax.scatter(pc[:,0] , pc[:,1], c="blue",cmap="viridis",s=0.5,alpha=0.5, zorder=1  )

            ax.set(xlabel="x", ylabel="y")
            ax.grid(True)
            
            # plt.show()
            # elif save_path is None and not os.path.exists(save_path , desc+".png"):
            print("saving plt")
            plt.savefig(os.path.join(save_path + desc+".png"),dpi=300,bbox_inches="tight")
            plt.savefig(os.path.join(save_path + desc+".svg"),dpi=300,bbox_inches="tight")
            plt.close(fig)

        
