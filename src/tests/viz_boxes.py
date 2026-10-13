from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from src.utils.centerpoint_utils import get_decoded_bbox_from_pred


def viz_boxes(predictions ,
              targets,
              pc,
              score_threshold,
              metres_per_cell,x_min,y_min)->None:
    """
    visualize 2d boxes from prediction and targert
    """
    B = predictions["heatmap"].shape[0]
    results = get_decoded_bbox_from_pred(predictions,score_threshold=score_threshold,metres_per_cell=metres_per_cell,x_min=x_min,y_min=y_min)

    for batch in range(B):
        
        fig,ax=plt.subplots()

        x_p, y_p, z_p , l_p , w_p, h_p, yaw_p=results[batch]["boxes"]

        x, y= targets[batch]["bbox"] #targets
        z , l , w, h, yaw = targets[batch]["height"],targets[batch]["3d_dimensions"] , targets[batch]["rotation"]

        # size_target=targets[batch]["size"]
        # width=size_target["w"]
        # length=size_target["l"]

        # x_p,y_p = boxes #prediction

        ax.set_title("object predictions vs targets")
        ax.scatter([x,y,z],
                   [x_p,y_p]) #plot predictions

        rectangle = Rectangle(  #target
                    (x - w_p / 2, y_p - l_p / 2),
                    w_p,
                    l_p,
                    fill=False,
                    edgecolor="green",
                    linewidth=2,
                    label="target",
                )
        
        rectangle=Rectangle(  #prediction
                    (x_p - w / 2, y_p - l / 2),
                    w,
                    l,
                    fill=False,
                    edgecolor="red",
                    linewidth=2,
                    label="predict",
                )
        
        ax.scatter(pc[:,0] , pc[:,1],c=pc[:,2] , cmap="viridis",s=0.5,alpha=0.5, zorder=1  )


        ax.add_patch(rectangle)
        ax.set(xlabel="x", ylabel="y")
        ax.grid(True)

        plt.show()
