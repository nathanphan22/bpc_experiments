import open3d as o3d
import numpy as np
import copy
import json

import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# scene in an integer between 0 and 49
# camera is an integer between 1 and 3
# photo is an integer between 0 and 99
#index is an integer : its range depends on the objects seen by the camera
def visualize_rotation(scene,camera,photo,index):
    T_mat, R_mat, id = get_pose_data(scene,camera,photo,index)
    show_object(R_mat,id)
    return()

def get_pose_data(scene,camera,photo,index):
    # Open correct json file
    path = "ipd/train_pbr/" + str(scene).zfill(6) + "/scene_gt_cam" + str(camera) + ".json"
    with open(path, 'r') as file:
        data = json.load(file)

    # Extract data
    T_mat = data[str(photo)][index]["cam_t_m2c"]
    R_mat_line = data[str(photo)][index]["cam_R_m2c"]
    R_mat = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            R_mat[i][j] = R_mat_line[3*i+j]
    id = data[str(photo)][index]["obj_id"]

    return(T_mat, R_mat, id)


def show_object(R_mat,id):
    # Load 3D object
    object_path = "ipd/models/obj_" + str(id).zfill(6) + ".ply"
    mesh = o3d.io.read_triangle_mesh(object_path)
    mesh.compute_vertex_normals()

    # Create rotated mesh
    rotated_mesh = copy.deepcopy(mesh)
    rotated_mesh.rotate(R_mat, center=rotated_mesh.get_center())

    # Translate objects to avoid overlaping
    bbox = mesh.get_axis_aligned_bounding_box()
    size = bbox.get_extent()
    offset_x = size[0] * 2
    mesh.translate((-offset_x, 0, 0))
    rotated_mesh.translate((offset_x, 0, 0))

    # Assign colors to the meshes
    mesh.paint_uniform_color([0.2, 0.6, 1.0])         # Original: blue-ish
    rotated_mesh.paint_uniform_color([1.0, 0.3, 0.3]) # Rotated: red-ish

    # Add reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50, origin=[0, 0, 0])

    # Prepare the plot
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh)
    vis.add_geometry(rotated_mesh)
    vis.add_geometry(coord_frame)

    # Set camera to look along Z-axis
    ctr = vis.get_view_control()
    ctr.set_lookat([offset_x, 0, 0])       # Camera looks at the rotated piece
    ctr.set_front([0, 0, -1])        # Camera faces along +Z
    ctr.set_up([0, -1, 0])          # Y down in image space

    # Show
    vis.run()
    vis.destroy_window()

    return()

# Get the coordinates of the click
def onclick(event,scene,camera,photo):

    x, y = event.xdata, event.ydata
    index = search_nearest_object(scene,camera,photo,x,y)
    visualize_rotation(scene,camera,photo,index)

def search_nearest_object(scene,camera,photo,x,y):
    # Load json file
    path = "ipd/train_pbr/" + str(scene).zfill(6) + "/scene_gt_info_cam" + str(camera) + ".json"
    with open(path, 'r') as file:
        data = json.load(file)

    # Loop over the objects
    closest_id = -1
    closest_distance = np.inf
    i=0
    for object in data[str(photo)]:
        if object["visib_fract"] > 0 :
            pos_x = object["bbox_obj"][0] + object["bbox_obj"][2]/2
            pos_y = object["bbox_obj"][1] + object["bbox_obj"][3]/2
            dist = np.sqrt( (pos_x-x)**2 + (pos_y-y)**2 )
            if dist < closest_distance :
                closest_id = i
                closest_distance = dist
        i+=1

    # Indicates the 2D coordinates of the closest object
    print("x : " + str(data[str(photo)][closest_id]["bbox_obj"][0] + data[str(photo)][closest_id]["bbox_obj"][2]/2))
    print("y : " + str(data[str(photo)][closest_id]["bbox_obj"][1] + data[str(photo)][closest_id]["bbox_obj"][3]/2))
    print()

    return closest_id

def main():

    # Ask for an image to open
    root = tk.Tk()
    root.withdraw()
    image_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image files","*.jpg;*.jpeg;*.png;*.bmp")])

    try :
        # Find the localisation of the data
        split_path = ("ipd" + image_path.split("ipd")[1]).split("/")
        scene = int(split_path[2])
        camera = int(split_path[3][-1])
        photo = int(split_path[4].split(".")[0])
    except:
        print("Something went wrong, couldn't find the image data")
    else:
        # Plot the selected image
        img = mpimg.imread(image_path)
        img_height, img_width = img.shape[:2]
        fig, ax = plt.subplots()
        ax.imshow(img, extent=[0, img_width, img_height, 0])
        cid = fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event,scene,camera,photo))
        plt.show()


if __name__ == "__main__":
    main()