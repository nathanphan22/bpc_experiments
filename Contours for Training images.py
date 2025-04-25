import open3d as o3d
import open3d.visualization.rendering as rendering
import numpy as np
import os
from scipy.spatial.transform import Rotation as R 
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform
import json
import cv2
from PIL import Image

def retrieve_data(ipd_directory,scene,camera,image):
    camera_json_path = ipd_directory + "/ipd/train_pbr/" + scene + "/scene_camera_cam" + camera + ".json"
    object_json_path = ipd_directory + "/ipd/train_pbr/" + scene + "/scene_gt_cam" + camera + ".json"

    with open(camera_json_path, 'r') as file:
        camera_file = json.load(file)
    with open(object_json_path, 'r') as file:
        object_file = json.load(file)
    
    camera_data = camera_file[image]
    object_data = object_file[image]

    return camera_data, object_data

def set_scene(camera_data):
    K = np.array(camera_data['cam_K']).reshape((3, 3))
    R_w2c = np.array(camera_data['cam_R_w2c']).reshape((3, 3))
    t_w2c = np.array(camera_data['cam_t_w2c']).reshape((3, 1))
    s = camera_data['depth_scale']

    R_c2w = R_w2c.T
    t_c2w = -R_c2w @ t_w2c * s

    camera_position = t_c2w.flatten()
    image_point = np.array([1200, 1200, 1])
    K_inv = np.linalg.inv(K)
    camera_forward = K_inv.dot(image_point)

    return camera_position, camera_forward

def prepare_mesh(model_path, obj_t, obj_R):

    # Load the mesh
    mesh = o3d.io.read_triangle_mesh(model_path)
    mesh.compute_vertex_normals()

    # Place the object
    mesh.rotate(obj_R, center=(0, 0, 0))
    mesh.translate(obj_t)

    return mesh

def improve_contour(contour_list):
    
    points = []
    for contour in contour_list:
        for point in contour:
            points.append(point)

    if len(points) < 4 :
        return points
    
    cvx_hull = ConvexHull(points)
    hull_indices = np.array(cvx_hull.vertices, dtype=int)
    hull_points = []
    for index in hull_indices:
        hull_points.append(points[index])

    hull_lines = []
    for line in contour_list:
        add = False
        i=0
        while not(add) and i < len(line):
            if line[i] in hull_points :
                add = True
            i+=1
        if add :
            hull_lines.append(line)
    
    while len(hull_lines)>1 :
        extremities = []
        for line in hull_lines:
            extremities.append(line[0])
            extremities.append(line[-1])

        dists = squareform(pdist(extremities))
        triu_indices = np.triu_indices_from(dists, k=1)
        distances = dists[triu_indices]

        k = len(hull_lines)+1
        k_smallest = np.argpartition(distances, k)[:k]
        k_indices = [(triu_indices[0][i], triu_indices[1][i]) for i in k_smallest]

        index = 0
        while abs(k_indices[index][0]-k_indices[index][1])==1 and (k_indices[index][0]+k_indices[index][1]-1)%4 == 0:
            index+=1
        if k_indices[index][0] > k_indices[index][1] :
            k_indices[index][0], k_indices[index][1] = k_indices[index][1], k_indices[index][0]
        line_a, line_b = k_indices[index][0]//2, k_indices[index][1]//2
        side_a, side_b = k_indices[index][0]%2, k_indices[index][1]%2
        
        if side_a==0 and side_b==0 :
            hull_lines[line_b].reverse()
            combined = hull_lines[line_b] + hull_lines[line_a]
        elif side_a==1 and side_b==0 :
            combined = hull_lines[line_a] + hull_lines[line_b]
        elif side_a==0 and side_b==1 :
            combined = hull_lines[line_b] + hull_lines[line_a]
        else : # side_a==1 and side_b==1
            hull_lines[line_b].reverse()
            combined = hull_lines[line_a] + hull_lines[line_b]

        hull_lines.pop(line_b)
        hull_lines.pop(line_a)
        hull_lines.append(combined)

    return np.array(hull_lines[0])

def make_one_image_contours(ipd_directory,scene,camera,image):
    ratio = 8
    width, height = 2400//ratio, 2400//ratio
    object_contour_list = []
    object_id_list = []

    camera_data, object_data = retrieve_data(ipd_directory,scene,camera,image)
    cam_pos, cam_forward = set_scene(camera_data)
    cam_up = [0,-1,0]

    renderer = rendering.OffscreenRenderer(width, height)
    material = rendering.MaterialRecord()
    material.shader = "defaultLit"
    material.base_color = [0.0, 0.0, 0.0, 1.0]

    i=0
    for object_instance in object_data:

        model_path = ipd_directory + "/ipd/models/obj_" + str(object_instance['obj_id']).zfill(6) + ".ply"
        obj_t = np.array(object_instance['cam_t_m2c']).reshape((3,1))
        obj_R = np.array(object_instance['cam_R_m2c']).reshape((3,3))
        mesh = prepare_mesh(model_path, obj_t, obj_R)
        mesh_name = f"mesh_{i}"
        
        renderer.scene.clear_geometry()
        renderer.scene.set_background([1, 1, 1, 1.0])
        renderer.scene.add_geometry(mesh_name, mesh, material)
        renderer.setup_camera(40.0, cam_pos + 10*cam_forward, cam_pos, cam_up)

        img = renderer.render_to_image()
        img_path = "obj_{i:03d}.png"
        o3d.io.write_image(img_path, img)
        i+=1

        # Reload image
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        epsilon_ratio = 0.001
        simplified_contours = []
        for contour in contours:
            simplified_contour = []
            epsilon = epsilon_ratio * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            for point in approx:
                x, y = point[0]
                simplified_contour.append([x,y])
            simplified_contours.append(simplified_contour)

        final_contour = improve_contour(simplified_contours)
        object_contour_list.append(final_contour*ratio)
        object_id_list.append(str(object_instance['obj_id']).zfill(6))
        os.remove(img_path)
    
    return object_contour_list, object_id_list

def convert_to_native(obj):
    if isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(elem) for elem in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def main():
    # Input here the location of the ipd folder
    ipd_directory = "/home/rama/dataset"
    
    annot_dir = 'training_dataset_annotations'
    os.makedirs(annot_dir, exist_ok=True)
    image_dir = 'training_dataset_images'
    os.makedirs(image_dir, exist_ok=True)

    base_path = '/home/rama/dataset/ipd/train_pbr'
    for scn in range(50):
        scene = str(scn).zfill(6)
        for cam in range(1,4):
            camera = cam
            for im in range(100):
                image = str(im).zfill(6)

                image_object_contours_list, image_object_id_list = make_one_image_contours(ipd_directory,scene,str(camera),str(int(image)))

                data = {}
                data['image_path'] = 'training_dataset_images/' + scene + '_cam' + str(camera) + '_' + image + '.jpg'
                data['height'] = 2400
                data['width'] = 2400
                data['masks'] = []

                for i in range(len(image_object_contours_list)):
                    polygon = {}
                    polygon['label'] = 'obj_' + image_object_id_list[i]
                    polygon['points'] = image_object_contours_list[i]
                    data['masks'].append(polygon)
                
                # Save the dictionary to a .json file
                data_native = convert_to_native(data)
                output_path = 'training_dataset_annotations/' + scene + '_cam' + str(camera) + '_' + image + '.json'
                with open(output_path, 'w', encoding='utf-8') as json_file:
                    json.dump(data_native, json_file, ensure_ascii=False, indent=4)

                image_path = base_path + '/' + scene + '/rgb_cam' + str(camera) + '/' + image + '.jpg'
                img = Image.open(image_path)
                img.save(data['image_path'])

                print('_____Completed image ' + str(im) + '/100 of camera ' + str(cam) + '/3 of scene ' + str(scn) + '/50_____')
                print()




'''
This main function only does one image and does not register the .json file

def main():
    scene = "000000"
    camera = "1"
    image = "0"
    width, height = 1200, 1200
    output_dir = "solos"
    os.makedirs(output_dir, exist_ok=True)

    camera_data, object_data = retrieve_data(scene,camera,image)
    
    cam_pos, cam_forward = set_scene(camera_data)
    cam_up = [0,-1,0]

    renderer = rendering.OffscreenRenderer(width, height)
    material = rendering.MaterialRecord()
    material.shader = "defaultLit"
    material.base_color = [0.0, 0.0, 0.0, 1.0]

    i=0
    contour_img = np.zeros((width,height))
    for object_instance in object_data:

        model_path = "/home/rama/dataset/ipd/models/obj_" + str(object_instance['obj_id']).zfill(6) + ".ply"
        obj_t = np.array(object_instance['cam_t_m2c']).reshape((3,1))
        obj_R = np.array(object_instance['cam_R_m2c']).reshape((3,3))
        mesh = prepare_mesh(model_path, obj_t, obj_R)
        mesh_name = f"mesh_{i}"
        
        renderer.scene.clear_geometry()
        renderer.scene.set_background([1, 1, 1, 1.0])
        renderer.scene.add_geometry(mesh_name, mesh, material)
        renderer.setup_camera(40.0, cam_pos + 10*cam_forward, cam_pos, cam_up)

        img = renderer.render_to_image()
        img_path = os.path.join(output_dir, f"obj_{i:03d}.png")
        o3d.io.write_image(img_path, img)

        i+=1

        # Reload image
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        epsilon_ratio = 0.001
        simplified_contours = []
        for contour in contours:
            simplified_contour = []
            epsilon = epsilon_ratio * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            for point in approx:
                x, y = point[0]
                simplified_contour.append([x,y])
            simplified_contours.append(simplified_contour)

        final_contour = improve_contour(simplified_contours)

        # Draw **all** contours on a blank canvas
        if len(final_contour) >= 3:
            cv2.drawContours(contour_img, [final_contour], -1, (255, 255, 255), 1)
        os.remove(img_path)

    # Save the contour image
    final_path = 'solos/final.png'
    cv2.imwrite(final_path, contour_img)
'''


if __name__ == "__main__":
    main()