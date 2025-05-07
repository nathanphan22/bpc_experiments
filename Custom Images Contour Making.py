import open3d as o3d
import open3d.visualization.rendering as rendering
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from PIL import Image
import cv2

def render_with_rotations(model_path, output_dir, rot_mat_file, txt_directory):
    os.makedirs(output_dir, exist_ok=True)

    # Load the mesh
    original_mesh = o3d.io.read_triangle_mesh(model_path)
    original_mesh.compute_vertex_normals()

    # Sample n random rotations
    flattened_data = np.loadtxt(rot_mat_file)
    rotation_matrices = flattened_data.reshape(-1, 3, 3) 

    # Renderer setup
    width, height = 640, 480
    renderer = rendering.OffscreenRenderer(width, height)
    material = rendering.MaterialRecord()
    material.shader = "defaultLit"
    material.base_color = [0.0, 0.0, 0.0, 1.0]  # RGBA black
    material.base_metallic = 0.0
    material.base_roughness = 1.0

    # Lighting
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])  # RGBA
    renderer.scene.scene.set_sun_light([1, 1, 1], [1, 1, 1], 75000)
    renderer.scene.scene.enable_sun_light(True)

    for i, R_matrix in enumerate(rotation_matrices):
        with open(txt_directory + "/" + f"rot_{i:03d}" + ".txt", 'w') as file:
            mesh = o3d.geometry.TriangleMesh(original_mesh)
            mesh.rotate(R_matrix, center=(0, 0, 0))

            mesh_name = f"mesh_{i}"
            renderer.scene.clear_geometry()
            renderer.scene.add_geometry(mesh_name, mesh, material)

            bounds = mesh.get_axis_aligned_bounding_box()
            center = bounds.get_center()
            extent = bounds.get_extent().max()
            cam_pos = center + [0, 0, extent * 2.5]
            up = [0, 1, 0]

            # Calculate camera position
            bounds = mesh.get_axis_aligned_bounding_box()
            center = bounds.get_center()
            extent = bounds.get_extent().max()
            eye = center + np.array([0, 0, extent * 2.5])  # camera "eye" position
            up = np.array([0, 1, 0], dtype=np.float32)     # "up" direction

            # Setup camera properly
            renderer.setup_camera(60.0, center.astype(np.float32), eye.astype(np.float32), up)
            renderer.scene.camera.look_at(center, cam_pos, up)

            # Render and save
            img = renderer.render_to_image()
            img_path = os.path.join(output_dir, f"rot_{i:03d}.png")
            o3d.io.write_image(img_path, img)
            
            # Reload image
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)

            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw **all** contours on a blank canvas
            contour_img = np.zeros_like(img)
            cv2.drawContours(contour_img, contours, -1, (255, 255, 255), 1)

            # Save the contour image
            cv2.imwrite(img_path, contour_img)

            # Save all vertices of all contours
            epsilon_ratio = 0.01
            for contour in contours:
                epsilon = epsilon_ratio * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                for point in approx:
                    x, y = point[0]
                    file.write(f"{x}, {y}\n")
                file.write("\n")

def main():
    input_directory = "ipd/models"
    output_base_dir = "custom_images_contours"
    polygons_dir = os.path.join(output_base_dir, "polygons")
    os.makedirs(polygons_dir, exist_ok=True)

    for filename in os.listdir(input_directory):
        if filename.endswith(".ply"):
            model_name = filename.strip("obj_").strip(".ply")
            full_model_path = os.path.join(input_directory, filename)
            output_dir = os.path.join(output_base_dir, model_name)
            rot_mat_file = os.path.join("custom_images/rotation_matrices",model_name) + ".txt"
            txt_dir = os.path.join(polygons_dir, model_name)
            os.makedirs(txt_dir, exist_ok=True)

            print(f"Rendering model: {filename}")
            render_with_rotations(full_model_path, output_dir, rot_mat_file, txt_dir)

if __name__ == "__main__":
    main()
