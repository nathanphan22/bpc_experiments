import open3d as o3d
import open3d.visualization.rendering as rendering
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from PIL import Image

def render_with_rotations(model_path, n, output_dir, txt_dir, txt_filename):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(txt_dir, exist_ok=True)

    # Load the mesh
    original_mesh = o3d.io.read_triangle_mesh(model_path)
    original_mesh.compute_vertex_normals()

    # Sample n random rotations
    random_rotations = R.random(n - 1).as_matrix()
    identity = np.eye(3).reshape(1, 3, 3)
    rotation_matrices = np.concatenate((identity, random_rotations), axis=0)

    # Renderer setup
    width, height = 640, 480
    renderer = rendering.OffscreenRenderer(width, height)
    material = rendering.MaterialRecord()
    material.shader = "defaultLit"

    # Lighting
    renderer.scene.set_background([0.5, 0.5, 0.5, 1.0])  # RGBA
    renderer.scene.scene.set_sun_light([1, 1, 1], [1, 1, 1], 75000)
    renderer.scene.scene.enable_sun_light(True)

    with open(txt_filename, 'w') as file:
        for i, R_matrix in enumerate(rotation_matrices):
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

            # Save rotation matrix
            flat_matrix = R_matrix.flatten()
            file.write(" ".join(f"{val:.6f}" for val in flat_matrix) + "\n")

def main():
    n = 100  # Number of random rotations per model
    input_directory = "ipd/models"
    output_base_dir = "custom_images"
    rotation_matrix_dir = os.path.join(output_base_dir, "rotation_matrices")
    os.makedirs(rotation_matrix_dir, exist_ok=True)

    for filename in os.listdir(input_directory):
        if filename.endswith(".ply"):
            model_name = filename.strip("obj_").strip(".ply")
            full_model_path = os.path.join(input_directory, filename)
            output_dir = os.path.join(output_base_dir, model_name)
            txt_file = os.path.join(rotation_matrix_dir, f"{model_name}.txt")

            print(f"Rendering model: {filename}")
            render_with_rotations(full_model_path, n, output_dir, rotation_matrix_dir, txt_file)

if __name__ == "__main__":
    main()
