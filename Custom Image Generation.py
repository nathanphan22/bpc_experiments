import open3d as o3d
import numpy as np
import os
from scipy.spatial.transform import Rotation as R

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

    # Set up Open3D visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    render_option = vis.get_render_option()
    render_option.background_color = np.array([0.5, 0.5, 0.5])

    cam = vis.get_view_control()
    cam.set_lookat([0, 0, 0])
    cam.set_front([0, 0, -1])
    cam.set_up([0, 1, 0])
    cam.set_zoom(1.2)

    with open(txt_filename, 'w') as file:
        for i, R_matrix in enumerate(rotation_matrices):
            # Apply rotation to a fresh copy of the original mesh
            mesh = o3d.geometry.TriangleMesh(original_mesh)
            mesh.rotate(R_matrix, center=(0, 0, 0))

            vis.clear_geometries()
            vis.add_geometry(mesh)

            vis.poll_events()
            vis.update_renderer()

            # Save image
            image_filename = os.path.join(output_dir, f"rot_{i:03d}.png")
            vis.capture_screen_image(image_filename)

            # Save rotation matrix
            flat_matrix = R_matrix.flatten()
            file.write(" ".join(f"{val:.6f}" for val in flat_matrix) + "\n")

    vis.destroy_window()

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
