"""
reconstruction.py: Collects images from the inspection camera and performs 
structure from motion to reconstruct the key object. (for now, a rubiks cube)

Detailed Explanation:
    For now, the desired behavior is as follows:
        - Cache images as the jackal drives around object
        - Pass camera pose from update_map (if needed, to increase robustness if frames are sparse) nav_camera.data.pos_w
        - Run structure from motion algorithm, don't reinvent the wheel, use an existing library
        - Display in open3d to verify the reconstruction (for debug/development only)

Important notes:
    - This needs to run offline on the Jetson Orin, not on the Jackal onboard computer
    - The inspection camera is mounted on the side of the jackal, facing sideways

Author: Noah Chapman
Date: 2026-02-04
"""

import numpy as np
try:
    import open3d as o3d
except ImportError:
    print("\n[ERROR] Open3D is not installed. Please install it to use the reconstruction feature:")
    print("        pip install open3d\n")
    o3d = None
import os
import shutil
import json

class ImageCollector:
    """
    Collects and saves RGB-D data, camera poses, and intrinsics for reconstruction.
    """
    def __init__(self, data_dir="output/reconstruction_data"):
        self.data_dir = data_dir
        
        # Clean and recreate directory
        if os.path.exists(self.data_dir):
            shutil.rmtree(self.data_dir)
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.frame_count = 0
        print(f"[ImageCollector] Saving data to: {self.data_dir}")

    def save_data(self, rgb, depth, pose, intrinsic, step):
        """
        Saves a single frame's worth of data.
        
        Args:
            rgb (np.ndarray): RGB image (H, W, 3) in range [0, 1] or [0, 255].
            depth (np.ndarray): Depth image (H, W) in meters.
            pose (np.ndarray): 4x4 Camera-to-World pose matrix.
            intrinsic (np.ndarray): 3x3 Intrinsic matrix.
            step (int): Simulation step number (for naming).
        """
        # Ensure RGB is uint8 [0, 255]
        if rgb.dtype != np.uint8:
            if rgb.max() <= 1.0:
                rgb = (rgb * 255).astype(np.uint8)
            else:
                rgb = rgb.astype(np.uint8)
        
        # Open3D expects depth in uint16 (millimeters) usually for PNG, 
        # or float for .tiff. We will save as .npy for precision or .png if scaled.
        # Let's save depth as .npy to allow perfect reconstruction without scaling artifacts.
        
        # Save Images
        saved_image = False
        if o3d is not None:
            try:
                 o3d_rgb = o3d.geometry.Image(rgb)
                 o3d.io.write_image(f"{self.data_dir}/color_{step:05d}.png", o3d_rgb)
                 saved_image = True
            except Exception as e:
                print(f"[ImageCollector] Failed to save image with Open3D: {e}")
        
        if not saved_image:
            try:
                from PIL import Image
                im = Image.fromarray(rgb)
                im.save(f"{self.data_dir}/color_{step:05d}.png")
            except ImportError:
                 print("[ImageCollector] PIL not found, cannot save RGB image.")
            except Exception as e:
                 print(f"[ImageCollector] Failed to save image with PIL: {e}")
            
        # Save raw data for reconstruction
        np.save(f"{self.data_dir}/depth_{step:05d}.npy", depth)
        np.save(f"{self.data_dir}/pose_{step:05d}.npy", pose)
        np.save(f"{self.data_dir}/intrinsic_{step:05d}.npy", intrinsic)
        
        self.frame_count += 1

class ObjectReconstruction:
    """
    Reconstructs 3D geometry from collected RGB-D and Pose data using Open3D TSDF Integration.
    """
    def __init__(self, data_dir="output/reconstruction_data"):
        self.data_dir = data_dir
        self.voxel_size = 0.01  # 1cm resolution
        self.trunc = 0.04       # Truncation distance (4cm)

    def reconstruct(self, visualize=True):
        """
        Reads data from data_dir and runs TSDF integration.
        """
        if not os.path.exists(self.data_dir):
            print(f"Data directory {self.data_dir} does not exist.")
            return

        print(f"[ObjectReconstruction] Starting reconstruction from {self.data_dir}...")
        
        files = sorted(os.listdir(self.data_dir))
        depth_files = [f for f in files if f.startswith("depth_") and f.endswith(".npy")]
        
        if not depth_files:
            print("No depth files found.")
            return

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self.voxel_size,
            sdf_trunc=self.trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        for f in depth_files:
            # Extract step number
            step_str = f.split('_')[1].split('.')[0]
            step = int(step_str)
            
            # Load paths
            color_path = f"{self.data_dir}/color_{step:05d}.png"
            pose_path = f"{self.data_dir}/pose_{step:05d}.npy"
            intrinsic_path = f"{self.data_dir}/intrinsic_{step:05d}.npy"
            
            if not os.path.exists(color_path) or not os.path.exists(pose_path):
                continue
                
            # Load Data
            color = o3d.io.read_image(color_path)
            depth_np = np.load(f"{self.data_dir}/depth_{step:05d}.npy").astype(np.float32)
            pose = np.load(pose_path)
            intrinsic_np = np.load(intrinsic_path)
            
            # Create Open3D objects
            # Convert depth to a form Open3D likes (Image)
            # Make sure it's contiguous
            depth_o3d = o3d.geometry.Image(depth_np.copy()) 
            
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color, depth_o3d, 
                depth_scale=1.0, 
                depth_trunc=3.0, 
                convert_rgb_to_intensity=False
            )
            
            # Intrinsic object
            h, w = depth_np.shape
            # Assuming intrinsic is 3x3
            fx, fy = intrinsic_np[0, 0], intrinsic_np[1, 1]
            cx, cy = intrinsic_np[0, 2], intrinsic_np[1, 2]
            intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
            
            # Integrate
            # Open3D integrate takes extrinsic (World-to-Camera)
            volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))
            
        print("Integration complete. Extracting mesh...")
        mesh = volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        
        if visualize:
            o3d.visualization.draw_geometries([mesh], window_name="Reconstructed Mesh")
            
        return mesh

if __name__ == "__main__":
    print("Running reconstruction from saved data...")
    recon = ObjectReconstruction()
    mesh = recon.reconstruct(visualize=True)

