
import os
import json
import torch
import numpy as np
import shutil
from PIL import Image
from scipy.spatial.transform import Rotation

class ReconstructionDataCollector:
    """
    Collects Depth and Semantic Mask data for offline point cloud generation.
    Saves:
        - Depth maps as .npy (float32, meters)
        - Semantic masks as .png (uint8)
        - Camera poses and intrinsics in transforms.json
    """
    def __init__(self, cfg, device="cpu"):
        self.cfg = cfg
        self.device = device
        self.output_path = getattr(cfg, "data_recording_path", "data/recorded_point_clouds")
        self.save_interval = getattr(cfg, "save_interval", 1)
        
        self.depth_path = os.path.join(self.output_path, "depth")
        self.masks_path = os.path.join(self.output_path, "masks")
        self.rgb_path = os.path.join(self.output_path, "rgb")
        self.rgb_cameras_path = os.path.join(self.output_path, "rgb_cameras")
        
        # Cleanup and Create Directories
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)
        os.makedirs(self.depth_path, exist_ok=True)
        os.makedirs(self.masks_path, exist_ok=True)
        os.makedirs(self.rgb_path, exist_ok=True)
        os.makedirs(self.rgb_cameras_path, exist_ok=True)
        
        print(f"[ReconstructionDataCollector] Initialized. Saving Depth/Masks/RGB to: {self.output_path}")
        
        # Buffer for transforms.json
        self.transforms_data = {
            "camera_angle_x": 0.0,
            "fl_x": 0.0, 
            "fl_y": 0.0,
            "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0,
            "cx": 0.0, "cy": 0.0,
            "w": 0, "h": 0,
            "frames": []
        }
        
        self.intrinsics_set = False
        self.frame_idx = 0
        
    def collect(self, camera, step_idx, semantic_mask=None, nav_camera=None):
        """
        Collects data from the camera.
        
        Args:
            camera: The TiledCamera object.
            step_idx: Current simulation step.
            semantic_mask: Optional boolean tensor (N, H, W, 1) to filter points.
        """
        if step_idx % self.save_interval != 0:
            return
            
        if getattr(self, "is_first_frame", True):
            self.is_first_frame = False
            return

        env_id = 0
        if "distance_to_image_plane" not in camera.data.output:
             return

        # Get Data
        depth_tensor = camera.data.output["distance_to_image_plane"][env_id] # (H, W) or (H, W, 1)
        intrinsic_matrix = camera.data.intrinsic_matrices[env_id]
        cam_pos_w = camera.data.pos_w[env_id]
        # cam_quat_w = camera.data.quat_w_world[env_id]

        
        # --- Handle Intrinsics (Once) ---
        if not self.intrinsics_set:
            h, w = depth_tensor.shape[:2]
            self._set_intrinsics(intrinsic_matrix, w, h)
            self.intrinsics_set = True

        # Convert Depth
        depth_np = depth_tensor.cpu().numpy()
        if depth_np.ndim == 3:
            depth_np = depth_np.squeeze(-1) # Ensure (H, W)
            
        mask_np = None
        if semantic_mask is not None:
            mask_tensor = semantic_mask[env_id] # (H, W, 1)
            mask_np = mask_tensor.cpu().numpy().squeeze(-1) # (H, W) boolean
            
        # Get Transform
        mat = self._get_matrix(cam_pos_w.cpu().numpy(), cam_quat_w.cpu().numpy())
        
        # Extract per-frame intrinsics for zooming
        K_np_frame = intrinsic_matrix.cpu().numpy()
        fl_x_frame = float(K_np_frame[0, 0])
        fl_y_frame = float(K_np_frame[1, 1])
        cx_frame = float(K_np_frame[0, 2])
        cy_frame = float(K_np_frame[1, 2])
        
        # Buffer the data
        rgb_np = None
        rgb_cameras_np = None
        if "rgb" in camera.data.output:
            rgb_tensor = camera.data.output["rgb"][env_id] # (H, W, 4) or (H, W, 3)
            rgb_np = rgb_tensor.cpu().numpy()
            if rgb_np.shape[-1] == 4:
                rgb_np = rgb_np[..., :3] # Remove Alpha
                
            rgb_cameras_np = rgb_np.copy()
            # If nav_camera is provided, stitch side-by-side
            if nav_camera is not None and "rgb" in nav_camera.data.output:
                nav_rgb_tensor = nav_camera.data.output["rgb"][env_id]
                nav_rgb_np = nav_rgb_tensor.cpu().numpy()
                if nav_rgb_np.shape[-1] == 4:
                    nav_rgb_np = nav_rgb_np[..., :3]
                
                # Resize if heights don't match
                if rgb_np.shape[0] != nav_rgb_np.shape[0]:
                    import cv2
                    h = rgb_np.shape[0]
                    w = int(nav_rgb_np.shape[1] * (h / nav_rgb_np.shape[0]))
                    nav_rgb_np = cv2.resize(nav_rgb_np, (w, h))

                # Concatenate horizontally
                rgb_cameras_np = np.concatenate((nav_rgb_np, rgb_np), axis=1)

        if not hasattr(self, 'frame_buffer'):
            self.frame_buffer = []

        self.frame_buffer.append({
            "depth": depth_np,
            "mask": mask_np,
            "rgb": rgb_np,
            "rgb_cameras": rgb_cameras_np,
            "transform_matrix": mat.tolist(),
            "fl_x": fl_x_frame,
            "fl_y": fl_y_frame,
            "cx": cx_frame,
            "cy": cy_frame
        })
        self.frame_idx += 1
        
    def _set_intrinsics(self, K, width, height):
        K_np = K.cpu().numpy()
        fl_x = float(K_np[0, 0])
        fl_y = float(K_np[1, 1])
        cx = float(K_np[0, 2])
        cy = float(K_np[1, 2])
        
        import math
        self.transforms_data["fl_x"] = fl_x
        self.transforms_data["fl_y"] = fl_y
        self.transforms_data["cx"] = cx
        self.transforms_data["cy"] = cy
        self.transforms_data["w"] = int(width)
        self.transforms_data["h"] = int(height)
        if fl_x > 0:
            self.transforms_data["camera_angle_x"] = 2 * math.atan(width / (2 * fl_x))
        else:
            self.transforms_data["camera_angle_x"] = 1.0 # Default fallback
            
    def _get_matrix(self, pos, quat):
        # Isaac Lab Quat is (w, x, y, z)
        # Scipy Rotation expects (x, y, z, w)
        rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        
        # Standard OpenGL/NVIDIA convention often used in NeRF/Gaussian Splatting
        # X Right, Y Up, Z Back (Camera looks down -Z)
        # Isaac Sim Camera: X Right, Y Down, Z Forward (or similar depending on USD)
        # Usually Isaac Sim cameras are often Y-up or Z-up depending on stage. 
        # But `annotators` outputs are often consistent.
        
        # Let's stick to the matrix conversion logic from DataCollector 
        # which converts ROS (X Forward) to OpenGL (-Z Forward)
        
        # ROS Basis: [Forward, Left, Up] = [X, Y, Z]
        # OpenGL Basis: [Right, Up, Back] = [X, Y, Z] (where Forward = -Z)
        
        # Correction Matrix T (post-multiply):
        # [[ 0,  0, -1],
        #  [-1,  0,  0],
        #  [ 0,  1,  0]]
        
        T = np.array([
            [ 0,  0, -1],
            [-1,  0,  0],
            [ 0,  1,  0]
        ])
        
        mat = np.eye(4)
        # Apply T to the rotation part
        mat[:3, :3] = rot.as_matrix() @ T
        mat[:3, 3] = pos
        return mat

    def flush_if_best(self, faces_discovered, max_distance):
        """Flushes the buffer to disk if this is the best episode so far."""
        if not hasattr(self, 'best_faces_discovered'):
            self.best_faces_discovered = 0
            
        if faces_discovered > self.best_faces_discovered:
            self.best_faces_discovered = faces_discovered
            print(f"[ReconstructionDataCollector] Flushing new best episode to disk. Faces: {faces_discovered}")
            
            # Clear old directories if they exist safely
            if os.path.exists(self.depth_path):
                shutil.rmtree(self.depth_path)
            if os.path.exists(self.masks_path):
                shutil.rmtree(self.masks_path)
            if os.path.exists(self.rgb_path):
                shutil.rmtree(self.rgb_path)
            if hasattr(self, 'rgb_cameras_path') and os.path.exists(self.rgb_cameras_path):
                shutil.rmtree(self.rgb_cameras_path)
            
            os.makedirs(self.depth_path, exist_ok=True)
            os.makedirs(self.masks_path, exist_ok=True)
            os.makedirs(self.rgb_path, exist_ok=True)
            if hasattr(self, 'rgb_cameras_path'):
                os.makedirs(self.rgb_cameras_path, exist_ok=True)
            
            data_copy = self.transforms_data.copy()
            data_copy["frames"] = []
            
            rgb_frames = []
            for i, frame in enumerate(getattr(self, 'frame_buffer', [])):
                file_name_base = f"frame_{i:05d}"
                
                # Save Depth
                depth_filename = f"{file_name_base}.npy"
                depth_filepath = os.path.join(self.depth_path, depth_filename)
                np.save(depth_filepath, frame["depth"])
                
                # Save Mask
                mask_filename = None
                if frame["mask"] is not None:
                    mask_filename = f"{file_name_base}.png"
                    mask_filepath = os.path.join(self.masks_path, mask_filename)
                    mask_img = Image.fromarray((frame["mask"] * 255).astype(np.uint8))
                    mask_img.save(mask_filepath)
                    
                # Collect RGB for GS and save to disk
                rgb_filename = None
                if frame.get("rgb") is not None:
                    rgb_filename = f"{file_name_base}.png"
                    rgb_filepath = os.path.join(self.rgb_path, rgb_filename)
                    # Save natively as 8-bit RGB image
                    rgb_to_save = frame["rgb"]
                    if rgb_to_save.dtype != np.uint8:
                        rgb_to_save = (rgb_to_save * 255).astype(np.uint8)
                    rgb_img = Image.fromarray(rgb_to_save)
                    rgb_img.save(rgb_filepath)

                # Collect RGB Cameras for video tracing and save to disk
                if frame.get("rgb_cameras") is not None:
                    rgb_cameras_to_save = frame["rgb_cameras"]
                    if rgb_cameras_to_save.dtype != np.uint8:
                        rgb_cameras_to_save = (rgb_cameras_to_save * 255).astype(np.uint8)
                    rgb_frames.append(rgb_cameras_to_save)
                    if hasattr(self, 'rgb_cameras_path'):
                        rgb_cameras_filepath = os.path.join(self.rgb_cameras_path, f"{file_name_base}.png")
                        Image.fromarray(rgb_cameras_to_save).save(rgb_cameras_filepath)

                # Add Entry
                frame_entry = {
                    "file_path": f"rgb/{rgb_filename}" if rgb_filename else f"depth/{depth_filename}",
                    "depth_file_path": f"depth/{depth_filename}",
                    "transform_matrix": frame["transform_matrix"],
                    "fl_x": frame.get("fl_x", data_copy.get("fl_x")),
                    "fl_y": frame.get("fl_y", data_copy.get("fl_y")),
                    "cx": frame.get("cx", data_copy.get("cx")),
                    "cy": frame.get("cy", data_copy.get("cy"))
                }
                if mask_filename:
                    frame_entry["mask_path"] = f"masks/{mask_filename}"
                data_copy["frames"].append(frame_entry)

            # Write transforms.json entirely
            json_path = os.path.join(self.output_path, "transforms.json")
            with open(json_path, "w") as f:
                json.dump(data_copy, f, indent=4)
                
            self.save_summary(faces_discovered, max_distance)
            
            # Write RGB Video
            if len(rgb_frames) > 0:
                try:
                    import imageio
                    video_path = os.path.join(self.output_path, "best_episode.mp4")
                    # fps calculation: 1 / (dt * decimation * save_interval). 
                    # Assuming standard Isaac Sim (0.0083 * 6 = ~0.05 step).
                    playback_speed = getattr(self.cfg, "video_playback_speed", 1.2)
                    fps = (1.0 / (0.05 * self.save_interval)) * playback_speed
                    print(f"[ReconstructionDataCollector] Generating RGB Video with {len(rgb_frames)} frames at {fps:.2f} FPS (x{playback_speed:.2f} speed)...")
                    writer = imageio.get_writer(video_path, fps=fps)
                    for rgb in rgb_frames:
                        writer.append_data(rgb)
                    writer.close()
                except Exception as e:
                    print(f"[ReconstructionDataCollector ERROR] Failed to save video: {e}")

        # Always clear buffer after episode
        self.frame_buffer = []
        self.frame_idx = 0
        self.is_first_frame = True

    def save(self):
        print(f"[ReconstructionDataCollector] Finalized.")

    def save_summary(self, max_faces, max_position):
        try:
            summary = {
                "max_faces": int(max_faces),
                "max_position_distance": float(max_position)
            }
            summary_path = os.path.join(self.output_path, "summary.json")
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=4)
            print(f"[ReconstructionDataCollector] Saved summary to {summary_path}")
        except Exception as e:
            print(f"[ReconstructionDataCollector ERROR] Failed to save summary: {e}")
