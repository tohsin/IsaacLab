
import os
import json
import torch
import numpy as np
import shutil
from PIL import Image
from scipy.spatial.transform import Rotation

class PointCloudCollector:
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
        
        # Cleanup and Create Directories
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)
        os.makedirs(self.depth_path, exist_ok=True)
        os.makedirs(self.masks_path, exist_ok=True)
        
        print(f"[PointCloudCollector] Initialized. Saving Depth/Masks to: {self.output_path}")
        
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
        
    def collect(self, camera, step_idx, semantic_mask=None):
        """
        Collects data from the camera.
        
        Args:
            camera: The TiledCamera object.
            step_idx: Current simulation step.
            semantic_mask: Optional boolean tensor (N, H, W, 1) to filter points.
        """
        if step_idx % self.save_interval != 0:
            return

        # print(f"[PointCloudCollector] collecting frame {self.frame_idx} at step {step_idx}")
        # if semantic_mask is not None:
             # print(f"  Mask shape: {semantic_mask.shape}")

        # We only record environment 0 for simplicity
        env_id = 0
        
        # Check for depth data
        if "distance_to_image_plane" not in camera.data.output:
             print(f"[PointCloudCollector DEBUG] No Depth data in camera output. Keys: {camera.data.output.keys()}")
             return

        # Get Data
        depth_tensor = camera.data.output["distance_to_image_plane"][env_id] # (H, W) or (H, W, 1)
        # print(f"  Depth shape: {depth_tensor.shape}")
        intrinsic_matrix = camera.data.intrinsic_matrices[env_id]
        cam_pos_w = camera.data.pos_w[env_id]
        cam_quat_w = camera.data.quat_w_world[env_id]
        
        # --- Handle Intrinsics (Once) ---
        if not self.intrinsics_set:
            h, w = depth_tensor.shape[:2]
            self._set_intrinsics(intrinsic_matrix, w, h)
            self.intrinsics_set = True

        # --- Save Depth (.npy) ---
        file_name_base = f"frame_{self.frame_idx:05d}"
        depth_filename = f"{file_name_base}.npy"
        depth_filepath = os.path.join(self.depth_path, depth_filename)
        
        depth_np = depth_tensor.cpu().numpy()
        if depth_np.ndim == 3:
            depth_np = depth_np.squeeze(-1) # Ensure (H, W)
            
        np.save(depth_filepath, depth_np)
        
        # --- Save Mask (.png) ---
        mask_filename = None
        if semantic_mask is not None:
            mask_filename = f"{file_name_base}.png"
            mask_filepath = os.path.join(self.masks_path, mask_filename)
            
            mask_tensor = semantic_mask[env_id] # (H, W, 1)
            mask_np = mask_tensor.cpu().numpy().squeeze(-1) # (H, W) boolean
            
            # Save as uint8 (0 or 255)
            mask_img = Image.fromarray((mask_np * 255).astype(np.uint8))
            mask_img.save(mask_filepath)
            
        # --- Save Transform ---
        mat = self._get_matrix(cam_pos_w.cpu().numpy(), cam_quat_w.cpu().numpy())
        
        frame_entry = {
            "file_path": f"depth/{depth_filename}",
            "transform_matrix": mat.tolist()
        }
        
        if mask_filename:
            frame_entry["mask_path"] = f"masks/{mask_filename}"
            
        self._append_to_json(frame_entry)
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
        # Usually Isaac Sim cameras look down -Z in their local frame if using correct convention?
        # Actually, Isaac Sim cameras are often Y-up or Z-up depending on stage. 
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

    def _append_to_json(self, frame_entry):
        """Appends a single frame entry to transforms.json in a crash-safe way."""
        json_path = os.path.join(self.output_path, "transforms.json")
        
        if self.frame_idx == 0:
            data_copy = self.transforms_data.copy()
            data_copy["frames"] = [frame_entry]
            
            with open(json_path, "w") as f:
                json.dump(data_copy, f, indent=4)
                
        else:
            # Append safely
            with open(json_path, "rb+") as f:
                f.seek(0, os.SEEK_END)
                pos = f.tell()
                
                # Check backward for the last ']'
                search_limit = min(pos, 200)
                found_closer = False
                
                for i in range(1, search_limit + 1):
                    f.seek(-i, os.SEEK_END)
                    char = f.read(1)
                    if char == b']':
                        f.seek(-i, os.SEEK_END)
                        found_closer = True
                        break
                
                if found_closer:
                    f.write(b",\n")
                    json_str = json.dumps(frame_entry, indent=4)
                    json_str = "        " + json_str.replace("\n", "\n        ")
                    f.write(json_str.encode('utf-8'))
                    f.write(b"\n    ]\n}")
                    f.truncate()
                else:
                    print(f"[PointCloudCollector ERROR] Could not find closing bracket in {json_path}. Appending failed.")

    def save(self):
        print(f"[PointCloudCollector] Finalized. Total frames: {self.frame_idx}")
