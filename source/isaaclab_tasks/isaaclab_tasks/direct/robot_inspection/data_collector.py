
import os
import json
import torch
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import shutil

class DataCollector:
    def __init__(self, cfg, device="cpu"):
        self.cfg = cfg
        self.device = device
        self.output_path = getattr(cfg, "data_recording_path", "data/recorded_trajectory")
        self.images_path = os.path.join(self.output_path, "images")
        self.save_images = getattr(cfg, "save_images", True)
        
        # Cleanup and Create Directories
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)
        os.makedirs(self.images_path, exist_ok=True)
        
        print(f"[DataCollector] Initialized. Saving to: {self.output_path}")
        
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
        
        self.frame_idx = 0
        self.intrinsics_set = False

    def collect(self, ptz_camera, step_idx):
        """
        Collect data from the Inspection Camera (PTZ).
        Args:
            ptz_camera: The TiledCamera object.
            step_idx: Current simulation step.
        """
        # Check interval
        save_interval = getattr(self.cfg, "save_interval", 1)
        if step_idx % save_interval != 0:
            return

        # We only record environment 0 for now as per run_config num_envs=1
        env_id = 0
        
        # 1. Get Image Data
        # shape: (N, H, W, 4) or (N, H, W, 3)
        # 1. Get Image Data
        # shape: (N, H, W, 4) or (N, H, W, 3)
        if "rgb" not in ptz_camera.data.output:
            print(f"[DataCollector DEBUG] No RGB data in ptz_camera output. Keys: {ptz_camera.data.output.keys()}")
            return
            
        rgb_tensor = ptz_camera.data.output["rgb"][env_id]
        
        # 2. Get Pose Data (World Frame)
        # Position (3,), Quaternion (4,) [w, x, y, z]
        cam_pos_w = ptz_camera.data.pos_w[env_id]
        cam_quat_w = ptz_camera.data.quat_w_world[env_id]
        
        # 3. Get Intrinsics
        K = ptz_camera.data.intrinsic_matrices[env_id]
        
        self._process_frame(rgb_tensor, cam_pos_w, cam_quat_w, K, step_idx)

    def _process_frame(self, rgb_tensor, pos, quat, K, step_idx):
        # --- Handle Intrinsics (Once) ---
        if not self.intrinsics_set:
            self._set_intrinsics(K, rgb_tensor.shape[1], rgb_tensor.shape[0])
            self.intrinsics_set = True
            
        # --- Save Image ---
        file_name = f"frame_{self.frame_idx:05d}.png"
        if self.save_images:
            rgb_np = rgb_tensor.cpu().numpy()
            
            # Normalize if needed (assuming 0-255 uint8 or 0-1 float)
            if rgb_np.dtype == np.uint8:
                pass
            elif rgb_np.max() <= 1.0:
                 rgb_np = (rgb_np * 255).astype(np.uint8)
            else:
                 rgb_np = rgb_np.astype(np.uint8)

            # Remove alpha
            if rgb_np.shape[2] == 4:
                rgb_np = rgb_np[:, :, :3]
                
            img = Image.fromarray(rgb_np)
            img.save(os.path.join(self.images_path, file_name))
            
        # --- Save Pose (Transform Matrix) ---
        # Convert to 4x4 Matrix
        mat = self._get_matrix(pos.cpu().numpy(), quat.cpu().numpy())
        
        frame_entry = {
            "file_path": f"images/{file_name}",
            "transform_matrix": mat.tolist()
        }
        
        # Incrementally save to JSON
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
        self.transforms_data["camera_angle_x"] = 2 * math.atan(width / (2 * fl_x))
    
    def _get_matrix(self, pos, quat):
        # Isaac Lab Quat is (w, x, y, z)
        # Scipy Rotation expects (x, y, z, w)
        rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        mat = np.eye(4)
        mat[:3, :3] = rot.as_matrix()
        mat[:3, 3] = pos
        return mat

    def _append_to_json(self, frame_entry):
        """Appends a single frame entry to transforms.json in a crash-safe way."""
        json_path = os.path.join(self.output_path, "transforms.json")
        
        if self.frame_idx == 0:
            # First frame: Write the full initial JSON structure
            # We use the current state of self.transforms_data which has intrinsics set
            data_copy = self.transforms_data.copy()
            data_copy["frames"] = [frame_entry]
            
            with open(json_path, "w") as f:
                json.dump(data_copy, f, indent=4)
                
        else:
            # Subsequent frames: Append to the "frames" list
            # We assume the file ends with:
            #     ]
            # }
            # We want to remove the last 2 lines (or find the closing bracket)
            # and append: , { ... } ] }
            
            with open(json_path, "rb+") as f:
                f.seek(0, os.SEEK_END)
                pos = f.tell()
                
                # Check backward for the last ']'
                # We expect the file to end with something like `    ]\n}`
                # We scan back a reasonable amount (e.g. 100 bytes) to find the list closer
                search_limit = min(pos, 200)
                found_closer = False
                
                for i in range(1, search_limit + 1):
                    f.seek(-i, os.SEEK_END)
                    char = f.read(1)
                    if char == b']':
                        # Found the end of the list. Move pointer just before it.
                        f.seek(-i, os.SEEK_END)
                        found_closer = True
                        break
                
                if found_closer:
                    # Write separator and new entry
                    # Prepend a comma since we are appending to an existing list item
                    f.write(b",\n")
                    
                    # Dump the new entry with indentation
                    # To match the indentation of existing items (usually 8 spaces if root is 4)
                    json_str = json.dumps(frame_entry, indent=4)
                    # Indent this block
                    json_str = "        " + json_str.replace("\n", "\n        ")
                    
                    f.write(json_str.encode('utf-8'))
                    
                    # Close the list and object
                    f.write(b"\n    ]\n}")
                    f.truncate() # Ensure we cut off any old remaining bytes
                else:
                    print(f"[DataCollector ERROR] Could not find closing bracket in {json_path}. Appending failed.")

    def save(self):
        # No-op since we match incremental saving
        # But maybe just print a message
        print(f"[DataCollector] Finalized. Total frames: {self.frame_idx}")
