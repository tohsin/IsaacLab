
import os
import json
import torch
import numpy as np
import shutil
from datetime import datetime, timezone
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
        base_output_path = getattr(cfg, "data_recording_path", "data/recorded_point_clouds")
        if getattr(cfg, "create_timestamped_run", False):
            run_name = datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")
            self.output_path = os.path.join(base_output_path, run_name)
        else:
            self.output_path = base_output_path
        self.save_interval = getattr(cfg, "save_interval", 1)
        self.save_images = bool(getattr(cfg, "save_images", False))
        # Preserve the old coupled behavior for configurations that do not yet
        # define save_video explicitly.
        self.save_video = bool(getattr(cfg, "save_video", self.save_images))
        
        self.depth_path = os.path.join(self.output_path, "depth")
        self.masks_path = os.path.join(self.output_path, "masks")
        self.rgb_path = os.path.join(self.output_path, "rgb")
        self.rgb_cameras_path = os.path.join(self.output_path, "rgb_cameras")
        
        # Cleanup and Create Directories
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)
        os.makedirs(self.depth_path, exist_ok=True)
        os.makedirs(self.masks_path, exist_ok=True)
        if self.save_images:
            os.makedirs(self.rgb_path, exist_ok=True)
            os.makedirs(self.rgb_cameras_path, exist_ok=True)
        
        enabled_outputs = ["Depth", "Masks"]
        if self.save_images:
            enabled_outputs.append("RGB images")
        if self.save_video:
            enabled_outputs.append("Video")
        print(
            "[ReconstructionDataCollector] Initialized. Saving "
            f"{'/'.join(enabled_outputs)} to: {self.output_path}"
        )
        
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
        self.episode_idx = 0
        self.episode_results = []
        
    def collect(
        self,
        camera,
        step_idx,
        semantic_mask=None,
        nav_camera=None,
        target_position=None,
        target_orientation=None,
        episode_step=None,
    ):
        """
        Collects data from the camera.
        
        Args:
            camera: The TiledCamera object.
            step_idx: Current simulation step.
            semantic_mask: Optional boolean tensor (N, H, W, 1) to filter points.
            target_position: Optional target root position in world coordinates.
            target_orientation: Optional target root quaternion (w, x, y, z).
            episode_step: Optional episode-local policy step.
        """
        if step_idx % self.save_interval != 0:
            return

        env_id = 0
        if "distance_to_image_plane" not in camera.data.output:
             return

        # Get Data
        depth_tensor = camera.data.output["distance_to_image_plane"][env_id] # (H, W) or (H, W, 1)
        intrinsic_matrix = camera.data.intrinsic_matrices[env_id]
        cam_pos_w = camera.data.pos_w[env_id]
        cam_quat_w = camera.data.quat_w_world[env_id]
        
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
        # Capture RGB when either individual images or an episode video is
        # requested. Video frames can be buffered without writing frame PNGs.
        if (self.save_images or self.save_video) and "rgb" in camera.data.output:
            rgb_tensor = camera.data.output["rgb"][env_id] # (H, W, 4) or (H, W, 3)
            inspection_rgb_np = rgb_tensor.cpu().numpy()
            if inspection_rgb_np.shape[-1] == 4:
                inspection_rgb_np = inspection_rgb_np[..., :3] # Remove Alpha

            if self.save_images:
                rgb_np = inspection_rgb_np
                
            rgb_cameras_np = inspection_rgb_np.copy()
            # If nav_camera is provided, stitch side-by-side
            if nav_camera is not None and "rgb" in nav_camera.data.output:
                nav_rgb_tensor = nav_camera.data.output["rgb"][env_id]
                nav_rgb_np = nav_rgb_tensor.cpu().numpy()
                if nav_rgb_np.shape[-1] == 4:
                    nav_rgb_np = nav_rgb_np[..., :3]
                
                # Resize if heights don't match
                if inspection_rgb_np.shape[0] != nav_rgb_np.shape[0]:
                    import cv2
                    h = inspection_rgb_np.shape[0]
                    w = int(nav_rgb_np.shape[1] * (h / nav_rgb_np.shape[0]))
                    nav_rgb_np = cv2.resize(nav_rgb_np, (w, h))

                # Concatenate horizontally
                rgb_cameras_np = np.concatenate((nav_rgb_np, inspection_rgb_np), axis=1)

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
            "cy": cy_frame,
            "target_position": (
                target_position.detach().cpu().tolist()
                if target_position is not None
                else None
            ),
            "target_orientation": (
                target_orientation.detach().cpu().tolist()
                if target_orientation is not None
                else None
            ),
            "episode_step": int(episode_step) if episode_step is not None else None,
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

    def flush_if_best(self, faces_discovered, episode_metadata=None):
        """Flush reconstruction data and record optional episode metrics.

        With ``record_all_episodes`` enabled, every completed episode is written
        to its own directory while RAM remains bounded to one episode. The
        legacy mode keeps only the best face-proxy episode.
        """
        episode_result = None
        if episode_metadata is not None:
            episode_result = self._record_episode_result(
                faces_discovered, episode_metadata
            )

        if not hasattr(self, 'best_faces_discovered'):
            self.best_faces_discovered = 0

        is_best = faces_discovered > self.best_faces_discovered
        record_all = bool(getattr(self.cfg, "record_all_episodes", False))
        should_write = is_best or (record_all and episode_result is not None)

        if is_best:
            self.best_faces_discovered = faces_discovered

        if should_write:
            if record_all and episode_result is not None:
                write_output_path = os.path.join(
                    self.output_path,
                    "episodes",
                    f"episode_{episode_result['episode_index']:05d}",
                )
                print(
                    "[ReconstructionDataCollector] Flushing completed episode "
                    f"{episode_result['episode_index']} to disk. Faces: {faces_discovered}"
                )
            else:
                write_output_path = self.output_path
                print(
                    "[ReconstructionDataCollector] Flushing new best episode to disk. "
                    f"Faces: {faces_discovered}"
                )

            depth_path = os.path.join(write_output_path, "depth")
            masks_path = os.path.join(write_output_path, "masks")
            rgb_path = os.path.join(write_output_path, "rgb")
            rgb_cameras_path = os.path.join(write_output_path, "rgb_cameras")

            # RGB directories are omitted in video-only mode.
            output_directories = [depth_path, masks_path]
            if self.save_images:
                output_directories.extend((rgb_path, rgb_cameras_path))
            for path in output_directories:
                if os.path.exists(path):
                    shutil.rmtree(path)
                os.makedirs(path, exist_ok=True)
            
            data_copy = self.transforms_data.copy()
            data_copy["frames"] = []
            
            rgb_frames = []
            for i, frame in enumerate(getattr(self, 'frame_buffer', [])):
                file_name_base = f"frame_{i:05d}"
                
                # Save Depth
                depth_filename = f"{file_name_base}.npy"
                depth_filepath = os.path.join(depth_path, depth_filename)
                np.save(depth_filepath, frame["depth"])
                
                # Save Mask
                mask_filename = None
                if frame["mask"] is not None:
                    mask_filename = f"{file_name_base}.png"
                    mask_filepath = os.path.join(masks_path, mask_filename)
                    mask_img = Image.fromarray((frame["mask"] * 255).astype(np.uint8))
                    mask_img.save(mask_filepath)
                    
                # Collect RGB for GS and save to disk
                rgb_filename = None
                if frame.get("rgb") is not None:
                    rgb_filename = f"{file_name_base}.png"
                    rgb_filepath = os.path.join(rgb_path, rgb_filename)
                    # Save natively as 8-bit RGB image
                    rgb_to_save = frame["rgb"]
                    if rgb_to_save.dtype != np.uint8:
                        rgb_to_save = (rgb_to_save * 255).astype(np.uint8)
                    rgb_img = Image.fromarray(rgb_to_save)
                    rgb_img.save(rgb_filepath)

                # Collect stitched RGB for video. Only retain frame PNGs when
                # save_images is enabled.
                if frame.get("rgb_cameras") is not None:
                    rgb_cameras_to_save = frame["rgb_cameras"]
                    if rgb_cameras_to_save.dtype != np.uint8:
                        rgb_cameras_to_save = (rgb_cameras_to_save * 255).astype(np.uint8)
                    if self.save_video:
                        rgb_frames.append(rgb_cameras_to_save)
                    if self.save_images:
                        rgb_cameras_filepath = os.path.join(rgb_cameras_path, f"{file_name_base}.png")
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
                if frame.get("target_position") is not None:
                    frame_entry["target_position"] = frame["target_position"]
                if frame.get("target_orientation") is not None:
                    frame_entry["target_orientation"] = frame["target_orientation"]
                if frame.get("episode_step") is not None:
                    frame_entry["episode_step"] = frame["episode_step"]
                if mask_filename:
                    frame_entry["mask_path"] = f"masks/{mask_filename}"
                data_copy["frames"].append(frame_entry)

            # Write transforms.json entirely
            json_path = os.path.join(write_output_path, "transforms.json")
            with open(json_path, "w") as f:
                json.dump(data_copy, f, indent=4)

            self.save_summary(faces_discovered, write_output_path)

            if record_all and is_best and episode_result is not None:
                best_path = os.path.join(self.output_path, "best_episode.json")
                with open(best_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "episode_index": episode_result["episode_index"],
                            "path": os.path.relpath(write_output_path, self.output_path),
                            "faces_discovered": int(faces_discovered),
                            "selection_note": "convenience pointer only; not geometric success",
                        },
                        f,
                        indent=2,
                    )
            
            # Write RGB Video
            if len(rgb_frames) > 0:
                try:
                    import imageio
                    video_path = os.path.join(write_output_path, "episode.mp4")
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

    def _record_episode_result(self, faces_discovered, metadata):
        result = {
            "episode_index": self.episode_idx,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "faces_discovered": int(faces_discovered),
            **metadata,
        }
        self.episode_results.append(result)
        self.episode_idx += 1

        results_path = os.path.join(self.output_path, "episode_results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(self.episode_results, f, indent=2)

        durations = np.asarray(
            [item["duration_seconds"] for item in self.episode_results], dtype=np.float64
        )
        crashed = np.asarray(
            [bool(item["crashed"]) for item in self.episode_results], dtype=np.bool_
        )
        face_goal_reached = np.asarray(
            [bool(item["face_goal_reached"]) for item in self.episode_results], dtype=np.bool_
        )
        summary = {
            "episodes": len(self.episode_results),
            "crashed_episodes": int(crashed.sum()),
            "crash_rate_percent": float(crashed.mean() * 100.0),
            "face_goal_reached_rate_percent": float(face_goal_reached.mean() * 100.0),
            "duration_seconds": {
                "mean": float(durations.mean()),
                "min": float(durations.min()),
                "max": float(durations.max()),
            },
            "note": "face_goal_reached is a proxy; geometric coverage is computed offline",
        }
        summary_path = os.path.join(self.output_path, "evaluation_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return result

    def save(self):
        print(f"[ReconstructionDataCollector] Finalized.")

    def save_summary(self, max_faces, output_path=None):
        try:
            summary = {
                "max_faces": int(max_faces),
            }
            summary_path = os.path.join(output_path or self.output_path, "summary.json")
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=4)
            print(f"[ReconstructionDataCollector] Saved summary to {summary_path}")
        except Exception as e:
            print(f"[ReconstructionDataCollector ERROR] Failed to save summary: {e}")
