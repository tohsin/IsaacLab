
import torch
import numpy as np
from isaaclab.sensors.camera.utils import create_pointcloud_from_depth
from isaaclab.utils.math import quat_mul

class EvalReconstructionDataCollector:
    """
    A lightweight collector for accumulating point clouds during an evaluation episode.
    Designed to work in-memory for single-environment evaluation to produce a fused point cloud
    of the target object (filtered by semantic mask).
    """
    def __init__(self, device="cpu"):
        self.device = device
        self.episode_points = []
        
        # Quaternion to rotate from Optical Frame (Z-forward) to ROS Camera Frame (X-forward)
        # R_optical_to_ros = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
        # q = (w=0.5, x=-0.5, y=0.5, z=-0.5)
        self.optical_to_ros_quat = torch.tensor([0.5, -0.5, 0.5, -0.5], device=device)
        self.chassis_min_dist_sq = 0.4 ** 2

    def reset(self):
        """Clears the accumulated point cloud buffer."""
        self.episode_points = []
        self.is_first_frame = True

    def add_frame(self, depth, intrinsic_matrix, position, orientation, semantic_mask=None):
        """
        Processes a single frame:
        1. Unprojects depth to 3D points.
        2. Filters using semantic mask (if provided).
        3. Filters using basic distance checks.
        4. Accumulates valid points.

        Args:
            depth (torch.Tensor): Depth map (H, W).
            intrinsic_matrix (torch.Tensor): Camera intrinsics (3, 3).
            position (torch.Tensor): Camera position (3,).
            orientation (torch.Tensor): Camera orientation (4,).
            semantic_mask (torch.Tensor, optional): Boolean mask (H, W) or (H, W, 1). True = Target.
        """
        # Skip the very first frame of every episode to prevent stale "phantom" objects
        # from being projected across episode boundaries
        if getattr(self, "is_first_frame", True):
            self.is_first_frame = False
            return
        # 1. Apply Semantic Mask to Depth (Early Filtering)
        # If we have a mask, we can set non-target depth to a "invalid" value (e.g. 0 or -1)
        # to prevent those points from being generated or to easily filter them later.
        
        d = depth.clone()
        
        if semantic_mask is not None:
            # Ensure mask is (H, W)
            if semantic_mask.dim() == 3:
                semantic_mask = semantic_mask.squeeze(-1)
            
            # Set depth of non-target pixels to 0 (effectively filtering them out)
            d[~semantic_mask] = 0.0

            if semantic_mask.sum() == 0:
                 pass
                 # print("[EvalReconstructionDataCollector] WARNING: Semantic Mask provided but has 0 target pixels!")
        else:
             pass
             # print("[EvalReconstructionDataCollector] WARNING: No Semantic Mask provided! Full scene will be reconstructed.")

        # Adjust Orientation:
        # The camera orientation from Isaac Lab text is typically in the ROS frame (X-forward).
        # However, `create_pointcloud_from_depth` assumes pinhole/optical frame (Z-forward) for the points it generates.
        # So we need to rotate the "camera frame" that the points are in (Optical) to match the "camera body frame" (ROS)
        # before applying the body-to-world rotation.
        # orientation_world_acc_optical = orientation_world_acc_ros * q_ros_acc_optical
        
        # Convert orientation from optical (Z-forward) to ROS (X-forward) frame
        corrected_orientation = quat_mul(orientation, self.optical_to_ros_quat)

        # 2. Generate Point Cloud
        # Generate point cloud using corrected orientation
        points = create_pointcloud_from_depth(
                intrinsic_matrix=intrinsic_matrix,
                depth=d,
                position=position,
                orientation=corrected_orientation,
                device=self.device,
        )

        # 3. Filter Points
        if points.shape[0] > 0:
            # Filter out points that were set to 0 depth (they stick to camera origin)
            # OR just general min distance filtering
            # We check distance from camera center
            # dist_sq = torch.sum((points - position)**2, dim=1)
            # valid_mask = dist_sq > 0.05**2 # 5cm min distance
            
            # Actually, `create_pointcloud_from_depth` in Isaac Lab usually returns valid points?
            # If d=0, it might return the camera position.
            # Let's check distance from camera to be sure.
            
            dist_sq = torch.sum((points - position)**2, dim=1)
            # Filter minimal distance (e.g. 10cm) and optionally chassis distance if needed
            valid_mask = dist_sq > 0.01 

            # If we used the semantic mask on depth, the points for non-target pixels
            # should be at the camera origin (dist ~ 0). So this distance filter effectively
            # removes the non-semantic points too!
            
            # Apply Filter
            points = points[valid_mask]

            # 4. Downsample if too large (optional, to save memory during long episodes)
            # But for high fidelity reconstruction we might want to keep more?
            # Let's cap per frame to avoid explosion, but keep it high.
            max_points_per_frame = 4096 
            if points.shape[0] > max_points_per_frame:
                 perm = torch.randperm(points.shape[0], device=self.device)
                 points = points[perm[:max_points_per_frame]]

            # 5. Accumulate
            if points.shape[0] > 0:
                self.episode_points.append(points.cpu().numpy())

    def get_full_cloud(self):
        """
        Returns the accumulated point cloud as a numpy array (N, 3).
        Returns None if empty.
        """
        if len(self.episode_points) == 0:
            return None
        
        try:
            return np.concatenate(self.episode_points, axis=0)
        except Exception as e:
            print(f"[EvalReconstructionDataCollector] Error enforcing concatenation: {e}")
            return None
