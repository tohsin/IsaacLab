"""
Script to convert recorded depth and mask data to a merged point cloud.
Reads data/recorded_point_clouds/transforms.json and associated files.
Standalone version: Does NOT require isaaclab or carb dependencies.
"""

import os
import json
import argparse
import numpy as np
import torch
import trimesh
from PIL import Image
from tqdm import tqdm
from scipy.spatial.transform import Rotation

def create_pointcloud_from_depth(intrinsic_matrix, depth, position=None, orientation=None, device="cpu"):
    """
    Creates point cloud from depth map using pinhole camera model.
    Re-implementation of isaaclab.sensors.camera.utils logic to avoid deps.
    
    Args:
        intrinsic_matrix: (3, 3) tensor
        depth: (H, W) tensor
        position: (3,) tensor (optional)
        orientation: (4,) tensor (w, x, y, z) (optional)
    """
    h, w = depth.shape
    
    # Create meshgrid
    # u: x-coord (0 to W-1), v: y-coord (0 to H-1)
    u = torch.arange(w, device=device)
    v = torch.arange(h, device=device)
    v, u = torch.meshgrid(v, u, indexing='ij')
    
    # Flatten
    u = u.flatten()
    v = v.flatten()
    z = depth.flatten()
    
    # Filter invalid depth
    valid_mask = (z > 0) & (~torch.isnan(z)) & (~torch.isinf(z))
    
    u = u[valid_mask]
    v = v[valid_mask]
    z = z[valid_mask]
    
    # Unproject
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # Stack to (N, 3)
    # Camera Frame: X Right, Y Down, Z Forward (Standard OpenCV/Isaac Pinhole)
    points_c = torch.stack([x, -y, -z], dim=1)
    
    # Transform to World Frame
    if position is not None and orientation is not None:
        # Rotate
        # orientation is (w, x, y, z) -> (x, y, z, w) for scipy if needed
        # But we are in torch.
        
        # Convert quat (w, x, y, z) to rotation matrix
        # Or just use scipy for the rotation part to be safe/easy if not performance critical,
        # but mixed torch/numpy is annoying. 
        # Let's implement simple quat rotate or convert to matrix.
        
        # Simple Quat to Mat (w, x, y, z)
        # Assuming normalized quaternion
        # R * p + t
        
        w, x, y, z = orientation.unbind()
        R = torch.stack([
            1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w,     2*x*z + 2*y*w,
            2*x*y + 2*z*w,     1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w,
            2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x**2 - 2*y**2
        ]).reshape(3, 3).to(device)
        
        points_w = points_c @ R.T + position
        return points_w
        
    return points_c
    # parser.add_argument("--output", type=str, default="data/point_clouds/flange_pc.ply", help="Output file.")
def main():
    parser = argparse.ArgumentParser(description="Convert recorded depth to point cloud.")
    parser.add_argument("--data_path", type=str, default="data/recorded_depth_data_eval", help="Path to recording directory.")
    parser.add_argument("--output", type=str, default="data/point_clouds/SEEIR_eval_point_cloud.ply", help="Output file.")
    parser.add_argument("--center", action="store_true", help="Center the final cloud.")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device for torch operations.")
    parser.add_argument("--downsample", type=float, default=0.005, help="Voxel downsample size (meters). 0 to disable.")
    
    args = parser.parse_args()
    device = args.device
    print(f"[INFO] Using device: {device}")
    
    json_path = os.path.join(args.data_path, "transforms.json")
    if not os.path.exists(json_path):
        print(f"[ERROR] transforms.json not found in {args.data_path}")
        return
        
    with open(json_path, "r") as f:
        data = json.load(f)
        
    print(f"[INFO] Loaded {len(data['frames'])} frames.")
    
    all_points = []
    
    print("[INFO] Processing frames...")
    for frame in tqdm(data['frames']):
        # Per-frame Intrinsics (supports dynamic zoom)
        K = np.eye(3)
        K[0, 0] = frame.get('fl_x', data['fl_x'])
        K[1, 1] = frame.get('fl_y', data['fl_y'])
        K[0, 2] = frame.get('cx', data['cx'])
        K[1, 2] = frame.get('cy', data['cy'])
        intrinsic_matrix = torch.tensor(K, dtype=torch.float32, device=device)
        # Load Depth
        depth_path = os.path.join(args.data_path, frame["file_path"])
        if not os.path.exists(depth_path):
            continue
            
        depth_np = np.load(depth_path)
        depth_tensor = torch.tensor(depth_np, dtype=torch.float32, device=device)
        
        # Load Mask
        mask_np = None
        if "mask_path" in frame:
            mask_path = os.path.join(args.data_path, frame["mask_path"])
            if os.path.exists(mask_path):
                  mask_np = np.array(Image.open(mask_path)) > 0
        
        # Load Transform (4x4 Matrix)
        c2w = np.array(frame["transform_matrix"])
        
        # Extract Pos/Rot
        pos = c2w[:3, 3]
        rot_mat = c2w[:3, :3]
        
        # Convert to Quat (w, x, y, z) for our function
        rot = Rotation.from_matrix(rot_mat)
        quat_xyzw = rot.as_quat() 
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        
        pos_tensor = torch.tensor(pos, dtype=torch.float32, device=device)
        quat_tensor = torch.tensor(quat_wxyz, dtype=torch.float32, device=device)
        
        # Generate Points
        points_w = create_pointcloud_from_depth(
            intrinsic_matrix=intrinsic_matrix,
            depth=depth_tensor,
            position=pos_tensor,
            orientation=quat_tensor,
            device=device
        )
        
        # Apply Mask
        # points_w corresponds to valid depth pixels. 
        # We need to filter based on mask.
        
        if mask_np is not None:
             # We need to flatten mask using same valid_mask logic
             mask_flat = torch.tensor(mask_np.flatten(), device=device)
             depth_flat = depth_tensor.flatten()
             
             # Re-compute valid mask from depth to align
             valid_depth = (depth_flat > 0) & (~torch.isnan(depth_flat)) & (~torch.isinf(depth_flat))
             
             # Filter semantic mask by valid depth
             mask_valid_depth = mask_flat[valid_depth]
             
             # Apply semantic filter to points
             if points_w.shape[0] == mask_valid_depth.shape[0]:
                 points_filtered = points_w[mask_valid_depth]
                 all_points.append(points_filtered.cpu().numpy())
             else:
                 # Should not happen
                 pass
        else:
             all_points.append(points_w.cpu().numpy())

    if not all_points:
        print("[WARN] No points collected.")
        return

    print("[INFO] Merging points...")
    merged_points = np.vstack(all_points)
    print(f"[INFO] Total points: {merged_points.shape[0]}")
    
    # Centering
    if args.center:
        centroid = np.mean(merged_points, axis=0)
        merged_points -= centroid
        print(f"[INFO] Centered. Centroid was: {centroid}")
        
    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    pcd = trimesh.points.PointCloud(merged_points)
    
    try:
        import open3d as o3d
        print("[INFO] Using Open3D for optimization/saving...")
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(merged_points)
        
        if args.downsample > 0:
            print(f"[INFO] Downsampling with voxel size: {args.downsample}")
            pcd_o3d = pcd_o3d.voxel_down_sample(voxel_size=args.downsample)
            
        o3d.io.write_point_cloud(args.output, pcd_o3d)
        print(f"[INFO] Saved to {args.output} (Open3D)")
        
    except ImportError:
        print("[INFO] Open3D not found. Saving full cloud with Trimesh.")
        pcd.export(args.output)
        print(f"[INFO] Saved to {args.output} (Trimesh)")

if __name__ == "__main__":
    main()
