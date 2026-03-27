"""
Script to generate a point cloud from an inspection object in Isaac Sim.
Uses Ray Casting to capture only the exterior visible surface.
Supports scaling and positioning to match simulation environment.
"""

from __future__ import annotations

import argparse
from isaaclab.app import AppLauncher
#python generate_object_pointcloud.py --dataset_key forklift --headless

# Create the parser
parser = argparse.ArgumentParser(description="Generate a point cloud from an inspection object.")
parser.add_argument("--usd_path", type=str, default=None, help="Path to the USD file.")
parser.add_argument("--dataset_key", type=str, default="small_corner_bracket_physics", help="Optional dataset key from data_set.py (e.g. 'forklift'). This overrides --usd_path and its config scale/orientation.")
parser.add_argument("--num_points", type=int, default=10_000, help="Number of points to sample.")
parser.add_argument("--output", type=str, default="data/point_clouds/dataset/small_corner_bracket_physics.ply", help="Output PLY file path.")
parser.add_argument("--scale", type=float, nargs=3, default=[10.0, 10.0, 10.0], help="Scale of the object (x, y, z).")
parser.add_argument("--pos", type=float, nargs=3, default=[0.0, -2.0, 0.3], help="Position of the object (x, y, z).")
parser.add_argument("--center", action="store_true", help="Center the point cloud at (0,0,0) by subtracting the centroid.")

# Append AppLauncher arguments
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Force headless mode since we are not passing it as a CLI argument
args.headless = True

# Launch the app
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os
import torch
import numpy as np
import trimesh
from pxr import Usd, UsdGeom, Gf
import isaacsim.core.utils.stage as stage_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from scipy.spatial.transform import Rotation as R

import sys
# Allow importing of data_set if running this script strictly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")))
usd_data_set = {}
usd_data_set_old = {}
try:
    from isaaclab_tasks.direct.robot_inspection.configs.data_set import usd_data_set
except ImportError:
    print("[WARN] Could not import usd_data_set")
try:
    from isaaclab_tasks.direct.robot_inspection.configs.data_set import usd_data_set_old
except ImportError:
    pass

def triangulate_mesh(counts, indices):
    """
    Triangulate a polygon mesh (fan triangulation).
    
    Args:
        counts (np.ndarray): Array of vertex counts per face.
        indices (np.ndarray): Array of vertex indices.
        
    Returns:
        np.ndarray: (N, 3) array of triangle indices.
    """
    triangles = []
    idx = 0
    for count in counts:
        # Fan triangulation: (0, 1, 2), (0, 2, 3), ...
        base_idx = idx
        for i in range(1, count - 1):
            triangles.append([indices[base_idx], indices[base_idx + i], indices[base_idx + i + 1]])
        idx += count
    return np.array(triangles)

def get_mesh_from_prim(mesh_prim: UsdGeom.Mesh):
    """
    Extract trimesh object from UsdGeom.Mesh.
    """
    # Get mesh attributes
    points = np.array(mesh_prim.GetPointsAttr().Get())
    face_vertex_counts = np.array(mesh_prim.GetFaceVertexCountsAttr().Get())
    face_vertex_indices = np.array(mesh_prim.GetFaceVertexIndicesAttr().Get())
    
    if len(points) == 0:
        return None
        
    # Triangulate
    faces = triangulate_mesh(face_vertex_counts, face_vertex_indices)
    
    if len(faces) == 0:
        return None
        
    # Create Trimesh
    mesh = trimesh.Trimesh(vertices=points, faces=faces)
    return mesh

def main():
    # Load dataset config if dataset_key is used
    orientation_quat = None
    if getattr(args, "dataset_key", None):
        combined_dataset = {**usd_data_set_old, **usd_data_set}
        if args.dataset_key not in combined_dataset:
            raise ValueError(f"[ERROR] Dataset key '{args.dataset_key}' not found in data_set.py")
        
        target_cfg = combined_dataset[args.dataset_key]
        args.usd_path = target_cfg.get("usd_path", args.usd_path)
        print(f"[INFO] Using dataset key: {args.dataset_key}")
        
        if "scale" in target_cfg:
            s = target_cfg["scale"]
            if isinstance(s, (list, tuple)):
                args.scale = list(s)
            else:
                args.scale = [s, s, s]
            print(f"[INFO] Loaded scale {args.scale} from dataset config.")
            
        if "orientation" in target_cfg:
            orientation_quat = target_cfg["orientation"] # (w, x, y, z)
            print(f"[INFO] Loaded orientation {orientation_quat} from dataset config.")

    # Default path if none provided
    usd_path = args.usd_path
    if not usd_path:
        usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/Rubiks_Cube/rubiks_cube.usd"
        print(f"[INFO] No USD path provided. Using default Rubik's Cube: {usd_path}")

    # Check existence if local
    if not usd_path.startswith("http") and not usd_path.startswith("omniverse://"):
         if not os.path.exists(usd_path) and not os.path.exists(usd_path.replace("file:", "")):
             pass # Let Isaac Sim handle error if it fails

    # Open stage
    stage_utils.create_new_stage()
    stage_utils.open_stage(usd_path)
    
    stage = stage_utils.get_current_stage()
    if not stage:
        raise RuntimeError("Failed to obtain stage.")

    print(f"[INFO] Stage loaded. Traversing for meshes...")
    
    all_meshes = []
    
    # Traverse all prims
    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        if prim.IsA(UsdGeom.Mesh):
            mesh_prim = UsdGeom.Mesh(prim)
            
            # Check visibility
            vis = mesh_prim.GetVisibilityAttr().Get()
            if vis == 'invisible':
                continue
            
            tm = get_mesh_from_prim(mesh_prim)
            if tm:
                # Apply World Transform (from USD structure)
                xform = UsdGeom.Xformable(prim)
                world_transform = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                
                # Convert Gf.Matrix4d to numpy (row-major in USD, compatible with trimesh)
                matrix = np.array(world_transform).T
                
                tm.apply_transform(matrix)
                all_meshes.append(tm)

    if not all_meshes:
        print("[ERROR] No valid meshes found in the USD file.")
        simulation_app.close()
        return

    print(f"[INFO] Found {len(all_meshes)} meshes. Combining...")
    
    # Combine all meshes
    if len(all_meshes) > 1:
        combined_mesh = trimesh.util.concatenate(all_meshes)
    else:
        combined_mesh = all_meshes[0]
        
    # --- Apply USER defined Scale and Position ---
    # User args: scale (x,y,z), pos (x,y,z)
    
    # Apply Scale
    scale_matrix = np.eye(4)
    scale_matrix[0, 0] = args.scale[0]
    scale_matrix[1, 1] = args.scale[1]
    scale_matrix[2, 2] = args.scale[2]
    combined_mesh.apply_transform(scale_matrix)
    print(f"[INFO] Applied Scale: {args.scale}")
    
    # Apply Orientation from dataset if available
    if orientation_quat is not None:
        w, x, y, z = orientation_quat
        # scipy Rotation expects [x, y, z, w]
        rot = R.from_quat([x, y, z, w])
        orientation_matrix = np.eye(4)
        orientation_matrix[:3, :3] = rot.as_matrix()
        combined_mesh.apply_transform(orientation_matrix)
        print(f"[INFO] Applied Orientation from dataset config")
    
    # Apply Position (Translation)
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = args.pos
    combined_mesh.apply_transform(translation_matrix)
    print(f"[INFO] Applied Position: {args.pos}")

    
    print(f"[INFO] Combined and Transformed Mesh Area: {combined_mesh.area}")
    print(f"[INFO] Sampling {args.num_points} points using Ray Casting (Virtual Scan)...")
    
    # --- Ray Casting Logic ---
    # 1. Get bounds and center
    bounds = combined_mesh.bounds
    center = combined_mesh.centroid
    extents = combined_mesh.extents
    radius = np.max(extents) * 1.5
    
    # 2. Generate rays
    # Sample points on a sphere around the object
    num_rays = args.num_points * 2 
    
    # Fibonacci sphere sampling for even distribution
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle
    i = np.arange(num_rays)
    y = 1 - (i / float(num_rays - 1)) * 2  # y goes from 1 to -1
    radius_at_y = np.sqrt(1 - y * y) # radius at y
    theta = phi * i 
    
    x = np.cos(theta) * radius_at_y
    z = np.sin(theta) * radius_at_y
    
    sphere_points = np.stack((x, y, z), axis=1) * radius + center
    
    # Directions towards center
    ray_origins = sphere_points
    ray_directions = center - ray_origins
    
    # Normalize
    ray_directions = ray_directions / np.linalg.norm(ray_directions, axis=1)[:, np.newaxis]
    
    # 3. Cast rays
    # Use trimesh ray intersector
    locations, index_ray, index_tri = combined_mesh.ray.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        multiple_hits=False # only first hit (exterior)
    )
    
    print(f"[INFO] Ray casting hit {len(locations)} points on the exterior.")
    
    # If we have too many, downsample
    if len(locations) > args.num_points:
        indices = np.random.choice(len(locations), args.num_points, replace=False)
        final_points = locations[indices]
    else:
        final_points = locations

    # --- Optional Centering ---
    if args.center:
        centroid = np.mean(final_points, axis=0)
        final_points -= centroid
        print(f"[INFO] Centered point cloud. Subtracted centroid: {centroid}")

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Save to ply
    pcd = trimesh.points.PointCloud(final_points)
    pcd.export(args.output)
    
    print(f"[INFO] Point cloud saved to: {os.path.abspath(args.output)}")
    
    simulation_app.close()

if __name__ == "__main__":
    main()
