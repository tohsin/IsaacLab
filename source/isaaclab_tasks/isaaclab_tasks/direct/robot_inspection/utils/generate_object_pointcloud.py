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
parser.add_argument(
    "--ray_batch_size",
    type=int,
    default=1024,
    help="Maximum rays per Trimesh query. Smaller values reduce peak RAM usage.",
)
parser.add_argument(
    "--ray_backend",
    choices=("open3d", "trimesh"),
    default="open3d",
    help="Raycasting backend. Open3D uses a bounded-memory BVH and is the default.",
)
parser.add_argument(
    "--num_views",
    type=int,
    default=128,
    help="Number of surrounding orthographic viewpoints used to expose the full exterior.",
)
parser.add_argument(
    "--ray_oversample_factor",
    type=float,
    default=8.0,
    help="Approximate rays cast per requested output point.",
)
parser.add_argument("--output", type=str, default="data/point_clouds/dataset/small_corner_bracket_physics.ply", help="Output PLY file path.")
parser.add_argument("--scale", type=float, nargs=3, default=[10.0, 10.0, 10.0], help="Scale of the object (x, y, z).")
parser.add_argument(
    "--pos",
    type=float,
    nargs=3,
    default=None,
    help="Object position. Dataset root_height is used when this is omitted.",
)
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
    target_cfg = None
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

    if args.pos is None:
        root_height = target_cfg.get("root_height", 0.3) if target_cfg else 0.3
        args.pos = [0.0, -2.0, float(root_height)]
        print(f"[INFO] Using configured root height: {root_height}")

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

    print(
        f"[INFO] Combined topology: {len(combined_mesh.vertices)} vertices, "
        f"{len(combined_mesh.faces)} triangles"
    )
        
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
    object_radius = 0.5 * np.linalg.norm(extents)
    camera_radius = object_radius * 2.5
    
    # 2. Generate rays
    # Distribute orthographic virtual cameras over a Fibonacci sphere. Each
    # camera casts a 2D grid that spans the object's complete bounding sphere.
    # Unlike one center-directed ray per viewpoint, this also reaches offset
    # components and surfaces visible around the sides of foreground geometry.
    if args.num_views <= 0:
        raise ValueError("--num_views must be greater than zero")
    if args.ray_oversample_factor <= 0:
        raise ValueError("--ray_oversample_factor must be greater than zero")

    requested_num_rays = max(
        args.num_views, int(np.ceil(args.num_points * args.ray_oversample_factor))
    )
    grid_side = int(np.ceil(np.sqrt(requested_num_rays / args.num_views)))

    # Fibonacci sphere sampling for even view-direction coverage.
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle
    i = np.arange(args.num_views)
    y = 1 - ((i + 0.5) / args.num_views) * 2
    radius_at_y = np.sqrt(1 - y * y) # radius at y
    theta = phi * i 
    x = np.cos(theta) * radius_at_y
    z = np.sin(theta) * radius_at_y
    view_directions = np.stack((x, y, z), axis=1)

    grid_coords = np.linspace(-object_radius * 1.05, object_radius * 1.05, grid_side)
    grid_u, grid_v = np.meshgrid(grid_coords, grid_coords, indexing="xy")
    grid_u = grid_u.reshape(-1, 1)
    grid_v = grid_v.reshape(-1, 1)

    ray_origin_batches = []
    ray_direction_batches = []
    for view_direction in view_directions:
        reference_axis = (
            np.array([0.0, 1.0, 0.0])
            if abs(view_direction[2]) > 0.9
            else np.array([0.0, 0.0, 1.0])
        )
        image_u = np.cross(view_direction, reference_axis)
        image_u /= np.linalg.norm(image_u)
        image_v = np.cross(view_direction, image_u)
        image_v /= np.linalg.norm(image_v)

        view_center = center + view_direction * camera_radius
        origins = view_center + grid_u * image_u + grid_v * image_v
        directions = np.broadcast_to(-view_direction, origins.shape).copy()
        ray_origin_batches.append(origins)
        ray_direction_batches.append(directions)

    ray_origins = np.concatenate(ray_origin_batches, axis=0)
    ray_directions = np.concatenate(ray_direction_batches, axis=0)
    num_rays = len(ray_origins)
    print(
        f"[INFO] Virtual scan pattern: {args.num_views} views, "
        f"{grid_side}x{grid_side} rays/view ({num_rays} total rays)."
    )
    
    # 3. Cast rays
    # Use trimesh ray intersector
    if args.ray_batch_size <= 0:
        raise ValueError("--ray_batch_size must be greater than zero")

    # Trimesh's pure-Python ray backend can create very large temporary arrays
    # when all rays are queried at once. Process bounded batches so requesting a
    # dense reference cloud does not exhaust system RAM (and kill the editor
    # process that launched Isaac Sim).
    location_batches = []
    num_batches = (num_rays + args.ray_batch_size - 1) // args.ray_batch_size
    progress_interval = max(1, num_batches // 10)

    raycasting_scene = None
    if args.ray_backend == "open3d":
        try:
            import open3d as o3d
        except ImportError as exc:
            raise RuntimeError(
                "Open3D is required for the bounded-memory ray backend. "
                "Run this script through ./isaaclab.sh -p."
            ) from exc

        legacy_mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.asarray(combined_mesh.vertices)),
            o3d.utility.Vector3iVector(np.asarray(combined_mesh.faces)),
        )
        tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(legacy_mesh)
        raycasting_scene = o3d.t.geometry.RaycastingScene()
        raycasting_scene.add_triangles(tensor_mesh)
        print("[INFO] Built Open3D raycasting scene.")

    for batch_idx, start in enumerate(range(0, num_rays, args.ray_batch_size)):
        end = min(start + args.ray_batch_size, num_rays)
        batch_origins = ray_origins[start:end]
        batch_directions = ray_directions[start:end]
        if args.ray_backend == "open3d":
            rays_np = np.concatenate((batch_origins, batch_directions), axis=1).astype(
                np.float32, copy=False
            )
            cast_result = raycasting_scene.cast_rays(o3d.core.Tensor(rays_np))
            hit_distances = cast_result["t_hit"].numpy()
            hit_mask = np.isfinite(hit_distances)
            batch_locations = (
                batch_origins[hit_mask]
                + batch_directions[hit_mask] * hit_distances[hit_mask, np.newaxis]
            )
        else:
            batch_locations, _, _ = combined_mesh.ray.intersects_location(
                ray_origins=batch_origins,
                ray_directions=batch_directions,
                multiple_hits=False,  # only first hit (exterior)
            )
        if len(batch_locations) > 0:
            location_batches.append(batch_locations)
        if (batch_idx + 1) % progress_interval == 0 or end == num_rays:
            print(f"[INFO] Ray-casting batch {batch_idx + 1}/{num_batches}")

    if location_batches:
        locations = np.concatenate(location_batches, axis=0)
    else:
        locations = np.empty((0, 3), dtype=np.float64)
    
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
