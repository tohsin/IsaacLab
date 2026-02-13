"""
Script to generate a point cloud from an inspection object in Isaac Sim.
"""

from __future__ import annotations

import argparse
from isaaclab.app import AppLauncher

# Create the parser
parser = argparse.ArgumentParser(description="Generate a point cloud from an inspection object.")
parser.add_argument("--usd_path", type=str, default=None, help="Path to the USD file.")
parser.add_argument("--num_points", type=int, default=10000, help="Number of points to sample.")
parser.add_argument("--output", type=str, default="inspection_object_baseline.ply", help="Output PLY file path.")
# parser.add_argument("--headless", action="store_true", default=True, help="Run in headless mode.") # AppLauncher handles this

# Append AppLauncher arguments
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

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
    # Default path if none provided
    usd_path = args.usd_path
    if not usd_path:
        # Fallback to Rubik's cube as seen in config
        # Note: ISAAC_NUCLEUS_DIR might be an environment variable or a placeholder in some contexts
        # but in isaaclab.utils.assets it should be resolved.
        usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/Rubiks_Cube/rubiks_cube.usd"
        print(f"[INFO] No USD path provided. Using default Rubik's Cube: {usd_path}")

    # Check if file exists if it's a local path, otherwise trust it's a nucleus path
    if not usd_path.startswith("http") and not usd_path.startswith("omniverse://"):
         if not os.path.exists(usd_path) and not os.path.exists(usd_path.replace("file:", "")):
             # Try to resolve relative to current dir?
             pass

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
            
            # Check visibility (simple check)
            vis = mesh_prim.GetVisibilityAttr().Get()
            if vis == 'invisible':
                continue
            
            tm = get_mesh_from_prim(mesh_prim)
            if tm:
                # Apply World Transform
                xform = UsdGeom.Xformable(prim)
                world_transform = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                
                # Convert Gf.Matrix4d to numpy
                matrix = np.array(world_transform).T # Transpose for standard layout if needed? 
                # Gf matrix is row-major? No, Gf is row-major, numpy usually expects standard.
                # Trimesh expects 4x4.
                # Let's check Gf.Matrix4d structure.
                # Standard USD matrices are row-major?
                # Actually, trimesh.apply_transform expects (4,4) 
                
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
    
    print(f"[INFO] Combined Mesh Area: {combined_mesh.area}")
    print(f"[INFO] Sampling {args.num_points} points...")
    
    points, _ = trimesh.sample.sample_surface(combined_mesh, args.num_points)
    
    # Save to ply
    pcd = trimesh.points.PointCloud(points)
    pcd.export(args.output)
    
    print(f"[INFO] Point cloud saved to: {os.path.abspath(args.output)}")
    
    simulation_app.close()

if __name__ == "__main__":
    main()
