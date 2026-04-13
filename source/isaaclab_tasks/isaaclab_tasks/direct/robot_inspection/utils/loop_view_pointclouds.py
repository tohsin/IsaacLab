"""
Simple script to loop through and visualize multiple PLY point cloud files in a directory.
Usage: python scripts/utils/loop_view_pointclouds.py --dir path/to/eval_results
"""

import argparse
import os
import sys
import glob

def view_with_open3d(filepath):
    import open3d as o3d
    print(f"Loading {filepath} with Open3D...")
    pcd = o3d.io.read_point_cloud(filepath)
    print(f"Loaded point cloud with {len(pcd.points)} points. Close the window to see the next one.")
    o3d.visualization.draw_geometries([pcd], window_name=f"Point Cloud: {os.path.basename(filepath)}")

def view_with_trimesh(filepath):
    import trimesh
    print(f"Loading {filepath} with Trimesh...")
    pcd = trimesh.load(filepath)
    print(f"Loaded point cloud with {len(pcd.vertices)} vertices. Close the window to see the next one.")
    pcd.show()

default_dir = "/home/tosin/Documents/GitHub/IsaacLab/data/recorded_depth_data_eval/eval_results/bracket-SEEIR"
default_dir = "/home/tosin/Documents/GitHub/IsaacLab/data/recorded_depth_data_eval/eval_results/ur10_mount"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View multiple PLY point cloud files in a directory sequentially.")
    parser.add_argument("--dir", type=str, default=default_dir, help="Path to the directory containing .ply files.")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "open3d", "trimesh"], help="Visualization backend.")
    args = parser.parse_args()
    
    if not os.path.exists(args.dir):
        print(f"Error: Directory not found: {args.dir}")
        sys.exit(1)

    ply_files = sorted(glob.glob(os.path.join(args.dir, "*.ply")))
    
    if not ply_files:
        print(f"No .ply files found in {args.dir}")
        sys.exit(0)

    print(f"Found {len(ply_files)} .ply files. Visualizing them one by one...")

    backend = args.backend
    if backend == "auto":
        try:
            import open3d
            backend = "open3d"
        except ImportError:
            try:
                import trimesh
                backend = "trimesh"
            except ImportError:
                print("Error: Neither Open3D nor Trimesh is installed. Please install one: pip install open3d trimesh")
                sys.exit(1)

    for i, filepath in enumerate(ply_files):
        print(f"\n--- [{i+1}/{len(ply_files)}] Visualizing {os.path.basename(filepath)} ---")
        if backend == "open3d":
            view_with_open3d(filepath)
        elif backend == "trimesh":
            view_with_trimesh(filepath)
            
    print("\nFinished iterating over all point clouds!")
