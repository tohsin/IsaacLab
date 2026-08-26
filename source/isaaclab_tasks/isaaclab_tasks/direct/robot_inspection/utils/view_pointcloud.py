"""
Simple script to visualize a PLY point cloud file.
Usage: python scripts/utils/view_pointcloud.py --file path/to/file.ply
"""

import argparse
import os
import sys

def view_with_open3d(filepaths):
    import open3d as o3d
    point_clouds = []
    for filepath in filepaths:
        print(f"Loading {filepath} with Open3D...")
        pcd = o3d.io.read_point_cloud(filepath)
        if "reconstruction" in os.path.basename(filepath).lower():
            pcd.paint_uniform_color([0.1, 0.35, 1.0])
        point_clouds.append(pcd)
    if len(point_clouds) > 1:
        print("Colors: reconstruction=blue, GT covered=green, GT missed=red")
    o3d.visualization.draw_geometries(point_clouds, window_name="Point Cloud Comparison")

def view_with_trimesh(filepaths):
    import trimesh
    scene = trimesh.Scene()
    for filepath in filepaths:
        print(f"Loading {filepath} with Trimesh...")
        pcd = trimesh.load(filepath)
        scene.add_geometry(pcd, node_name=os.path.basename(filepath))
    scene.show()
path_ = "/home/tosin/Documents/GitHub/IsaacLab/data/point_clouds/dataset/ur10_mount.ply"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View a PLY point cloud file.")
    # parser.add_argument("file", nargs="?", default="data/point_clouds/reconstructed_object.ply", help="Path to the PLY file.")
    parser.add_argument("files", nargs="*", default=[path_], help="One or more PLY files.")

    # parser.add_argument("file", nargs="?", default="data/point_clouds/inspection_object_baseline.ply", help="Path to the PLY file.")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "open3d", "trimesh"], help="Visualization backend.")
    args = parser.parse_args()
    
    for filepath in args.files:
        if not os.path.exists(filepath):
            print(f"Error: File not found: {filepath}")
            sys.exit(1)

    if args.backend == "auto":
        try:
            import open3d
            view_with_open3d(args.files)
        except ImportError:
            try:
                import trimesh
                view_with_trimesh(args.files)
            except ImportError:
                print("Error: Neither Open3D nor Trimesh is installed. Please install one: pip install open3d trimesh")
                sys.exit(1)
    elif args.backend == "open3d":
        view_with_open3d(args.files)
    elif args.backend == "trimesh":
        view_with_trimesh(args.files)
