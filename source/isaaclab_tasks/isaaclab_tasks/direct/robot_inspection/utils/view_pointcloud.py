"""
Simple script to visualize a PLY point cloud file.
Usage: python scripts/utils/view_pointcloud.py --file path/to/file.ply
"""

import argparse
import os
import sys

def view_with_open3d(filepath):
    import open3d as o3d
    print(f"Loading {filepath} with Open3D...")
    pcd = o3d.io.read_point_cloud(filepath)
    o3d.visualization.draw_geometries([pcd], window_name=f"Point Cloud: {os.path.basename(filepath)}")

def view_with_trimesh(filepath):
    import trimesh
    print(f"Loading {filepath} with Trimesh...")
    pcd = trimesh.load(filepath)
    pcd.show()
#path_ = "/home/tosin/Documents/GitHub/IsaacLab/data/point_clouds/flange_pc.ply"
#path_ = "/home/tosin/Documents/GitHub/IsaacLab/data/point_clouds/dataset/small_corner_bracket_physics.ply"
#path_ = "/home/tosin/Documents/GitHub/IsaacLab/data/point_clouds/eval/small_corner_bracket_physics.ply"
#path_ = "/home/tosin/Documents/GitHub/IsaacLab/data/point_clouds/compare/small_corner_bracket_physics.ply"
path_ = "/home/tosin/Documents/GitHub/IsaacLab/data/recorded_depth_data_eval/eval_results/reconstructed_env0_ep15522.ply"
path_ = "/home/tosin/Documents/GitHub/IsaacLab/data/point_clouds/compare/small_corner_bracket_physics.ply"
path_ = "/home/tosin/Documents/GitHub/IsaacLab/data/recorded_depth_data_eval/eval_results/ur10_mount/reconstructed_env0_ep14994.ply"
#path_ = "data/point_clouds/comparison_vis.ply"
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View a PLY point cloud file.")
    # parser.add_argument("file", nargs="?", default="data/point_clouds/reconstructed_object.ply", help="Path to the PLY file.")
    parser.add_argument("file", nargs="?", default=path_, help="Path to the PLY file.")

    # parser.add_argument("file", nargs="?", default="data/point_clouds/inspection_object_baseline.ply", help="Path to the PLY file.")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "open3d", "trimesh"], help="Visualization backend.")
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"Error: File not found: {args.file}")
        sys.exit(1)

    if args.backend == "auto":
        try:
            import open3d
            view_with_open3d(args.file)
        except ImportError:
            try:
                import trimesh
                view_with_trimesh(args.file)
            except ImportError:
                print("Error: Neither Open3D nor Trimesh is installed. Please install one: pip install open3d trimesh")
                sys.exit(1)
    elif args.backend == "open3d":
        view_with_open3d(args.file)
    elif args.backend == "trimesh":
        view_with_trimesh(args.file)
