"""
Script to compare two point clouds using Chamfer Distance.
Usage:
    python compare_pointclouds.py --source <path_to_ply1> --target <path_to_ply2>
python compare_pointclouds.py --source data/point_clouds/inspection_object_baseline.ply --target data/point_clouds/reconstructed_object.ply
Computes:
    - Chamfer Distance (mean closest point distance)
    - Hausdorff Distance (max closest point distance)
"""

import argparse
import numpy as np
import trimesh
from scipy.spatial import cKDTree

def compute_metrics(source_points, target_points):
    """
    Computes Chamfer Distance and other metrics between two point sets.
    """
    print(f"[INFO] Building KD-Trees...")
    # Tree for Target
    tree_target = cKDTree(target_points)
    # Tree for Source
    tree_source = cKDTree(source_points)
    
    print(f"[INFO] Querying Source -> Target...")
    dist_s2t, _ = tree_target.query(source_points, k=1, workers=-1)
    
    print(f"[INFO] Querying Target -> Source...")
    dist_t2s, _ = tree_source.query(target_points, k=1, workers=-1)
    
    # Chamfer Distance (Mean)
    chamfer_dist = np.mean(dist_s2t) + np.mean(dist_t2s)
    
    # Hausdorff Distance (Max)
    hausdorff_dist = max(np.max(dist_s2t), np.max(dist_t2s))
    
    metrics = {
        "Chamfer Distance (Mean)": chamfer_dist,
        "Hausdorff Distance": hausdorff_dist,
        "Source -> Target Mean": np.mean(dist_s2t),
        "Source -> Target Max": np.max(dist_s2t),
        "Target -> Source Mean": np.mean(dist_t2s),
        "Target -> Source Max": np.max(dist_t2s)
    }
    
    # Calculate Coverage at thresholds
    # Coverage = % of Source (GT) points that have a neighbor in Target (Recon) within threshold
    thresholds = [0.01, 0.02, 0.05, 0.1, 0.2]
    for thresh in thresholds:
        covered_count = np.sum(dist_s2t < thresh)
        coverage_pct = (covered_count / len(source_points)) * 100
        metrics[f"Coverage @ {thresh:.2f}"] = coverage_pct
        
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Compare two point clouds.")
    parser.add_argument("--source", type=str, default="data/point_clouds/inspection_object_baseline.ply", help="Path to source PLY (e.g. reconstructed).")
    parser.add_argument("--target", type=str, default="data/point_clouds/reconstructed_object.ply", help="Path to target PLY (e.g. baseline/GT).")
    parser.add_argument("--visualize", action="store_true", help="Visualize error (requires Open3D/matplotlib - simpler just prints).")
    
    args = parser.parse_args()
    
    print(f"[INFO] Loading Source: {args.source}")
    pcd_source = trimesh.load(args.source)  
    if isinstance(pcd_source, trimesh.Scene):
        # Flatten if scene
        geometries = list(pcd_source.geometry.values())
        if not geometries:
             print("[ERROR] Source PLY contains no geometry.")
             return
        points_source = np.vstack([g.vertices for g in geometries])
    else:
        points_source = pcd_source.vertices
        
    print(f"[INFO] Loading Target: {args.target}")
    pcd_target = trimesh.load(args.target)
    if isinstance(pcd_target, trimesh.Scene):
        geometries = list(pcd_target.geometry.values())
        if not geometries:
             print("[ERROR] Target PLY contains no geometry.")
             return
        points_target = np.vstack([g.vertices for g in geometries])
    else:
        points_target = pcd_target.vertices
        
    print(f"[INFO] Source Points: {points_source.shape[0]}")
    print(f"[INFO] Target Points: {points_target.shape[0]}")
    
    metrics = compute_metrics(points_source, points_target)
    
    print("\n" + "="*30)
    print("COMPARISON RESULTS")
    print("="*30)
    for k, v in metrics.items():
        print(f"{k:<25}: {v:.6f}")
    print("="*30 + "\n")
    
    # Optional Visualization Saving
    # Create a colored version of the Source (GT) where:
    # Green = Covered (< 0.05)
    # Red = Missed (> 0.05)
    
    print(f"[INFO] Computing visualization cloud...")
    tree_target = cKDTree(points_target)
    dist_s2t, _ = tree_target.query(points_source, k=1, workers=-1)
    
    colors = np.zeros((points_source.shape[0], 4), dtype=np.uint8)
    
    # Standard Green for Covered, Red for Missed
    # Distance mapped to color intensity
    # Threshold at 0.05 (5cm)
    mask_covered = dist_s2t < 0.05
    
    # Green (0, 255, 0, 255)
    colors[mask_covered] = [0, 255, 0, 255]
    
    # Red (255, 0, 0, 255)
    colors[~mask_covered] = [255, 0, 0, 255]
    
    pcd_vis = trimesh.points.PointCloud(points_source, colors=colors)
    vis_path = "data/point_clouds/comparison_vis.ply"
    pcd_vis.export(vis_path)
    print(f"[INFO] Saved visualization to: {vis_path}")
    print(f"       Use view_pointcloud.py to see Green (Covered) vs Red (Missed).")

if __name__ == "__main__":
    main()
