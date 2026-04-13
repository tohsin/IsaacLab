"""
Script to compare multiple point clouds against a single reference point cloud using Chamfer Distance and Coverage metrics.
Outputs a summary table with Mean, Max, Min, and Std of the evaluated point clouds.

Usage:
    python compare_multiple_pointclouds.py --source_dir <dir_with_plys> --target <path_to_reference_ply>
OR
    python compare_multiple_pointclouds.py --sources <ply1> <ply2> ... --target <path_to_reference_ply>
"""

import argparse
import numpy as np
import trimesh
from scipy.spatial import cKDTree
import os
import glob

def compute_metrics(source_points, target_points):
    """
    Computes Chamfer Distance and other metrics between two point sets.
    """
    # Tree for Target
    tree_target = cKDTree(target_points)
    # Tree for Source
    tree_source = cKDTree(source_points)
    
    dist_s2t, _ = tree_target.query(source_points, k=1, workers=-1)
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
    thresholds = [0.01, 0.02, 0.05, 0.1, 0.2]
    for thresh in thresholds:
        covered_count = np.sum(dist_s2t < thresh)
        coverage_pct = (covered_count / len(source_points)) * 100
        metrics[f"Coverage @ {thresh:.2f}"] = coverage_pct
        
    # Calculate AUC (Area Under Curve) of Coverage up to 0.2m
    max_thresh_auc = 0.2
    dense_thresholds = np.linspace(0.0, max_thresh_auc, 100)
    coverages = [np.sum(dist_s2t < d) / len(source_points) for d in dense_thresholds]
    
    auc_area = np.trapz(coverages, dense_thresholds)
    auc_normalized = auc_area / max_thresh_auc
    
    metrics[f"Coverage AUC (up to {max_thresh_auc}m)"] = auc_normalized
        
    return metrics
source_dir_bracket = "data/recorded_depth_data_eval/eval_results/bracket-SEEIR"
source_dir = source_dir_bracket
target_dir_bracket = "data/point_clouds/Ground-Truth/small_corner_bracket_physics.ply"
target_GT = target_dir_bracket
def main():
    parser = argparse.ArgumentParser(description="Compare multiple point clouds against a reference.")
    parser.add_argument("--source_dir", type=str, default=source_dir, help="Directory containing source PLYs (e.g. reconstructed).")
    parser.add_argument("--sources", type=str, nargs='+', help="List of source PLY files.")
    parser.add_argument("--target", type=str, default=target_GT, help="Path to reference target PLY (e.g. baseline/GT).")
    parser.add_argument("--icp", action=argparse.BooleanOptionalAction, default=True, help="Run ICP alignment before comparison (default: True). Use --no-icp to disable.")
    parser.add_argument("--outlier_removal", action=argparse.BooleanOptionalAction, default=True, help="Run Statistical Outlier Removal before evaluation.")
    
    args = parser.parse_args()
    
    source_files = []
    if args.sources:
        source_files.extend(args.sources)
    if args.source_dir:
        source_files.extend(glob.glob(os.path.join(args.source_dir, "*.ply")))
        
    if not source_files:
        print("[ERROR] No source files found or provided.")
        return
        
    print(f"[INFO] Found {len(source_files)} source point clouds to evaluate.")
    print(f"[INFO] Outlier Removal: {'Enabled' if args.outlier_removal else 'Disabled'}")
    
    print(f"[INFO] Loading Reference Target: {args.target}")
    pcd_target = trimesh.load(args.target)
    if isinstance(pcd_target, trimesh.Scene):
        geometries = list(pcd_target.geometry.values())
        if not geometries:
             print("[ERROR] Target PLY contains no geometry.")
             return
        points_target = np.vstack([g.vertices for g in geometries])
    else:
        points_target = pcd_target.vertices
        
    print(f"[INFO] Target Points: {points_target.shape[0]}")
    
    all_metrics = []
    
    for idx, path in enumerate(source_files):
        print(f"\n[INFO] --- Evaluating {idx+1}/{len(source_files)}: {os.path.basename(path)} ---")
        try:
            pcd_source = trimesh.load(path)
            if isinstance(pcd_source, trimesh.Scene):
                geometries = list(pcd_source.geometry.values())
                if not geometries:
                     print(f"[WARN] Empty geometry in {path}")
                     continue
                points_source = np.vstack([g.vertices for g in geometries])
            else:
                points_source = pcd_source.vertices
                
            if len(points_source) == 0:
                print(f"[WARN] No points in {path}")
                continue
                
            if args.outlier_removal:
                print("[INFO] Running Statistical Outlier Removal...")
                try:
                    import open3d as o3d
                    source_o3d = o3d.geometry.PointCloud()
                    source_o3d.points = o3d.utility.Vector3dVector(points_source)
                    # nb_neighbors=20, std_ratio=2.0 is a standard conservative filter
                    # Using std_ratio=1.0 for slightly more aggressive filtering
                    cl, ind = source_o3d.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.0)
                    filtered_points = np.asarray(cl.points)
                    print(f"[INFO] Filtered from {len(points_source)} to {len(filtered_points)} points.")
                    points_source = filtered_points
                except ImportError:
                    print("[WARN] Open3D not found, skipping outlier removal.")

            if args.icp:
                print("[INFO] Running ICP Alignment...")
                try:
                    import open3d as o3d
                    source_o3d = o3d.geometry.PointCloud()
                    source_o3d.points = o3d.utility.Vector3dVector(points_source)
                    target_o3d = o3d.geometry.PointCloud()
                    target_o3d.points = o3d.utility.Vector3dVector(points_target)
                    
                    # Initial alignment using centroids
                    source_center = np.mean(points_source, axis=0)
                    target_center = np.mean(points_target, axis=0)
                    source_o3d.translate(target_center - source_center)
                    
                    # ICP
                    threshold = 0.5  # 50cm search radius
                    trans_init = np.eye(4)
                    reg_p2p = o3d.pipelines.registration.registration_icp(
                        source_o3d, target_o3d, threshold, trans_init,
                        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
                    )
                    source_o3d.transform(reg_p2p.transformation)
                    points_source = np.asarray(source_o3d.points)
                except ImportError:
                    print("[WARN] Open3D not found, skipping ICP.")
            
            metrics = compute_metrics(points_source, points_target)
            metrics["File"] = os.path.basename(path)
            all_metrics.append(metrics)
            
        except Exception as e:
            print(f"[ERROR] Failed measuring {path}: {e}")
            
    if not all_metrics:
        print("[ERROR] No valid metrics computed.")
        return
        
    # Aggregate Metrics
    agg_metrics = {}
    metric_keys = [k for k in all_metrics[0].keys() if k != "File"]
    
    for k in metric_keys:
        values = [m[k] for m in all_metrics]
        agg_metrics[k] = {
            "Mean": np.mean(values),
            "Max": np.max(values),
            "Min": np.min(values),
            "Std": np.std(values)
        }
        
    print("\n" + "="*84)
    print(f"{'BATCH COMPARISON RESULTS':^84}")
    print(f"{'(Aggregated over ' + str(len(all_metrics)) + ' files)':^84}")
    print("="*84)
    print(f"{'Metric':<35} | {'Mean':<10} | {'Max':<10} | {'Min':<10} | {'Std':<10}")
    print("-" * 84)
    
    for k in metric_keys:
        print(f"{k:<35} | {agg_metrics[k]['Mean']:<10.5f} | {agg_metrics[k]['Max']:<10.5f} | {agg_metrics[k]['Min']:<10.5f} | {agg_metrics[k]['Std']:<10.5f}")
        
    print("="*84 + "\n")

if __name__ == "__main__":
    main()
