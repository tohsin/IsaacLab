"""Evaluate every recorded reconstruction episode against a ground-truth cloud.

The reconstruction is the metric source and the ground-truth cloud is the
metric target. Coverage/recall is therefore the fraction of ground-truth
points within a threshold of any reconstructed point.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import cKDTree


DEFAULT_THRESHOLDS = (0.01, 0.02, 0.05)


def load_point_cloud(path: Path) -> np.ndarray:
    geometry = trimesh.load(path, process=False)
    if isinstance(geometry, trimesh.Scene):
        geometries = list(geometry.geometry.values())
        if not geometries:
            raise ValueError(f"Point cloud contains no geometry: {path}")
        points = np.vstack([np.asarray(item.vertices) for item in geometries])
    else:
        points = np.asarray(geometry.vertices)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError(f"Expected a non-empty Nx3 point cloud: {path}")
    return points.astype(np.float64, copy=False)


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if voxel_size <= 0 or len(points) == 0:
        return points
    voxel_ids = np.floor(points / voxel_size).astype(np.int64)
    _, unique_indices = np.unique(voxel_ids, axis=0, return_index=True)
    return points[np.sort(unique_indices)]


def load_frame_points(episode_dir: Path, metadata: dict, frame: dict) -> np.ndarray:
    depth_path = episode_dir / frame.get("depth_file_path", frame["file_path"])
    depth = np.load(depth_path, allow_pickle=False)
    if depth.ndim == 3:
        depth = depth.squeeze(-1)

    valid = (depth > 0) & np.isfinite(depth)
    mask_path = frame.get("mask_path")
    if mask_path:
        semantic_mask = np.asarray(Image.open(episode_dir / mask_path)) > 0
        valid &= semantic_mask
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float64)

    pixel_v, pixel_u = np.nonzero(valid)
    z = depth[valid].astype(np.float64, copy=False)
    fx = float(frame.get("fl_x", metadata["fl_x"]))
    fy = float(frame.get("fl_y", metadata["fl_y"]))
    cx = float(frame.get("cx", metadata["cx"]))
    cy = float(frame.get("cy", metadata["cy"]))

    x = (pixel_u - cx) * z / fx
    y = (pixel_v - cy) * z / fy
    points_camera = np.column_stack((x, -y, -z))

    camera_to_world = np.asarray(frame["transform_matrix"], dtype=np.float64)
    return points_camera @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]


def estimate_alignment(
    reconstruction: np.ndarray,
    ground_truth: np.ndarray,
    voxel_size: float,
    max_correspondence_distance: float,
) -> tuple[np.ndarray, dict]:
    """Estimate translation-only ICP for fixed-orientation evaluation targets."""
    source = voxel_downsample(reconstruction, voxel_size)
    target = voxel_downsample(ground_truth, voxel_size)
    target_tree = cKDTree(target)

    # Bounding-box centers are less biased than point centroids when sampling
    # densities differ between the reconstructed and ground-truth clouds.
    source_center = 0.5 * (np.min(source, axis=0) + np.max(source, axis=0))
    target_center = 0.5 * (np.min(target, axis=0) + np.max(target, axis=0))
    translation = target_center - source_center

    for _ in range(200):
        aligned = source + translation
        distances, indices = target_tree.query(aligned, k=1, workers=-1)
        inliers = distances < max_correspondence_distance
        if np.count_nonzero(inliers) < 100:
            # Preserve progress for unusually partial clouds by retaining the
            # closest half of correspondences rather than estimating rotation.
            cutoff = np.percentile(distances, 50)
            inliers = distances <= cutoff
        residuals = target[indices[inliers]] - aligned[inliers]
        delta = np.median(residuals, axis=0)
        translation += delta
        if np.linalg.norm(delta) < 1.0e-6:
            break

    aligned = source + translation
    distances, _ = target_tree.query(aligned, k=1, workers=-1)
    inliers = distances < max_correspondence_distance
    fitness = float(np.mean(inliers))
    inlier_rmse = (
        float(np.sqrt(np.mean(np.square(distances[inliers]))))
        if np.any(inliers)
        else math.inf
    )
    transform = np.eye(4)
    transform[:3, 3] = translation
    return transform, {"fitness": fitness, "inlier_rmse": inlier_rmse}


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def quaternion_matrix(quaternion_wxyz: list[float]) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion_wxyz, dtype=np.float64)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0:
        raise ValueError("Target orientation quaternion has zero length")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def compensate_target_motion(
    points: np.ndarray,
    frame: dict,
    reference_position: np.ndarray,
    reference_rotation: np.ndarray,
) -> np.ndarray:
    """Map world points through the current target pose into its first-frame pose."""
    current_position = np.asarray(frame["target_position"], dtype=np.float64)
    current_rotation = quaternion_matrix(frame["target_orientation"])
    points_target_local = (points - current_position) @ current_rotation
    return points_target_local @ reference_rotation.T + reference_position


def descriptive_stats(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        return {}
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "p05": float(np.percentile(array, 5)),
        "p95": float(np.percentile(array, 95)),
    }


def metric_name(prefix: str, threshold: float) -> str:
    return f"{prefix}_{int(round(threshold * 1000)):03d}mm_percent"


def evaluate_episode(
    episode_dir: Path,
    episode_result: dict,
    ground_truth: np.ndarray,
    ground_truth_tree: cKDTree,
    thresholds: tuple[float, ...],
    voxel_size: float,
    alignment_voxel_size: float,
    max_correspondence_distance: float,
    output_dir: Path,
) -> tuple[dict, list[dict]]:
    transforms_path = episode_dir / "transforms.json"
    metadata = json.loads(transforms_path.read_text(encoding="utf-8"))
    frames = metadata.get("frames", [])
    if not frames:
        raise ValueError("episode has no recorded frames")

    has_target_poses = all(
        frame.get("target_position") is not None
        and frame.get("target_orientation") is not None
        for frame in frames
    )
    reference_position = None
    reference_rotation = None
    if has_target_poses:
        reference_position = np.asarray(frames[0]["target_position"], dtype=np.float64)
        reference_rotation = quaternion_matrix(frames[0]["target_orientation"])

    raw_frame_points = []
    for frame in frames:
        points = load_frame_points(episode_dir, metadata, frame)
        if has_target_poses and len(points):
            points = compensate_target_motion(
                points, frame, reference_position, reference_rotation
            )
        raw_frame_points.append(voxel_downsample(points, voxel_size))
    nonempty = [points for points in raw_frame_points if len(points)]
    if not nonempty:
        raise ValueError("episode has no masked reconstruction points")

    final_unaligned = voxel_downsample(np.vstack(nonempty), voxel_size)
    alignment, alignment_quality = estimate_alignment(
        final_unaligned,
        ground_truth,
        alignment_voxel_size,
        max_correspondence_distance,
    )

    transformed_frames = [
        transform_points(points, alignment) if len(points) else points
        for points in raw_frame_points
    ]
    final_reconstruction = voxel_downsample(
        np.vstack([points for points in transformed_frames if len(points)]), voxel_size
    )

    episode_steps = int(episode_result["episode_steps"])
    duration_seconds = float(episode_result["duration_seconds"])
    running_gt_distance = np.full(len(ground_truth), np.inf, dtype=np.float64)
    time_rows = []
    for frame_index, (frame, frame_points) in enumerate(zip(frames, transformed_frames)):
        if len(frame_points):
            frame_tree = cKDTree(frame_points)
            frame_distance, _ = frame_tree.query(ground_truth, k=1, workers=-1)
            np.minimum(running_gt_distance, frame_distance, out=running_gt_distance)

        fraction = (frame_index + 1) / len(frames)
        episode_step = int(frame.get("episode_step", round(fraction * episode_steps)))
        if frame.get("elapsed_seconds") is not None:
            elapsed_seconds = float(frame["elapsed_seconds"])
        elif frame.get("episode_step") is not None and episode_steps > 0:
            elapsed_seconds = episode_step / episode_steps * duration_seconds
        else:
            elapsed_seconds = fraction * duration_seconds
        row = {
            "episode_index": int(episode_result["episode_index"]),
            "frame_index": frame_index,
            "episode_step": episode_step,
            "elapsed_seconds": elapsed_seconds,
            "episode_progress": fraction,
            "crashed": bool(episode_result["crashed"]),
        }
        for threshold in thresholds:
            row[metric_name("coverage", threshold)] = float(
                np.mean(running_gt_distance < threshold) * 100.0
            )
        time_rows.append(row)

    recon_to_gt, _ = ground_truth_tree.query(final_reconstruction, k=1, workers=-1)
    final_gt_to_recon = running_gt_distance
    chamfer = float(np.mean(recon_to_gt) + np.mean(final_gt_to_recon))
    hausdorff = float(max(np.max(recon_to_gt), np.max(final_gt_to_recon)))

    dense_thresholds = np.linspace(0.0, 0.2, 201)
    coverage_curve = np.asarray(
        [np.mean(final_gt_to_recon < threshold) for threshold in dense_thresholds]
    )
    coverage_distance_auc = float(np.trapz(coverage_curve, dense_thresholds) / 0.2)

    result = {
        **episode_result,
        "recorded_frames": len(frames),
        "reconstruction_points": int(len(final_reconstruction)),
        "alignment_fitness": alignment_quality["fitness"],
        "alignment_inlier_rmse": alignment_quality["inlier_rmse"],
        "target_pose_compensated": has_target_poses,
        "chamfer_distance": chamfer,
        "hausdorff_distance": hausdorff,
        "reconstruction_to_gt_mean": float(np.mean(recon_to_gt)),
        "gt_to_reconstruction_mean": float(np.mean(final_gt_to_recon)),
        "coverage_distance_auc_0_to_200mm": coverage_distance_auc,
    }
    for threshold in thresholds:
        result[metric_name("coverage", threshold)] = float(
            np.mean(final_gt_to_recon < threshold) * 100.0
        )
        result[metric_name("precision", threshold)] = float(
            np.mean(recon_to_gt < threshold) * 100.0
        )

    primary_threshold = 0.05 if 0.05 in thresholds else thresholds[-1]
    primary_key = metric_name("coverage", primary_threshold)
    curve_times = np.asarray([row["elapsed_seconds"] for row in time_rows])
    curve_values = np.asarray([row[primary_key] for row in time_rows]) / 100.0
    if len(curve_times) > 1 and curve_times[-1] > 0:
        result["coverage_time_auc"] = float(
            np.trapz(curve_values, curve_times) / curve_times[-1]
        )
    else:
        result["coverage_time_auc"] = 0.0

    for target_percent in (50.0, 80.0, 90.0):
        reached = np.flatnonzero(curve_values * 100.0 >= target_percent)
        result[f"time_to_{int(target_percent)}pct_coverage_seconds"] = (
            float(curve_times[reached[0]]) if len(reached) else None
        )

    episode_output = output_dir / f"episode_{int(episode_result['episode_index']):05d}"
    episode_output.mkdir(parents=True, exist_ok=True)
    # Preserve the reconstruction before ground-truth alignment for debugging
    # camera geometry, object motion, and physically impossible coverage. For
    # recordings with target poses this has target-motion compensation applied,
    # but it remains in the episode's original world frame.
    trimesh.points.PointCloud(final_unaligned).export(
        episode_output / "reconstruction_unaligned.ply"
    )
    trimesh.points.PointCloud(final_reconstruction).export(
        episode_output / "reconstruction_aligned.ply"
    )
    covered = final_gt_to_recon < primary_threshold
    colors = np.zeros((len(ground_truth), 4), dtype=np.uint8)
    colors[covered] = (0, 255, 0, 255)
    colors[~covered] = (255, 0, 0, 255)
    trimesh.points.PointCloud(ground_truth, colors=colors).export(
        episode_output / "ground_truth_coverage.ply"
    )
    (episode_output / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result, time_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_plots(
    output_dir: Path,
    episode_rows: list[dict],
    time_rows: list[dict],
    thresholds: tuple[float, ...],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_episode = {}
    for row in time_rows:
        by_episode.setdefault(row["episode_index"], []).append(row)

    for threshold in thresholds:
        key = metric_name("coverage", threshold)
        fig, axis = plt.subplots(figsize=(9, 5.5))
        max_duration = max(row["duration_seconds"] for row in episode_rows)
        common_time = np.linspace(0.0, max_duration, 201)
        interpolated = []
        for episode in episode_rows:
            rows = by_episode[episode["episode_index"]]
            times = np.asarray([0.0] + [row["elapsed_seconds"] for row in rows])
            coverage = np.asarray([0.0] + [row[key] for row in rows])
            style = "--" if episode["crashed"] else "-"
            axis.plot(times, coverage, style, alpha=0.22, linewidth=1.0)
            interpolated.append(np.interp(common_time, times, coverage))
        interpolated = np.asarray(interpolated)
        mean_curve = np.mean(interpolated, axis=0)
        std_curve = np.std(interpolated, axis=0)
        axis.plot(common_time, mean_curve, color="black", linewidth=2.5, label="Mean")
        axis.fill_between(
            common_time,
            np.clip(mean_curve - std_curve, 0, 100),
            np.clip(mean_curve + std_curve, 0, 100),
            color="black",
            alpha=0.12,
            label="Mean ± 1 SD",
        )
        axis.set(
            xlabel="Elapsed episode time (s)",
            ylabel=f"Geometric coverage @ {threshold * 100:.0f} cm (%)",
            ylim=(0, 100),
        )
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"coverage_over_time_{int(threshold * 1000):03d}mm.png", dpi=200)
        plt.close(fig)


def aggregate_results(
    episode_rows: list[dict], thresholds: tuple[float, ...], alignment_mode: str
) -> dict:
    crashed = [row for row in episode_rows if row["crashed"]]
    completed = [row for row in episode_rows if not row["crashed"]]
    summary = {
        "episodes": len(episode_rows),
        "crashed_episodes": len(crashed),
        "crash_rate_percent": len(crashed) / len(episode_rows) * 100.0,
        "alignment": alignment_mode,
        "duration_seconds": descriptive_stats(
            [float(row["duration_seconds"]) for row in episode_rows]
        ),
        "time_to_crash_seconds": descriptive_stats(
            [float(row["duration_seconds"]) for row in crashed]
        ),
        "chamfer_distance": descriptive_stats(
            [float(row["chamfer_distance"]) for row in episode_rows]
        ),
        "hausdorff_distance": descriptive_stats(
            [float(row["hausdorff_distance"]) for row in episode_rows]
        ),
        "coverage_time_auc": descriptive_stats(
            [float(row["coverage_time_auc"]) for row in episode_rows]
        ),
        "coverage_distance_auc_0_to_200mm": descriptive_stats(
            [float(row["coverage_distance_auc_0_to_200mm"]) for row in episode_rows]
        ),
    }
    for threshold in thresholds:
        coverage_key = metric_name("coverage", threshold)
        precision_key = metric_name("precision", threshold)
        summary[coverage_key] = descriptive_stats(
            [float(row[coverage_key]) for row in episode_rows]
        )
        summary[precision_key] = descriptive_stats(
            [float(row[precision_key]) for row in episode_rows]
        )
        summary[f"{coverage_key}_crashed"] = descriptive_stats(
            [float(row[coverage_key]) for row in crashed]
        )
        summary[f"{coverage_key}_non_crashed"] = descriptive_stats(
            [float(row[coverage_key]) for row in completed]
        )

    for target_percent in (50, 80, 90):
        key = f"time_to_{target_percent}pct_coverage_seconds"
        values = [row[key] for row in episode_rows if row[key] is not None]
        summary[key] = {
            "episodes_reached": len(values),
            "reach_rate_percent": len(values) / len(episode_rows) * 100.0,
            **descriptive_stats([float(value) for value in values]),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a recorded depth run against a ground-truth point cloud."
    )
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--ground_truth", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=list(DEFAULT_THRESHOLDS),
        help="Coverage/precision distance thresholds in meters.",
    )
    parser.add_argument("--voxel_size", type=float, default=0.005)
    parser.add_argument("--alignment_voxel_size", type=float, default=0.02)
    parser.add_argument("--max_correspondence_distance", type=float, default=0.5)
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument(
        "--episode_ids",
        type=int,
        nargs="+",
        default=None,
        help="Optional recorded episode indices to evaluate.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else run_dir / "pointcloud_evaluation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = tuple(sorted(set(float(value) for value in args.thresholds)))
    if not thresholds or thresholds[0] <= 0:
        raise ValueError("All thresholds must be greater than zero")

    episode_results_path = run_dir / "episode_results.json"
    episode_metadata = json.loads(episode_results_path.read_text(encoding="utf-8"))
    # Initial environment reset can create an empty zero-step record; it is not
    # an evaluation episode and must not enter aggregate statistics.
    episode_metadata = [
        row
        for row in episode_metadata
        if int(row.get("episode_steps", 0)) > 0
        and (run_dir / "episodes" / f"episode_{int(row['episode_index']):05d}" / "transforms.json").exists()
    ]
    if args.episode_ids is not None:
        selected_ids = set(args.episode_ids)
        episode_metadata = [
            row for row in episode_metadata if int(row["episode_index"]) in selected_ids
        ]
    if args.max_episodes is not None:
        episode_metadata = episode_metadata[: args.max_episodes]
    if not episode_metadata:
        raise ValueError("No completed episodes with reconstruction data were found")

    ground_truth = load_point_cloud(args.ground_truth.resolve())
    ground_truth_tree = cKDTree(ground_truth)
    episode_rows = []
    time_rows = []
    failures = []
    for index, metadata in enumerate(episode_metadata, start=1):
        episode_index = int(metadata["episode_index"])
        episode_dir = run_dir / "episodes" / f"episode_{episode_index:05d}"
        print(
            f"[INFO] Evaluating episode {episode_index} "
            f"({index}/{len(episode_metadata)})..."
        )
        try:
            result, episode_time_rows = evaluate_episode(
                episode_dir,
                metadata,
                ground_truth,
                ground_truth_tree,
                thresholds,
                args.voxel_size,
                args.alignment_voxel_size,
                args.max_correspondence_distance,
                output_dir,
            )
            episode_rows.append(result)
            time_rows.extend(episode_time_rows)
        except Exception as exc:
            failures.append({"episode_index": episode_index, "error": str(exc)})
            print(f"[WARNING] Skipping episode {episode_index}: {exc}")

    if not episode_rows:
        raise RuntimeError(f"Every episode failed evaluation: {failures}")

    summary = aggregate_results(
        episode_rows,
        thresholds,
        alignment_mode=(
            "one translation-only ICP transform estimated from each final reconstruction "
            "and reused for all of that episode's timesteps; target orientation is fixed"
        ),
    )
    summary.update(
        {
            "run_dir": str(run_dir),
            "ground_truth": str(args.ground_truth.resolve()),
            "thresholds_meters": list(thresholds),
            "voxel_size_meters": args.voxel_size,
            "failed_episodes": failures,
            "timing_note": (
                "Frame times are interpolated over episode duration when old recordings "
                "do not contain explicit episode_step values."
            ),
        }
    )
    (output_dir / "aggregate_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_csv(output_dir / "episode_metrics.csv", episode_rows)
    write_csv(output_dir / "coverage_over_time.csv", time_rows)
    make_plots(output_dir, episode_rows, time_rows, thresholds)

    print("\n" + "=" * 64)
    print(f"Evaluated episodes: {len(episode_rows)}")
    print(f"Crash rate: {summary['crash_rate_percent']:.2f}%")
    for threshold in thresholds:
        key = metric_name("coverage", threshold)
        stats = summary[key]
        print(
            f"Coverage @ {threshold * 100:.0f} cm: "
            f"mean={stats['mean']:.2f}%  max={stats['max']:.2f}%  "
            f"std={stats['std']:.2f}%"
        )
    print(f"Results: {output_dir}")
    print("=" * 64)


if __name__ == "__main__":
    main()
