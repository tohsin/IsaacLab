# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the point-cloud SWAP baseline in the robot-inspection environment.

The planner receives the occupancy state built from navigation depth and a
semantic point cloud built from the policy's PTZ depth and semantic channels.
Ground-truth ray-caster faces are read only below, in evaluator functions.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Run the UGV-adapted SWAP baseline.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Inspection-Camera-Direct-v0",
    help="Registered Isaac Lab inspection task.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel trials.")
parser.add_argument("--steps", type=int, default=1200, help="Per-trial simulation-step budget.")
parser.add_argument("--seed", type=int, default=42, help="Environment and planner seed.")
parser.add_argument(
    "--coverage_goal",
    type=float,
    default=0.95,
    help="Evaluation-only feasible-face coverage termination threshold.",
)
parser.add_argument(
    "--inspection_dataset",
    choices=("primitive", "evaluation"),
    default="evaluation",
    help="Inspection target dataset.",
)
parser.add_argument(
    "--inspection_target",
    type=str,
    default="ur10_mount",
    help="Optional single target name from the selected dataset.",
)
parser.add_argument(
    "--output_dir",
    type=Path,
    default=None,
    help="Output directory; defaults to outputs/swap/<timestamp>.",
)
parser.add_argument(
    "--replan_interval",
    type=int,
    default=12,
    help="Maximum steps between receding-horizon graph replans.",
)
parser.add_argument(
    "--graph_samples",
    type=int,
    default=220,
    help="Collision-free samples in each GBPlanner-style local graph.",
)
parser.add_argument(
    "--max_linear_speed",
    type=float,
    default=None,
    help=(
        "Optional chassis speed cap in m/s. By default SWAP receives the same "
        "full normalized linear-action range as the RL policy."
    ),
)
parser.add_argument(
    "--max_yaw_rate",
    type=float,
    default=None,
    help=(
        "Optional yaw-rate cap in rad/s. By default SWAP receives the same "
        "full normalized yaw-action range as the RL policy."
    ),
)
parser.add_argument(
    "--semantic_standoff",
    type=float,
    default=1.0,
    help="Semantic viewpoint standoff in meters.",
)
parser.add_argument(
    "--hole_completion_fraction",
    type=float,
    default=0.50,
    help="Fraction of azimuth sectors seen before entering inspection mode.",
)
parser.add_argument(
    "--visualize_planner",
    action="store_true",
    default=False,
    help="Draw the SWAP route and semantic cloud in the Isaac Sim viewport.",
)
parser.add_argument(
    "--visualize_env_id",
    type=int,
    default=0,
    help="Environment row displayed by --visualize_planner.",
)
parser.add_argument(
    "--visualization_interval",
    type=int,
    default=10,
    help="Steps between planner-overlay and status updates.",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable Fabric and use USD I/O operations.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.steps <= 0:
    parser.error("--steps must be positive")
if args_cli.num_envs <= 0:
    parser.error("--num_envs must be positive")
if not 0.0 < args_cli.coverage_goal <= 1.0:
    parser.error("--coverage_goal must be in (0, 1]")
if not 0.0 <= args_cli.hole_completion_fraction <= 1.0:
    parser.error("--hole_completion_fraction must be in [0, 1]")
if args_cli.visualize_env_id < 0:
    parser.error("--visualize_env_id must be non-negative")
if args_cli.visualize_env_id >= args_cli.num_envs:
    parser.error("--visualize_env_id must be smaller than --num_envs")
if args_cli.visualization_interval <= 0:
    parser.error("--visualization_interval must be positive")
if args_cli.visualize_planner and args_cli.headless:
    parser.error("--visualize_planner requires a viewport; remove --headless")

# These must be set before isaaclab_tasks is imported because task modules
# select datasets and observation shapes at import time.
os.environ["ISAACLAB_INSPECTION_RUN_MODE"] = "eval"
os.environ["ISAACLAB_INSPECTION_DATASET"] = args_cli.inspection_dataset
if args_cli.inspection_target is not None:
    os.environ["ISAACLAB_INSPECTION_TARGET"] = args_cli.inspection_target

args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from swap.planner import (  # noqa: E402
    MapSnapshot,
    SwapPlanner,
    SwapPlannerConfig,
    write_ascii_ply,
    yaw_from_quaternion_wxyz,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class _PlannerViewport:
    """Lightweight debug drawing for one planner in the Isaac Sim viewport."""

    _MODE_COLORS = {
        "exploration": (0.1, 0.7, 1.0, 1.0),
        "semantic_hole_coverage": (1.0, 0.55, 0.05, 1.0),
        "semantic_inspection": (0.1, 1.0, 0.25, 1.0),
    }

    def __init__(self, env_id: int, interval: int):
        import isaacsim.util.debug_draw._debug_draw as omni_debug_draw

        self.env_id = int(env_id)
        self.interval = int(interval)
        self.draw = omni_debug_draw.acquire_debug_draw_interface()

    @staticmethod
    def _segments(points: list[tuple[float, float, float]]):
        return points[:-1], points[1:]

    def update(self, task, planner: SwapPlanner, step: int, coverage: float) -> None:
        if step % self.interval != 0:
            return

        self.draw.clear_lines()
        self.draw.clear_points()

        robot = task.robot.data.root_pos_w[self.env_id].detach().cpu().numpy()
        ground_z = float(task.scene.env_origins[self.env_id, 2].item()) + 0.10
        mode_color = self._MODE_COLORS[planner.mode.value]

        # The remaining receding-horizon route starts at the current robot pose.
        remaining = planner.path_world[planner.path_cursor :]
        route = [(float(robot[0]), float(robot[1]), ground_z)]
        route.extend((float(point[0]), float(point[1]), ground_z) for point in remaining)
        if len(route) >= 2:
            starts, ends = self._segments(route)
            self.draw.draw_lines(
                starts,
                ends,
                [mode_color] * len(starts),
                [6.0] * len(starts),
            )
            self.draw.draw_points(
                route[1:],
                [mode_color] * (len(route) - 1),
                [8.0] * (len(route) - 1),
            )

        # Semantic observations are magenta; the estimated centroid is yellow.
        semantic_points, _, _ = planner.semantic_cloud.arrays()
        if len(semantic_points):
            stride = max(1, int(np.ceil(len(semantic_points) / 2000)))
            visible_points = semantic_points[::stride]
            point_tuples = [tuple(map(float, point)) for point in visible_points]
            self.draw.draw_points(
                point_tuples,
                [(1.0, 0.0, 0.8, 1.0)] * len(point_tuples),
                [3.0] * len(point_tuples),
            )
            centroid = planner.target_centroid
            if centroid is not None:
                self.draw.draw_points(
                    [tuple(map(float, centroid))],
                    [(1.0, 1.0, 0.0, 1.0)],
                    [14.0],
                )

        print(
            f"[SWAP-VIS] step={step} env={self.env_id} mode={planner.mode.value} "
            f"path_remaining={len(remaining)} semantic_voxels={len(planner.semantic_cloud)} "
            f"azimuth_bins={len(planner.observed_azimuth_bins)}/"
            f"{planner.config.semantic_azimuth_bins} coverage={coverage:.2f}%"
        )

    def close(self) -> None:
        self.draw.clear_lines()
        self.draw.clear_points()


def _output_directory() -> Path:
    if args_cli.output_dir is not None:
        output = args_cli.output_dir.expanduser().resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = REPOSITORY_ROOT / "outputs" / "swap" / timestamp
    output.mkdir(parents=True, exist_ok=True)
    return output


def _configure_episode(task) -> None:
    """Set one common evaluator budget/goal without changing planner inputs."""

    curriculum = task.curriculum
    curriculum.current_coverage_threshold = float(args_cli.coverage_goal)
    curriculum.start_coverage_ratio = float(args_cli.coverage_goal)
    curriculum.max_coverage_threshold = float(args_cli.coverage_goal)
    # inspection_env.py declares timeout when episode_length_buf >= max_steps - 1.
    # Add one so the public CLI budget is the exact number of env.step calls.
    curriculum.min_episode_length_limit = int(args_cli.steps) + 1
    curriculum.max_episode_length_limit = int(args_cli.steps) + 1


def _copy_map_snapshots(task) -> list[MapSnapshot]:
    """Copy the mapper once, then split it into independent planner inputs."""

    manager = task.map_manager
    maps = np.asarray(manager.occupancy_map.numpy(), dtype=np.float32).reshape(
        task.num_envs, *manager.map_dims
    )
    return [
        MapSnapshot(
            log_odds=maps[env_id].copy(),
            world_origin=np.asarray(manager.world_map_origins[env_id], dtype=np.float64),
            resolution=float(manager.resolution),
        )
        for env_id in range(task.num_envs)
    ]


def _camera_channels(observation: dict) -> np.ndarray:
    policy = observation["policy"]
    if not isinstance(policy, dict) or "cameras" not in policy:
        raise RuntimeError("SWAP requires the dictionary camera observation")
    cameras = policy["cameras"].detach().cpu().numpy()
    if cameras.shape[-1] != 3:
        raise RuntimeError(
            "SWAP runner expects eval-mode [nav depth, PTZ depth, semantic mask] "
            f"channels, got shape {cameras.shape}"
        )
    return cameras


def _fuse_observations(
    task,
    planners: list[SwapPlanner],
    observation: dict,
    active: np.ndarray,
    step: int,
) -> None:
    cameras = _camera_channels(observation)
    intrinsics = task._ptz_camera.data.intrinsic_matrices.detach().cpu().numpy()
    positions = task._ptz_camera.data.pos_w.detach().cpu().numpy()
    quaternions = task._ptz_camera.data.quat_w_ros.detach().cpu().numpy()
    for env_id, planner in enumerate(planners):
        if not active[env_id]:
            continue
        planner.observe(
            depth=cameras[env_id, ..., 1],
            semantic_mask=cameras[env_id, ..., 2] > 0.5,
            intrinsic_matrix=intrinsics[env_id],
            camera_position=positions[env_id],
            camera_quaternion_wxyz=quaternions[env_id],
            step=step,
        )


def _actions(task, planners: list[SwapPlanner], active: np.ndarray, step: int) -> torch.Tensor:
    positions = task.robot.data.root_pos_w.detach().cpu().numpy()
    quaternions = task.robot.data.root_quat_w.detach().cpu().numpy()
    pt_positions = task.robot.data.joint_pos[:, task._ptz_joint_indices].detach().cpu().numpy()
    action_dim = int(task.single_action_space.shape[0])
    actions = np.zeros((task.num_envs, action_dim), dtype=np.float32)
    for env_id, planner in enumerate(planners):
        if not active[env_id]:
            continue
        command = planner.action(
            robot_position=positions[env_id],
            robot_yaw=yaw_from_quaternion_wxyz(quaternions[env_id]),
            pt_joint_positions=pt_positions[env_id],
            step=step,
        )
        actions[env_id, : min(4, action_dim)] = command[: min(4, action_dim)]
        # A fifth zoom action, when present, remains zero (middle focal length).
    return torch.as_tensor(actions, device=task.device)


def _live_face_coverage(task) -> np.ndarray:
    """Evaluator-only access to accumulated ray-caster face coverage."""

    discovered = (task.best_q_per_face > 0.0).sum(dim=1).float()
    coverage = discovered / task.coverage_num_faces.float().clamp_min(1.0)
    return (100.0 * coverage).detach().cpu().numpy()


def _completed_log_values(info: dict, key: str, count: int) -> list:
    value = info.get("log", {}).get(key)
    if value is None:
        return [None] * count
    if isinstance(value, torch.Tensor):
        values = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, np.ndarray):
        values = value.reshape(-1).tolist()
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = [value]
    return (values + [None] * count)[:count]


def _write_outputs(
    output_dir: Path,
    planners: list[SwapPlanner],
    results: list[dict],
    coverage_rows: list[dict],
    planner_config: SwapPlannerConfig,
) -> None:
    run_config = {
        "task": args_cli.task,
        "num_envs": args_cli.num_envs,
        "steps": args_cli.steps,
        "seed": args_cli.seed,
        "coverage_goal": args_cli.coverage_goal,
        "inspection_dataset": args_cli.inspection_dataset,
        "inspection_target": args_cli.inspection_target,
        "planner": asdict(planner_config),
        "planner_inputs": [
            "navigation-depth occupancy map",
            "PTZ depth",
            "PTZ semantic mask",
            "robot and camera odometry",
        ],
        "evaluation_only": ["ray-caster face coverage", "termination"],
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    with (output_dir / "coverage.csv").open("w", newline="", encoding="utf-8") as stream:
        fieldnames = [
            "env_id",
            "step",
            "coverage_percent",
            "mode",
            "semantic_voxels",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(coverage_rows)

    enriched_results = []
    for env_id, (planner, result) in enumerate(zip(planners, results)):
        points, _, _ = planner.semantic_cloud.arrays()
        cloud_name = f"env_{env_id:03d}_semantic_cloud.ply"
        write_ascii_ply(str(output_dir / cloud_name), points)
        enriched_results.append(
            {
                **result,
                "semantic_cloud": cloud_name,
                "planner": planner.diagnostics(),
            }
        )
    (output_dir / "results.json").write_text(
        json.dumps(enriched_results, indent=2), encoding="utf-8"
    )


def main() -> None:

    output_dir = _output_directory()
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg)
    task = env.unwrapped
    _configure_episode(task)

    actuation_scale = float(task.cfg.action_scale)
    # With no explicit physical cap, give SWAP exactly the same normalized
    # chassis-action envelope as the learned policy. Both methods then pass
    # through the environment's wheel conversion, wheel saturation, and final
    # action_scale, including their coupled effect during simultaneous motion.
    max_linear_action = 1.0
    if args_cli.max_linear_speed is not None:
        max_linear_action = min(
            1.0,
            args_cli.max_linear_speed
            / (float(task.cfg.robot_phys_cfg.max_linear_velocity) * actuation_scale),
        )
    max_angular_action = 1.0
    if args_cli.max_yaw_rate is not None:
        max_angular_action = min(
            1.0,
            args_cli.max_yaw_rate
            / (float(task.cfg.robot_phys_cfg.max_angular_velocity) * actuation_scale),
        )
    base_config = SwapPlannerConfig(
        seed=args_cli.seed,
        replan_interval=args_cli.replan_interval,
        graph_samples=args_cli.graph_samples,
        semantic_standoff_m=args_cli.semantic_standoff,
        hole_completion_fraction=args_cli.hole_completion_fraction,
        max_linear_action=max_linear_action,
        max_angular_action=max_angular_action,
    )
    planners = [
        SwapPlanner(
            SwapPlannerConfig(
                **{
                    **asdict(base_config),
                    "seed": args_cli.seed + env_id,
                }
            )
        )
        for env_id in range(task.num_envs)
    ]
    planner_viewport = (
        _PlannerViewport(
            env_id=args_cli.visualize_env_id,
            interval=args_cli.visualization_interval,
        )
        if args_cli.visualize_planner
        else None
    )
    active = np.ones(task.num_envs, dtype=bool)
    results = [
        {
            "env_id": env_id,
            "target_name": str(task.env_target_names[env_id]),
            "termination_reason": "budget",
            "episode_steps": args_cli.steps,
            "coverage_percent": 0.0,
            "success": False,
        }
        for env_id in range(task.num_envs)
    ]
    coverage_rows: list[dict] = []

    print(
        f"[SWAP] Starting {task.num_envs} trial(s), budget={args_cli.steps} steps, "
        f"face termination={100.0 * args_cli.coverage_goal:.1f}%"
    )
    print(
        "[SWAP] Planner inputs: occupancy + semantic depth point cloud. "
        "Ray-caster faces are evaluator-only."
    )

    try:
        observation, _ = env.reset(seed=args_cli.seed)
        _fuse_observations(task, planners, observation, active, step=0)

        with torch.inference_mode():
            for step in range(args_cli.steps):
                if not active.any() or not simulation_app.is_running():
                    break

                refresh_map = step == 0 or step % args_cli.replan_interval == 0
                refresh_map |= any(
                    active[index]
                    and planners[index].force_replan
                    and len(planners[index].path_world) == 0
                    for index in range(task.num_envs)
                )
                if refresh_map:
                    snapshots = _copy_map_snapshots(task)
                    for env_id, planner in enumerate(planners):
                        if active[env_id]:
                            planner.set_map(snapshots[env_id])

                actions = _actions(task, planners, active, step)
                observation, _, terminated, truncated, info = env.step(actions)
                done = (terminated | truncated).detach().cpu().numpy().astype(bool)
                live_coverage = _live_face_coverage(task)

                done_ids = np.flatnonzero(active & done)
                logged_coverage = _completed_log_values(
                    info, "coverage_percent", len(done_ids)
                )
                logged_success = _completed_log_values(
                    info, "episode_success", len(done_ids)
                )
                logged_crashes = _completed_log_values(info, "crashes", len(done_ids))

                for env_id in np.flatnonzero(active):
                    coverage = float(live_coverage[env_id])
                    if env_id in done_ids:
                        completed_index = int(np.flatnonzero(done_ids == env_id)[0])
                        if logged_coverage[completed_index] is not None:
                            coverage = float(logged_coverage[completed_index])
                    coverage_rows.append(
                        {
                            "env_id": int(env_id),
                            "step": step + 1,
                            "coverage_percent": coverage,
                            "mode": planners[env_id].mode.value,
                            "semantic_voxels": len(planners[env_id].semantic_cloud),
                        }
                    )

                for completed_index, env_id in enumerate(done_ids):
                    success_value = logged_success[completed_index]
                    crash_value = logged_crashes[completed_index]
                    coverage_value = logged_coverage[completed_index]
                    success = bool(success_value) if success_value is not None else False
                    crashes = int(crash_value) if crash_value is not None else 0
                    coverage = (
                        float(coverage_value)
                        if coverage_value is not None
                        else float(live_coverage[env_id])
                    )
                    reason = "coverage_goal" if success else ("crash" if crashes > 0 else "budget")
                    results[env_id].update(
                        {
                            "termination_reason": reason,
                            "episode_steps": step + 1,
                            "coverage_percent": coverage,
                            "success": success,
                            "crashes": crashes,
                        }
                    )
                    active[env_id] = False
                    print(
                        f"[SWAP] env={env_id} target={results[env_id]['target_name']} "
                        f"ended at step={step + 1}: {reason}, coverage={coverage:.2f}%"
                    )

                # DirectRLEnv returns reset observations for completed rows.
                # Never fuse those into the just-completed planner cloud.
                _fuse_observations(task, planners, observation, active, step=step + 1)
                if (
                    planner_viewport is not None
                    and active[args_cli.visualize_env_id]
                ):
                    planner_viewport.update(
                        task,
                        planners[args_cli.visualize_env_id],
                        step=step + 1,
                        coverage=float(live_coverage[args_cli.visualize_env_id]),
                    )

            for env_id in np.flatnonzero(active):
                coverage = float(_live_face_coverage(task)[env_id])
                results[env_id]["coverage_percent"] = coverage
                results[env_id]["episode_steps"] = args_cli.steps
                print(
                    f"[SWAP] env={env_id} target={results[env_id]['target_name']} "
                    f"used full budget: coverage={coverage:.2f}%"
                )
    finally:
        if planner_viewport is not None:
            planner_viewport.close()
        _write_outputs(output_dir, planners, results, coverage_rows, base_config)
        env.close()
        print(f"[SWAP] Results written to {output_dir}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
