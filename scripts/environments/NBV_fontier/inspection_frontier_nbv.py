# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a frontier-exploration + semantic next-best-view inspection baseline.

The baseline has two stages:

1. Spend a fixed exploration budget building the task's occupancy map and
   navigating to reachable free/unknown frontiers while sweeping the PTZ
   camera for the inspection target.
2. Fuse target-only masked depth into a small semantic voxel volume, estimate
   its convex-volume centroid, and greedily visit collision-free viewpoints
   around that centroid.  Viewpoint utility rewards angular novelty and
   line-of-sight knowledge while penalizing path length.

No target pose or mesh geometry is read from simulator ground truth.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from isaaclab.app import AppLauncher



def _find_isaaclab_repo_root(script_path: Path) -> Path:
    """Find the repository root without depending on this script's folder depth."""

    for candidate in script_path.resolve().parents:
        if (candidate / "source").is_dir() and (candidate / "apps").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate the IsaacLab repository root above {script_path}")


ISAACLAB_REPO_ROOT = _find_isaaclab_repo_root(Path(__file__))
ENVIRONMENT_PRESETS = {
    "simple_warehouse": None,
    "usd_explorer_factory": (
        ISAACLAB_REPO_ROOT
        / "assets/environments/nvidia_usd_explorer"
        / "Usd_Explorer/Samples/Examples/2023_2/Factory/Factory.usd"
    ),
    "defect_workshop": ISAACLAB_REPO_ROOT / "assets/environments/nvidia_defect_detection/shop.usdc",
}


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classical frontier exploration plus semantic next-best-view inspection baseline."
    )
    parser.add_argument("--task", default="Isaac-Inspection-Camera-Direct-v0")
    parser.add_argument("--num_envs", type=int, default=1, help="Only one environment is currently supported.")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=1200, help="Control steps per episode.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable_fabric", action="store_true", default=False)
    parser.add_argument(
        "--environment",
        choices=tuple(ENVIRONMENT_PRESETS) + ("custom",),
        default="simple_warehouse",
    )
    parser.add_argument("--environment_usd", type=str, default=None)
    parser.add_argument("--environment_scale", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--environment_offset", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument(
        "--add_environment_mesh_colliders",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--keep_procedural_obstacles", action="store_true")
    parser.add_argument("--inspection_dataset", choices=("primitive", "evaluation"), default=None)
    parser.add_argument("--inspection_target", type=str, default=None)
    parser.add_argument("--fixed_spawns", action="store_true", help="Use deterministic task spawning.")
    parser.add_argument("--bootstrap_steps", type=int, default=160, help="Initial in-place mapping scan.")
    parser.add_argument(
        "--bootstrap_turn_action",
        type=float,
        default=-0.8,
        help="Normalized angular action during the initial scan (negative turns right).",
    )
    parser.add_argument("--replan_interval", type=int, default=20)
    parser.add_argument(
        "--exploration_budget",
        type=int,
        default=400,
        help="Minimum frontier-exploration steps before switching to semantic NBV.",
    )
    parser.add_argument(
        "--early_nbv",
        action="store_true",
        help="Ablation: switch as soon as semantics are confirmed instead of honoring the exploration budget.",
    )
    parser.add_argument("--semantic_confirmations", type=int, default=3)
    parser.add_argument("--semantic_min_pixels", type=int, default=20)
    parser.add_argument("--semantic_voxel_size", type=float, default=0.05)
    parser.add_argument("--viewpoints", type=int, default=12)
    parser.add_argument("--view_standoff", type=float, default=1.0)
    parser.add_argument("--view_dwell_steps", type=int, default=18)
    parser.add_argument("--goal_tolerance", type=float, default=0.30)
    parser.add_argument("--lookahead", type=float, default=0.25)
    parser.add_argument("--robot_radius", type=float, default=0.35)
    parser.add_argument("--safety_margin", type=float, default=0.15)
    parser.add_argument(
        "--max_start_snap",
        type=float,
        default=0.30,
        help="Maximum distance for snapping the robot to a traversable planning cell.",
    )
    parser.add_argument(
        "--shield_blocked_steps",
        type=int,
        default=3,
        help="Consecutive fully shielded motion commands before recovery starts.",
    )
    parser.add_argument("--recovery_backoff_steps", type=int, default=8)
    parser.add_argument("--recovery_turn_steps", type=int, default=14)
    parser.add_argument("--recovery_reverse_action", type=float, default=0.22)
    parser.add_argument("--recovery_turn_action", type=float, default=0.55)
    parser.add_argument(
        "--safety_shield",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the environment's directional occupancy safety shield.",
    )
    parser.add_argument("--status_interval", type=int, default=25)
    parser.add_argument(
        "--debug_ui",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open a live six-panel planner/perception dashboard (requires a non-headless run).",
    )
    parser.add_argument(
        "--debug_save_dir",
        type=Path,
        default=None,
        help="Optional directory for dashboard PNGs and raw .npz snapshots when the robot gets stuck.",
    )
    parser.add_argument("--debug_plot_interval", type=int, default=5, help="Dashboard refresh period in steps.")
    parser.add_argument("--debug_save_interval", type=int, default=25, help="Dashboard PNG period in steps.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON summary path.")
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _configure_process(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.num_envs != 1:
        parser.error("inspection_frontier_nbv currently supports --num_envs 1 only")
    if args.episodes < 1 or args.max_steps < 1:
        parser.error("--episodes and --max_steps must be positive")
    if args.exploration_budget < 0:
        parser.error("--exploration_budget must be non-negative")
    if args.debug_plot_interval < 1 or args.debug_save_interval < 1:
        parser.error("--debug_plot_interval and --debug_save_interval must be positive")
    if args.max_start_snap < 0.0:
        parser.error("--max_start_snap must be non-negative")
    if args.shield_blocked_steps < 1:
        parser.error("--shield_blocked_steps must be positive")
    if args.recovery_backoff_steps < 0 or args.recovery_turn_steps < 0:
        parser.error("recovery step counts must be non-negative")
    if not 0.0 <= args.recovery_reverse_action <= 1.0:
        parser.error("--recovery_reverse_action must be between 0 and 1")
    if not 0.0 <= args.recovery_turn_action <= 1.0:
        parser.error("--recovery_turn_action must be between 0 and 1")

    os.environ["ISAACLAB_INSPECTION_RUN_MODE"] = "debug"
    if args.inspection_dataset is not None:
        os.environ["ISAACLAB_INSPECTION_DATASET"] = args.inspection_dataset
    if args.inspection_target is not None:
        os.environ["ISAACLAB_INSPECTION_TARGET"] = args.inspection_target

    explicit_environment_path = args.environment_usd is not None
    selected_environment = args.environment_usd or ENVIRONMENT_PRESETS.get(args.environment)
    if args.environment == "custom" and selected_environment is None:
        parser.error("--environment custom requires --environment_usd PATH")
    if selected_environment is None:
        os.environ["ISAACLAB_INSPECTION_ENV_NAME"] = "simple_warehouse"
        for variable in (
            "ISAACLAB_INSPECTION_ENV_USD",
            "ISAACLAB_INSPECTION_ENV_SCALE",
            "ISAACLAB_INSPECTION_ENV_OFFSET",
            "ISAACLAB_INSPECTION_ENV_ADD_MESH_COLLIDERS",
            "ISAACLAB_INSPECTION_DISABLE_PROCEDURAL_OBSTACLES",
        ):
            os.environ.pop(variable, None)
        return

    selected_path = Path(selected_environment).expanduser().resolve()
    if not selected_path.is_file():
        parser.error(f"Environment USD not found: {selected_path}")
    os.environ["ISAACLAB_INSPECTION_ENV_USD"] = str(selected_path)
    environment_name = "custom" if explicit_environment_path else args.environment
    os.environ["ISAACLAB_INSPECTION_ENV_NAME"] = environment_name
    add_colliders = args.add_environment_mesh_colliders
    if add_colliders is None:
        add_colliders = explicit_environment_path or args.environment != "simple_warehouse"
    os.environ["ISAACLAB_INSPECTION_ENV_ADD_MESH_COLLIDERS"] = "1" if add_colliders else "0"
    if args.environment_scale is not None:
        os.environ["ISAACLAB_INSPECTION_ENV_SCALE"] = ",".join(map(str, args.environment_scale))
    if args.environment_offset is not None:
        os.environ["ISAACLAB_INSPECTION_ENV_OFFSET"] = ",".join(map(str, args.environment_offset))
    if not args.keep_procedural_obstacles:
        os.environ["ISAACLAB_INSPECTION_DISABLE_PROCEDURAL_OBSTACLES"] = "1"


parser = _build_argument_parser()
args_cli = parser.parse_args()
_configure_process(args_cli, parser)
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import numpy as np
import torch
import warp as wp

import isaaclab_tasks  # noqa: F401, E402
from isaaclab.utils.math import quat_apply
from isaaclab_tasks.direct.robot_inspection import run_config
from isaaclab_tasks.utils import parse_env_cfg

if __package__:
    from .inspection_frontier_nbv_planner import (
        PathPlan,
        PlannerConfig,
        SemanticTargetModel,
        ViewCandidate,
        build_shortest_path_tree,
        compute_frontier_mask,
        plan_to_world_goal,
        project_voxel_map,
        rank_viewpoints,
        select_frontier,
        wrap_angle,
    )
    from .inspection_frontier_nbv_visualizer import FrontierNBVDebugVisualizer
else:
    from inspection_frontier_nbv_planner import (
        PathPlan,
        PlannerConfig,
        SemanticTargetModel,
        ViewCandidate,
        build_shortest_path_tree,
        compute_frontier_mask,
        plan_to_world_goal,
        project_voxel_map,
        rank_viewpoints,
        select_frontier,
        wrap_angle,
    )
    from inspection_frontier_nbv_visualizer import FrontierNBVDebugVisualizer


def _yaw_from_quaternion(quaternion_wxyz: np.ndarray) -> float:
    w, x, y, z = map(float, quaternion_wxyz)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class FrontierNBVController:
    """State machine joining frontier search, semantic fusion, and NBV control."""

    def __init__(self, environment, args: argparse.Namespace):
        self.env = environment
        self.args = args
        self.device = environment.device
        self.action_dim = int(environment.cfg.action_space.shape[0])
        if self.action_dim < 4:
            raise ValueError("The frontier/NBV baseline requires active pan/tilt (four or five actions).")

        resolution = float(environment.map_manager.resolution)
        self.planner_config = PlannerConfig(
            resolution=resolution,
            robot_radius=args.robot_radius,
            safety_margin=args.safety_margin,
            max_start_snap_radius=args.max_start_snap,
            viewpoint_count=args.viewpoints,
            viewpoint_standoff=args.view_standoff,
        )
        self.target = SemanticTargetModel(voxel_size=args.semantic_voxel_size)
        self.phase = "bootstrap"
        self.step_count = 0
        self.plan: PathPlan | None = None
        self.path_index = 0
        self.last_replan_step = -args.replan_interval
        self.dwell_remaining = 0
        self.consecutive_detections = 0
        self.target_locked = False
        self.semantic_prior_ready_step: int | None = None
        self.exploration_steps_at_lock: int | None = None
        self._prior_wait_message_printed = False
        self.completed_viewpoints = 0
        self.frontier_plans = 0
        self.viewpoint_plans = 0
        self.failed_plans = 0
        self.trajectory_length = 0.0
        self._last_position = self._robot_position().copy()
        self._last_commanded_linear = 0.0
        self._stuck_steps = 0
        self._shield_blocked_steps = 0
        self._recovery_backoff_remaining = 0
        self._recovery_turn_remaining = 0
        self._recovery_turn_sign = 1.0
        self.recovery_attempts = 0
        self.stuck_event_count = 0
        self.last_stuck_plan: PathPlan | None = None
        self.last_stuck_step: int | None = None
        self.last_coverage_percent = 0.0
        self.last_action = np.zeros(self.action_dim, dtype=np.float32)
        self.debug_projection = None
        self.debug_tree = None
        self.debug_frontier_mask: np.ndarray | None = None
        self.last_view_candidates: list[ViewCandidate] = []

    def _robot_position(self) -> np.ndarray:
        return self.env.robot.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)

    def _robot_yaw(self) -> float:
        quaternion = self.env.robot.data.root_quat_w[0].detach().cpu().numpy()
        return _yaw_from_quaternion(quaternion)

    def robot_position(self) -> np.ndarray:
        """Return the robot world position for diagnostics."""

        return self._robot_position()

    def robot_yaw(self) -> float:
        """Return the robot world yaw for diagnostics."""

        return self._robot_yaw()

    @property
    def stuck_steps(self) -> int:
        return self._stuck_steps

    def _semantic_observation(self) -> tuple[float, float] | None:
        mask = self.env._get_semantic_mask(self.env._ptz_camera)
        if mask is None:
            self.consecutive_detections = 0
            return None
        mask_2d = mask[0]
        if mask_2d.ndim == 3:
            mask_2d = mask_2d[..., 0]
        depth = self.env._ptz_camera.data.output["distance_to_image_plane"][0]
        if depth.ndim == 3:
            depth = depth[..., 0]
        valid = mask_2d & torch.isfinite(depth) & (depth > 0.10) & (depth < 25.0)
        indices = valid.nonzero(as_tuple=False)
        if indices.shape[0] < self.args.semantic_min_pixels:
            self.consecutive_detections = 0
            return None

        # Deterministic pixel thinning bounds host transfer and semantic fusion.
        if indices.shape[0] > 4096:
            selection = torch.linspace(0, indices.shape[0] - 1, 4096, device=indices.device).long()
            indices = indices[selection]
        v = indices[:, 0]
        u = indices[:, 1]
        z = depth[v, u]
        intrinsics = self.env._ptz_camera.data.intrinsic_matrices[0]
        x = (u.float() - intrinsics[0, 2]) * z / intrinsics[0, 0]
        y = (v.float() - intrinsics[1, 2]) * z / intrinsics[1, 1]
        points_camera = torch.stack((x, y, z), dim=-1)
        camera_quaternion = self.env._ptz_camera.data.quat_w_ros[0].expand(points_camera.shape[0], -1)
        camera_position = self.env._ptz_camera.data.pos_w[0]
        points_world = quat_apply(camera_quaternion, points_camera) + camera_position
        self.target.update(points_world.detach().cpu().numpy())
        self.consecutive_detections += 1

        height, width = mask_2d.shape
        horizontal_error = float((u.float().mean() - 0.5 * (width - 1)) / max(1.0, 0.5 * width))
        vertical_error = float((v.float().mean() - 0.5 * (height - 1)) / max(1.0, 0.5 * height))
        return horizontal_error, vertical_error

    def update_debug_projection(self):
        """Refresh and cache the same inflated map projection used for planning."""

        manager = self.env.map_manager
        occupancy = (
            wp.to_torch(manager.occupancy_map)
            .view(self.env.num_envs, *manager.map_dims)[0]
            .detach()
            .cpu()
            .numpy()
        )
        robot_position = self._robot_position()
        projection = project_voxel_map(
            occupancy,
            manager.world_map_origins[0],
            self.planner_config,
            robot_xy=robot_position[:2],
        )
        self.debug_projection = projection
        self.debug_tree = build_shortest_path_tree(
            projection,
            robot_position[:2],
            max_start_snap_radius=self.planner_config.max_start_snap_radius,
        )
        if self.debug_tree is None:
            self.debug_frontier_mask = np.zeros_like(projection.traversable)
        else:
            self.debug_frontier_mask = compute_frontier_mask(projection, self.debug_tree)
        return projection

    def _projection_and_tree(self):
        projection = self.update_debug_projection()
        return projection, self.debug_tree

    def _install_plan(self, plan: PathPlan, phase: str) -> None:
        self.plan = plan
        self.path_index = 0
        self.phase = phase
        self.last_replan_step = self.step_count

    def _plan_frontier(self, target_bias: bool = False) -> bool:
        self.last_view_candidates = []
        projection, tree = self._projection_and_tree()
        if tree is None:
            self.failed_plans += 1
            self.plan = None
            return False
        bias = self.target.centroid if target_bias else None
        plan = select_frontier(
            projection,
            tree,
            self.planner_config,
            bias_world_xy=None if bias is None else bias[:2],
            bias_weight=0.35 if target_bias else 0.0,
        )
        if plan is None:
            self.failed_plans += 1
            self.plan = None
            return False
        self.frontier_plans += 1
        self._install_plan(plan, "target_frontier" if target_bias else "frontier")
        return True

    def _plan_viewpoint(self) -> bool:
        centroid = self.target.centroid
        if centroid is None:
            return False
        projection, tree = self._projection_and_tree()
        if tree is None:
            self.failed_plans += 1
            return False
        candidates = rank_viewpoints(
            projection,
            tree,
            centroid,
            self.target.horizontal_radius,
            self.target.view_angles,
            self.target.view_positions,
            self.planner_config,
        )
        self.last_view_candidates = candidates
        if not candidates:
            self.failed_plans += 1
            return self._plan_frontier(target_bias=True)
        selected = candidates[0]
        self.viewpoint_plans += 1
        self._install_plan(selected, "viewpoint")
        print(
            f"[NBV] goal=({selected.goal_world[0]:.2f}, {selected.goal_world[1]:.2f}) "
            f"score={selected.score:.2f} novelty={selected.novelty:.2f} "
            f"path={selected.path_length:.2f} m candidates={len(candidates)}"
        )
        return True

    def _refresh_active_plan(self) -> bool:
        if self.plan is None:
            return False
        projection, tree = self._projection_and_tree()
        if tree is None:
            return False
        replanned = plan_to_world_goal(
            projection,
            tree,
            self.plan.goal_world,
            self.planner_config.viewpoint_snap_radius,
        )
        if replanned is None:
            return False
        if isinstance(self.plan, ViewCandidate):
            replanned = ViewCandidate(
                **replanned.__dict__,
                angle=self.plan.angle,
                novelty=self.plan.novelty,
                clearance=float(projection.clearance[replanned.goal_grid]),
                known_los_fraction=self.plan.known_los_fraction,
                desired_world=self.plan.desired_world,
            )
        self._install_plan(replanned, self.phase)
        return True

    def _choose_recovery_turn_sign(self) -> float:
        """Turn toward the less occupied side of the robot-local map."""

        local_map = getattr(self.env, "current_local_occ_map", None)
        if local_map is None:
            return 1.0 if self.recovery_attempts % 2 == 0 else -1.0

        height = local_map.shape[3]
        z_min = max(1, int(math.floor(0.10 / self.planner_config.resolution)))
        z_max = min(height, int(math.ceil(0.80 / self.planner_config.resolution)) + 1)
        occupied = (local_map[0, :, :, z_min:z_max] > 1.1).any(dim=2)
        center_x = occupied.shape[0] // 2
        center_y = occupied.shape[1] // 2
        front = occupied[center_x:, :]
        right_count = int(front[:, :center_y].sum().item())
        left_count = int(front[:, center_y + 1 :].sum().item())
        if left_count == right_count:
            return 1.0 if self.recovery_attempts % 2 == 0 else -1.0
        return 1.0 if left_count < right_count else -1.0

    def _start_recovery(self, reason: str) -> None:
        """Enter a bounded back-off and turn maneuver before replanning."""

        if self.phase == "recovery":
            return
        if self.plan is not None:
            self.last_stuck_plan = self.plan
        self.last_stuck_step = self.step_count
        self.stuck_event_count += 1
        self.recovery_attempts += 1
        self._recovery_turn_sign = self._choose_recovery_turn_sign()
        self._recovery_backoff_remaining = self.args.recovery_backoff_steps
        self._recovery_turn_remaining = self.args.recovery_turn_steps
        self._shield_blocked_steps = 0
        self._stuck_steps = 0
        self.plan = None
        self.path_index = 0
        self.phase = "recovery"
        direction = "left" if self._recovery_turn_sign > 0.0 else "right"
        print(
            f"[RECOVERY] {reason}; backing for {self._recovery_backoff_remaining} steps, "
            f"then turning {direction} for {self._recovery_turn_remaining} steps."
        )

    def _update_shield_statistics(self) -> None:
        shield = getattr(self.env, "current_safety_shield_scale", None)
        shield_scale = 1.0 if shield is None else float(shield[0].item())
        blocked = abs(self._last_commanded_linear) > 0.05 and shield_scale <= 0.01
        if blocked and self.phase != "recovery":
            self._shield_blocked_steps += 1
        else:
            self._shield_blocked_steps = 0
        if self._shield_blocked_steps >= self.args.shield_blocked_steps:
            self._start_recovery(
                f"safety shield held linear scale at zero for {self._shield_blocked_steps} steps"
            )

    def _recovery_action(self) -> tuple[float, float]:
        """Return the next protected recovery command."""

        if self._recovery_backoff_remaining > 0:
            self._recovery_backoff_remaining -= 1
            return -self.args.recovery_reverse_action, 0.0
        if self._recovery_turn_remaining > 0:
            self._recovery_turn_remaining -= 1
            return 0.0, self._recovery_turn_sign * self.args.recovery_turn_action

        self.phase = "viewpoint" if self.target_locked else "frontier"
        self.last_replan_step = self.step_count - self.args.replan_interval
        return 0.0, 0.0

    def _update_motion_statistics(self) -> None:
        position = self._robot_position()
        displacement = float(np.linalg.norm(position[:2] - self._last_position[:2]))
        if (
            self.phase != "recovery"
            and displacement < 0.006
            and abs(self._last_commanded_linear) > 0.05
        ):
            self._stuck_steps += 1
        else:
            self._stuck_steps = 0
        self.trajectory_length += displacement
        self._last_position = position
        if self._stuck_steps >= 25:
            self._start_recovery("no physical progress for 25 commanded-motion steps")

    def _follow_path(self) -> tuple[float, float, bool]:
        if self.plan is None or self.plan.path_world.size == 0:
            return 0.0, 0.22, False
        position = self._robot_position()[:2]
        path = self.plan.path_world
        start_index = min(self.path_index, len(path) - 1)
        search_end = min(len(path), start_index + 40)
        local_distances = np.linalg.norm(path[start_index:search_end] - position[None, :], axis=1)
        self.path_index = start_index + int(np.argmin(local_distances))

        final_distance = float(np.linalg.norm(path[-1] - position))
        if final_distance <= self.args.goal_tolerance:
            return 0.0, 0.0, True

        lookahead_index = self.path_index
        while lookahead_index + 1 < len(path):
            if np.linalg.norm(path[lookahead_index] - position) >= self.args.lookahead:
                break
            lookahead_index += 1
        target_point = path[lookahead_index]
        desired_heading = math.atan2(target_point[1] - position[1], target_point[0] - position[0])
        heading_error = wrap_angle(desired_heading - self._robot_yaw())
        angular_action = float(np.clip(0.45 * heading_error, -0.55, 0.55))
        if abs(heading_error) > 1.0:
            linear_action = 0.0
        else:
            linear_action = float(0.34 * max(0.15, math.cos(heading_error)))
            if final_distance < 0.7:
                linear_action *= max(0.25, final_distance / 0.7)
        return linear_action, angular_action, False

    def _align_with_target(self) -> tuple[float, bool]:
        centroid = self.target.centroid
        if centroid is None:
            return 0.0, True
        position = self._robot_position()
        desired_heading = math.atan2(centroid[1] - position[1], centroid[0] - position[0])
        error = wrap_angle(desired_heading - self._robot_yaw())
        return float(np.clip(0.45 * error, -0.45, 0.45)), abs(error) < 0.16

    def _camera_action(self, mask_error: tuple[float, float] | None) -> tuple[float, float]:
        if mask_error is not None:
            # Positive pan turns left; positive tilt turns down in this task.
            pan = float(np.clip(-0.75 * mask_error[0], -0.8, 0.8))
            tilt = float(np.clip(0.75 * mask_error[1], -0.8, 0.8))
            if self.phase == "dwell":
                pan += 0.08 * math.sin(0.35 * self.step_count)
                tilt += 0.06 * math.sin(0.23 * self.step_count)
            return float(np.clip(pan, -1.0, 1.0)), float(np.clip(tilt, -1.0, 1.0))

        centroid = self.target.centroid
        if centroid is not None:
            position = self._robot_position()
            bearing = math.atan2(centroid[1] - position[1], centroid[0] - position[0])
            desired_pan = wrap_angle(bearing - self._robot_yaw())
            joint_positions = self.env.robot.data.joint_pos[0, self.env._ptz_joint_indices]
            pan_position = float(joint_positions[0].item())
            camera_height = float(self.env._ptz_camera.data.pos_w[0, 2].item())
            horizontal_distance = float(np.linalg.norm(centroid[:2] - position[:2]))
            desired_tilt = -math.atan2(centroid[2] - camera_height, max(horizontal_distance, 1.0e-4))
            tilt_position = float(joint_positions[1].item())
            return (
                float(np.clip(0.7 * wrap_angle(desired_pan - pan_position), -0.8, 0.8)),
                float(np.clip(0.7 * (desired_tilt - tilt_position), -0.8, 0.8)),
            )

        # Position-servo a bounded raster during exploration.  Unlike an
        # open-loop velocity sine, this cannot drift into a joint limit, and it
        # lets semantics accumulate even while the heavy wheel base is still
        # ramping into its initial turn.
        joint_positions = self.env.robot.data.joint_pos[0, self.env._ptz_joint_indices]
        pan_position = float(joint_positions[0].item())
        tilt_position = float(joint_positions[1].item())
        desired_pan = -1.25 * math.sin(2.0 * math.pi * self.step_count / 90.0)
        desired_tilt = -0.08 + 0.12 * math.sin(0.037 * self.step_count)
        return (
            float(np.clip(0.8 * wrap_angle(desired_pan - pan_position), -0.8, 0.8)),
            float(np.clip(0.8 * (desired_tilt - tilt_position), -0.5, 0.5)),
        )

    def _coverage_percent(self) -> float:
        discovered = int((self.env.best_q_per_face[0] > 0.0).sum().item())
        denominator = max(1.0, float(self.env.coverage_num_faces[0].item()))
        self.last_coverage_percent = 100.0 * discovered / denominator
        return self.last_coverage_percent

    def current_coverage_percent(self) -> float:
        """Return current surface coverage for the debug history."""

        return self._coverage_percent()

    def current_goal_distance(self) -> float:
        """Return planar distance to the active goal, or NaN without a plan."""

        if self.plan is None:
            return math.nan
        return float(np.linalg.norm(self.plan.goal_world - self._robot_position()[:2]))

    def action(self) -> torch.Tensor:
        self._update_motion_statistics()
        self._update_shield_statistics()
        mask_error = self._semantic_observation()
        newly_confirmed_prior = (
            self.consecutive_detections >= self.args.semantic_confirmations
            and self.target.voxel_count >= self.args.semantic_min_pixels
        )
        if newly_confirmed_prior and self.semantic_prior_ready_step is None:
            self.semantic_prior_ready_step = self.step_count
        semantic_prior_ready = self.semantic_prior_ready_step is not None
        budget_complete = self.step_count >= self.args.exploration_budget
        if (
            not self.target_locked
            and semantic_prior_ready
            and (budget_complete or self.args.early_nbv)
        ):
            self.target_locked = True
            self.exploration_steps_at_lock = self.step_count
            self.plan = None
            centroid = self.target.centroid
            print(
                f"[SEMANTIC] target locked at step {self.step_count}; "
                f"centroid=({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}), "
                f"voxels={self.target.voxel_count}, method={self.target.centroid_method}, "
                f"exploration_budget={self.args.exploration_budget}"
            )
        elif (
            semantic_prior_ready
            and not budget_complete
            and not self.args.early_nbv
            and not self._prior_wait_message_printed
        ):
            print(
                f"[SEMANTIC] prior available at step {self.step_count}; continuing frontier exploration "
                f"until step {self.args.exploration_budget}."
            )
            self._prior_wait_message_printed = True

        linear_action = 0.0
        angular_action = 0.0
        if self.phase == "recovery":
            linear_action, angular_action = self._recovery_action()
        elif self.phase == "dwell":
            self.dwell_remaining -= 1
            if self.dwell_remaining <= 0:
                self.target.mark_view(self._robot_position())
                self.completed_viewpoints += 1
                self.plan = None
                self.phase = "viewpoint"
        elif not self.target_locked and self.step_count < self.args.bootstrap_steps:
            self.phase = "bootstrap"
            angular_action = float(np.clip(self.args.bootstrap_turn_action, -1.0, 1.0))
        else:
            needs_replan = self.plan is None
            periodic_replan = self.step_count - self.last_replan_step >= self.args.replan_interval
            if periodic_replan and not needs_replan:
                needs_replan = not self._refresh_active_plan()
            if needs_replan:
                planned = self._plan_viewpoint() if self.target_locked else self._plan_frontier()
                if not planned:
                    if self.debug_tree is None:
                        self._start_recovery("no traversable planner start near the physical robot")
                        linear_action, angular_action = self._recovery_action()
                    else:
                        angular_action = 0.22
            if self.plan is not None:
                linear_action, angular_action, reached = self._follow_path()
                if reached:
                    if self.phase == "viewpoint":
                        angular_action, aligned = self._align_with_target()
                        if aligned:
                            self.phase = "dwell"
                            self.dwell_remaining = self.args.view_dwell_steps
                            linear_action = 0.0
                            angular_action = 0.0
                    else:
                        self.plan = None

        pan_action, tilt_action = self._camera_action(mask_error)
        action = torch.zeros((1, self.action_dim), device=self.device, dtype=torch.float32)
        action[0, 0] = linear_action
        action[0, 1] = angular_action
        action[0, 2] = pan_action
        action[0, 3] = tilt_action
        # A five-action environment interprets zero as the middle zoom bin.
        self._last_commanded_linear = linear_action
        self.last_action = action[0].detach().cpu().numpy().copy()
        self.step_count += 1
        return action

    def print_status(self) -> None:
        centroid = self.target.centroid
        centroid_text = "none" if centroid is None else f"({centroid[0]:.2f},{centroid[1]:.2f},{centroid[2]:.2f})"
        print(
            f"[STATUS] step={self.step_count} phase={self.phase} "
            f"coverage={self._coverage_percent():.2f}% target_voxels={self.target.voxel_count} "
            f"centroid={centroid_text} yaw={self._robot_yaw():.2f} "
            f"explore={min(self.step_count, self.args.exploration_budget)}/{self.args.exploration_budget} "
            f"completed_views={self.completed_viewpoints}"
        )

    def summary(self) -> dict:
        centroid = self.target.centroid
        extent = self.target.extent
        return {
            "steps": self.step_count,
            "final_phase": self.phase,
            "target_locked": self.target_locked,
            "semantic_detection_frames": self.target.detection_frames,
            "semantic_voxels": self.target.voxel_count,
            "semantic_centroid": None if centroid is None else centroid.tolist(),
            "semantic_extent": None if extent is None else extent.tolist(),
            "semantic_convex_volume_m3": self.target.convex_volume,
            "semantic_centroid_method": self.target.centroid_method,
            "semantic_prior_ready_step": self.semantic_prior_ready_step,
            "exploration_budget_steps": self.args.exploration_budget,
            "exploration_steps_at_lock": self.exploration_steps_at_lock,
            "completed_viewpoints": self.completed_viewpoints,
            "frontier_plans": self.frontier_plans,
            "viewpoint_plans": self.viewpoint_plans,
            "failed_plans": self.failed_plans,
            "recovery_attempts": self.recovery_attempts,
            "trajectory_length_m": self.trajectory_length,
            "coverage_percent": self.last_coverage_percent,
        }


def _scalar(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        value = value.flatten()[0].item()
    if isinstance(value, np.generic):
        value = value.item()
    return value


def _finish_episode(controller: FrontierNBVController, info: dict, terminated: bool, truncated: bool) -> dict:
    controller._coverage_percent()
    summary = controller.summary()
    summary["terminated"] = terminated
    summary["truncated"] = truncated
    summary["target_name"] = str(controller.env.env_target_names[0])
    log = info.get("log", {}) if isinstance(info, dict) else {}
    for key in (
        "coverage_percent",
        "faces_discovered",
        "crashes",
        "collision_contact_steps",
        "tipovers",
        "episode_success",
        "episode_timed_out",
        "safety_shield_interventions",
        "safety_shield_intervention_rate",
    ):
        if key in log:
            summary[key] = _scalar(log[key])
    return summary


def main() -> None:
    mode = run_config.cfg_mode
    mode.max_episode_length = args_cli.max_steps
    mode.min_episode_length = args_cli.max_steps
    mode.logging_interval = max(args_cli.max_steps + 1, 1500)
    mode.display_cameras = False
    mode.display_ray_counts = False
    mode.visualisation_mode = None
    mode.enable_voxel_visualization = False
    mode.use_wandb = False
    mode.nav_camera_modality = "depth"
    mode.ptz_camera_modality = "depth"
    mode.enable_depth_sensor_noise = False
    mode.enable_semantic_mask_noise = False
    mode.use_directional_safety_shield = args_cli.safety_shield
    mode.fixed_spawns = args_cli.fixed_spawns
    mode.randomize_spawns = not args_cli.fixed_spawns

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    environment = gym.make(args_cli.task, cfg=env_cfg)
    base_env = environment.unwrapped
    results: list[dict] = []
    controller: FrontierNBVController | None = None
    visualizer: FrontierNBVDebugVisualizer | None = None

    try:
        environment.reset(seed=args_cli.seed)
        print(f"[INFO] Observation space: {environment.observation_space}")
        print(f"[INFO] Action space: {environment.action_space}")
        print("[INFO] Baseline uses occupancy + PTZ depth/semantics only (no ground-truth target pose).")
        controller = FrontierNBVController(base_env, args_cli)
        debug_live = args_cli.debug_ui and not args_cli.headless
        debug_save_dir = args_cli.debug_save_dir
        if args_cli.debug_ui and args_cli.headless:
            if debug_save_dir is None:
                debug_save_dir = Path("data/debug/frontier_nbv")
            print(f"[DEBUG] --debug_ui cannot open under --headless; saving dashboards to {debug_save_dir}.")
        visualizer = FrontierNBVDebugVisualizer(
            live=debug_live,
            save_dir=debug_save_dir,
            plot_interval=args_cli.debug_plot_interval,
            save_interval=args_cli.debug_save_interval,
        )

        while simulation_app.is_running() and len(results) < args_cli.episodes:
            with torch.inference_mode():
                action = controller.action()
                _, _, terminated, truncated, info = environment.step(action)
            visualizer.record(controller)
            visualizer.render(controller)
            if controller.step_count % args_cli.status_interval == 0:
                controller.print_status()

            episode_terminated = bool(terminated[0].item())
            episode_truncated = bool(truncated[0].item())
            if episode_terminated or episode_truncated:
                result = _finish_episode(controller, info, episode_terminated, episode_truncated)
                results.append(result)
                print(f"[EPISODE {len(results)}] {json.dumps(result, sort_keys=True)}")
                if len(results) < args_cli.episodes:
                    controller = FrontierNBVController(base_env, args_cli)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
        if controller is not None and (not results or controller.step_count > 0):
            results.append(_finish_episode(controller, {}, False, False))
    finally:
        if visualizer is not None:
            visualizer.close(controller)
        environment.close()
        if args_cli.output is not None:
            output_path = args_cli.output.expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"[INFO] Wrote {len(results)} episode summaries to {output_path}")
        simulation_app.close()


if __name__ == "__main__":
    main()
