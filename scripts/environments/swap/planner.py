"""A native, point-cloud SWAP baseline for the robot-inspection task.

The original SWAP system uses GBPlanner for volumetric exploration and a
semantic mesh for hole coverage and inspection.  This implementation keeps
the GBPlanner structure (sampled collision-free graph, volumetric gain, and
receding-horizon execution), but represents the detected object as a semantic
voxelized point cloud.  This is the useful adaptation for the UGV and sensors
in this repository.

Nothing in this module accepts target poses, ground-truth meshes, ray-caster
face IDs, or face coverage.  Those are evaluator-only quantities in the
runner.
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass
from enum import Enum

import numpy as np


class PlannerMode(str, Enum):
    """Top-level behaviors from SWAP."""

    EXPLORATION = "exploration"
    HOLE_COVERAGE = "semantic_hole_coverage"
    INSPECTION = "semantic_inspection"


@dataclass(frozen=True)
class SwapPlannerConfig:
    """Configuration for one independent planner instance."""

    seed: int = 42
    replan_interval: int = 12
    graph_samples: int = 220
    graph_radius_m: float = 5.0
    graph_connection_radius_m: float = 1.25
    graph_neighbors: int = 10
    gain_radius_m: float = 2.0
    gain_weight: float = 1.0
    distance_weight: float = 0.12
    known_epsilon: float = 1.0e-5
    occupied_log_odds: float = 0.0
    navigation_z_min_m: float = 0.10
    navigation_z_max_m: float = 0.85
    robot_inflation_radius_m: float = 0.45
    frontier_fallback_candidates: int = 32
    waypoint_tolerance_m: float = 0.25
    semantic_voxel_size_m: float = 0.04
    semantic_max_range_m: float = 8.0
    semantic_max_points_per_frame: int = 2048
    semantic_min_pixels: int = 64
    semantic_min_voxels: int = 20
    semantic_standoff_m: float = 1.0
    semantic_standoff_tolerance_m: float = 0.45
    semantic_azimuth_bins: int = 16
    hole_completion_fraction: float = 0.50
    candidate_search_radius_m: float = 0.75
    max_linear_action: float = 0.50
    max_angular_action: float = 0.25
    turn_in_place_angle_rad: float = 0.65
    linear_gain: float = 0.8
    angular_gain: float = 1.2
    camera_pan_limit_rad: float = 1.40
    camera_pan_gain: float = 1.5
    camera_tilt_gain: float = 1.5
    inspection_tilt_scan_amplitude_rad: float = 0.22
    stuck_window_steps: int = 45
    stuck_distance_m: float = 0.08


@dataclass(frozen=True)
class MapSnapshot:
    """A copied per-environment occupancy map in world coordinates."""

    log_odds: np.ndarray
    world_origin: np.ndarray
    resolution: float

    def __post_init__(self) -> None:
        if self.log_odds.ndim != 3:
            raise ValueError(f"log_odds must be XxYxZ, got {self.log_odds.shape}")
        if np.asarray(self.world_origin).shape != (3,):
            raise ValueError("world_origin must have shape (3,)")
        if self.resolution <= 0.0:
            raise ValueError("resolution must be positive")

    def world_to_grid(self, point_xy: np.ndarray) -> tuple[int, int]:
        grid = np.floor(
            (np.asarray(point_xy, dtype=np.float64)[:2] - self.world_origin[:2])
            / self.resolution
        ).astype(np.int64)
        return int(grid[0]), int(grid[1])

    def grid_to_world(self, cell: tuple[int, int]) -> np.ndarray:
        return (
            np.asarray(cell, dtype=np.float64) * self.resolution
            + self.world_origin[:2]
            + 0.5 * self.resolution
        )

    def navigation_layers(
        self, config: SwapPlannerConfig
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return traversable, occupied, known and 3-D unknown masks."""

        z0 = max(
            0,
            int(
                math.floor(
                    (config.navigation_z_min_m - self.world_origin[2])
                    / self.resolution
                )
            ),
        )
        z1 = min(
            self.log_odds.shape[2],
            int(
                math.ceil(
                    (config.navigation_z_max_m - self.world_origin[2])
                    / self.resolution
                )
            ),
        )
        if z1 <= z0:
            z0, z1 = 0, self.log_odds.shape[2]

        slab = self.log_odds[:, :, z0:z1]
        voxel_known = np.abs(slab) > config.known_epsilon
        occupied = np.any(slab > config.occupied_log_odds, axis=2)
        known_column = np.any(voxel_known, axis=2)
        free = known_column & ~occupied

        inflation_cells = int(
            math.ceil(config.robot_inflation_radius_m / self.resolution)
        )
        inflated = _inflate_mask(occupied, inflation_cells)
        traversable = free & ~inflated
        # Navigation uses the robot-height slab, but information gain remains
        # volumetric over the full map height as in GBPlanner.
        unknown_volume = np.abs(self.log_odds) <= config.known_epsilon
        return traversable, occupied, known_column, unknown_volume


class SemanticVoxelCloud:
    """Incremental voxelized target cloud with observation-facing normals."""

    def __init__(self, voxel_size: float):
        self.voxel_size = float(voxel_size)
        # value: sum_xyz, count, sum_observation_normal
        self._voxels: dict[tuple[int, int, int], tuple[np.ndarray, int, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self._voxels)

    def add(self, points: np.ndarray, camera_position: np.ndarray) -> int:
        if len(points) == 0:
            return 0
        points = np.asarray(points, dtype=np.float64)
        keys = np.floor(points / self.voxel_size).astype(np.int64)
        _, unique_indices = np.unique(keys, axis=0, return_index=True)
        added = 0
        camera_position = np.asarray(camera_position, dtype=np.float64)
        for index in unique_indices:
            point = points[index]
            key_array = keys[index]
            key = (int(key_array[0]), int(key_array[1]), int(key_array[2]))
            normal = camera_position - point
            normal_norm = np.linalg.norm(normal)
            if normal_norm > 1.0e-8:
                normal /= normal_norm
            previous = self._voxels.get(key)
            if previous is None:
                self._voxels[key] = (point.copy(), 1, normal.copy())
                added += 1
            else:
                point_sum, count, normal_sum = previous
                self._voxels[key] = (
                    point_sum + point,
                    count + 1,
                    normal_sum + normal,
                )
        return added

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self._voxels:
            empty = np.empty((0, 3), dtype=np.float64)
            return empty, empty.copy(), np.empty(0, dtype=np.int64)
        points = []
        normals = []
        counts = []
        for point_sum, count, normal_sum in self._voxels.values():
            points.append(point_sum / count)
            normal_norm = np.linalg.norm(normal_sum)
            normals.append(
                normal_sum / normal_norm
                if normal_norm > 1.0e-8
                else np.zeros(3, dtype=np.float64)
            )
            counts.append(count)
        return (
            np.asarray(points),
            np.asarray(normals),
            np.asarray(counts, dtype=np.int64),
        )

    @property
    def centroid(self) -> np.ndarray | None:
        points, _, _ = self.arrays()
        if len(points) == 0:
            return None
        # A median is stable while the first few noisy semantic points arrive.
        return np.median(points, axis=0)


def quaternion_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    """Convert a normalized Isaac Lab wxyz quaternion to a rotation matrix."""

    q = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm <= 1.0e-12:
        raise ValueError("quaternion has zero length")
    w, x, y, z = q / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def unproject_semantic_depth(
    depth: np.ndarray,
    semantic_mask: np.ndarray,
    intrinsic_matrix: np.ndarray,
    camera_position: np.ndarray,
    camera_quaternion_wxyz: np.ndarray,
    *,
    max_points: int,
    max_range_m: float,
) -> np.ndarray:
    """Unproject masked ``distance_to_image_plane`` pixels into world space."""

    depth = np.asarray(depth).squeeze()
    semantic_mask = np.asarray(semantic_mask).squeeze().astype(bool)
    valid = semantic_mask & np.isfinite(depth) & (depth > 0.0) & (depth <= max_range_m)
    rows, cols = np.nonzero(valid)
    if len(rows) == 0:
        return np.empty((0, 3), dtype=np.float64)
    if len(rows) > max_points:
        selection = np.linspace(0, len(rows) - 1, max_points, dtype=np.int64)
        rows, cols = rows[selection], cols[selection]

    z = depth[rows, cols].astype(np.float64, copy=False)
    intrinsics = np.asarray(intrinsic_matrix, dtype=np.float64)
    x = (cols - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (rows - intrinsics[1, 2]) * z / intrinsics[1, 1]
    points_camera = np.column_stack((x, y, z))
    rotation = quaternion_matrix_wxyz(camera_quaternion_wxyz)
    return points_camera @ rotation.T + np.asarray(camera_position, dtype=np.float64)


class SwapPlanner:
    """One UGV-adapted SWAP planner.

    Call :meth:`observe` with semantic depth, update the map with
    :meth:`set_map`, and request a normalized action with :meth:`action`.
    """

    def __init__(self, config: SwapPlannerConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.mode = PlannerMode.EXPLORATION
        self.semantic_cloud = SemanticVoxelCloud(config.semantic_voxel_size_m)
        self.target_detected_step: int | None = None
        self.observed_azimuth_bins: set[int] = set()
        self.attempted_azimuth_bins: dict[int, int] = {}
        self.snapshot: MapSnapshot | None = None
        self.path_world = np.empty((0, 2), dtype=np.float64)
        self.path_cursor = 0
        self.path_mode = self.mode
        self.goal_bin: int | None = None
        self.last_plan_step = -10**9
        self.force_replan = True
        self.scan_direction = 1.0
        self.pose_history: list[tuple[int, np.ndarray]] = []
        self.replans = 0
        self.gbplanner_replans = 0
        self.frontier_fallbacks = 0
        self.semantic_replans = 0
        self.planning_seconds = 0.0
        self.mode_steps = {mode.value: 0 for mode in PlannerMode}

    @property
    def target_centroid(self) -> np.ndarray | None:
        return self.semantic_cloud.centroid

    def set_map(self, snapshot: MapSnapshot) -> None:
        self.snapshot = snapshot

    def observe(
        self,
        depth: np.ndarray,
        semantic_mask: np.ndarray,
        intrinsic_matrix: np.ndarray,
        camera_position: np.ndarray,
        camera_quaternion_wxyz: np.ndarray,
        step: int,
    ) -> int:
        """Fuse one semantic depth observation and update behavior state."""

        semantic_mask = _largest_connected_component(
            np.asarray(semantic_mask).squeeze().astype(bool)
        )
        if np.count_nonzero(semantic_mask) < self.config.semantic_min_pixels:
            return 0
        points = unproject_semantic_depth(
            depth,
            semantic_mask,
            intrinsic_matrix,
            camera_position,
            camera_quaternion_wxyz,
            max_points=self.config.semantic_max_points_per_frame,
            max_range_m=self.config.semantic_max_range_m,
        )
        added = self.semantic_cloud.add(points, camera_position)
        if len(self.semantic_cloud) >= self.config.semantic_min_voxels:
            centroid = self.target_centroid
            if centroid is not None:
                angle = math.atan2(
                    camera_position[1] - centroid[1],
                    camera_position[0] - centroid[0],
                )
                self.observed_azimuth_bins.add(self._angle_to_bin(angle))
            if self.target_detected_step is None:
                self.target_detected_step = int(step)
                self.mode = PlannerMode.HOLE_COVERAGE
                self.force_replan = True
        return added

    def action(
        self,
        robot_position: np.ndarray,
        robot_yaw: float,
        pt_joint_positions: np.ndarray,
        step: int,
    ) -> np.ndarray:
        """Return ``[linear, angular, pan, tilt]`` normalized actions."""

        robot_xy = np.asarray(robot_position, dtype=np.float64)[:2]
        self._update_mode()
        self.mode_steps[self.mode.value] += 1
        self._record_motion(step, robot_xy)

        if self._path_finished(robot_xy):
            if self.goal_bin is not None:
                self.attempted_azimuth_bins[self.goal_bin] = (
                    self.attempted_azimuth_bins.get(self.goal_bin, 0) + 1
                )
            self.force_replan = True

        if self._is_stuck(step):
            self.path_world = np.empty((0, 2), dtype=np.float64)
            self.path_cursor = 0
            self.force_replan = True

        if (
            self.force_replan
            or step - self.last_plan_step >= self.config.replan_interval
            or self.path_mode != self.mode
        ):
            self._replan(robot_xy, step)

        linear, angular = self._path_control(robot_xy, robot_yaw)
        pan, tilt = self._camera_control(
            robot_position=np.asarray(robot_position, dtype=np.float64),
            robot_yaw=robot_yaw,
            pt_joint_positions=np.asarray(pt_joint_positions, dtype=np.float64),
            step=step,
        )
        return np.asarray([linear, angular, pan, tilt], dtype=np.float32)

    def diagnostics(self) -> dict:
        return {
            "mode": self.mode.value,
            "target_detected_step": self.target_detected_step,
            "semantic_voxels": len(self.semantic_cloud),
            "observed_azimuth_bins": len(self.observed_azimuth_bins),
            "total_azimuth_bins": self.config.semantic_azimuth_bins,
            "replans": self.replans,
            "gbplanner_replans": self.gbplanner_replans,
            "frontier_fallbacks": self.frontier_fallbacks,
            "semantic_replans": self.semantic_replans,
            "planning_seconds": self.planning_seconds,
            "mode_steps": dict(self.mode_steps),
        }

    def _update_mode(self) -> None:
        if self.target_detected_step is None:
            self.mode = PlannerMode.EXPLORATION
            return
        observed_fraction = len(self.observed_azimuth_bins) / max(
            1, self.config.semantic_azimuth_bins
        )
        if observed_fraction < self.config.hole_completion_fraction:
            self.mode = PlannerMode.HOLE_COVERAGE
        else:
            self.mode = PlannerMode.INSPECTION

    def _replan(self, robot_xy: np.ndarray, step: int) -> None:
        started = time.perf_counter()
        self.replans += 1
        self.last_plan_step = int(step)
        self.force_replan = False
        self.goal_bin = None
        self.path_mode = self.mode

        new_path = np.empty((0, 2), dtype=np.float64)
        if self.snapshot is not None:
            if self.mode == PlannerMode.EXPLORATION:
                self.gbplanner_replans += 1
                new_path = self._plan_exploration(robot_xy)
            else:
                self.semantic_replans += 1
                new_path = self._plan_semantic_viewpoint(robot_xy)
                if len(new_path) == 0:
                    # The known free map may not yet connect to a target-side
                    # candidate. Continue volumetric exploration without
                    # discarding semantic state.
                    self.gbplanner_replans += 1
                    new_path = self._plan_exploration(robot_xy)

        self.path_world = new_path
        self.path_cursor = 0
        # An empty plan commonly occurs on the reset frame before the first
        # navigation-depth map update. Retry on the next step instead of
        # waiting a full replan interval with a stale empty map.
        self.force_replan = len(new_path) == 0
        self.planning_seconds += time.perf_counter() - started

    def _plan_exploration(self, robot_xy: np.ndarray) -> np.ndarray:
        assert self.snapshot is not None
        traversable, _, known, unknown_voxels = self.snapshot.navigation_layers(
            self.config
        )
        start = _nearest_true_cell(
            traversable,
            self.snapshot.world_to_grid(robot_xy),
            max_radius=max(4, int(round(1.0 / self.snapshot.resolution))),
        )
        if start is None:
            return np.empty((0, 2), dtype=np.float64)

        free_cells = np.argwhere(traversable)
        if len(free_cells) == 0:
            return np.empty((0, 2), dtype=np.float64)
        graph_radius_cells = self.config.graph_radius_m / self.snapshot.resolution
        offsets = free_cells - np.asarray(start)
        local_cells = free_cells[np.sum(offsets * offsets, axis=1) <= graph_radius_cells**2]
        if len(local_cells) == 0:
            return np.empty((0, 2), dtype=np.float64)

        sample_count = min(self.config.graph_samples, len(local_cells))
        selected = self.rng.choice(len(local_cells), size=sample_count, replace=False)
        nodes = local_cells[selected]

        # Frontier cells are high-value graph samples and should not be lost to
        # uniform sampling in a large known-free room.
        frontier = _frontier_mask(traversable, known)
        frontier_cells = np.argwhere(frontier)
        if len(frontier_cells):
            frontier_offsets = frontier_cells - np.asarray(start)
            frontier_cells = frontier_cells[
                np.sum(frontier_offsets * frontier_offsets, axis=1) <= graph_radius_cells**2
            ]
            if len(frontier_cells):
                frontier_take = min(48, len(frontier_cells))
                ids = self.rng.choice(len(frontier_cells), frontier_take, replace=False)
                nodes = np.vstack((nodes, frontier_cells[ids]))

        nodes = np.unique(np.vstack((np.asarray(start), nodes)), axis=0)
        root_matches = np.flatnonzero(np.all(nodes == np.asarray(start), axis=1))
        root = int(root_matches[0])
        adjacency = _sample_graph(
            nodes,
            traversable,
            connection_radius=self.config.graph_connection_radius_m
            / self.snapshot.resolution,
            max_neighbors=self.config.graph_neighbors,
        )
        distances, parents = _dijkstra(adjacency, root)
        gains = np.asarray(
            [
                _volumetric_gain(
                    unknown_voxels,
                    (int(node[0]), int(node[1])),
                    self.config.gain_radius_m / self.snapshot.resolution,
                )
                for node in nodes
            ],
            dtype=np.float64,
        )
        reachable = np.isfinite(distances) & (np.arange(len(nodes)) != root)
        if np.any(reachable):
            gain_scale = max(1.0, float(np.max(gains[reachable])))
            metric_distance = distances * self.snapshot.resolution
            utility = (
                self.config.gain_weight * gains / gain_scale
                - self.config.distance_weight * metric_distance
            )
            utility[~reachable] = -np.inf
            best = int(np.argmax(utility))
            if np.isfinite(utility[best]) and gains[best] > 0.0:
                node_path = _reconstruct_node_path(parents, root, best)
                cells = [tuple(map(int, nodes[index])) for index in node_path[1:]]
                if cells:
                    return np.vstack([self.snapshot.grid_to_world(cell) for cell in cells])

        self.frontier_fallbacks += 1
        return self._plan_frontier_fallback(start, traversable, known)

    def _plan_frontier_fallback(
        self,
        start: tuple[int, int],
        traversable: np.ndarray,
        known: np.ndarray,
    ) -> np.ndarray:
        assert self.snapshot is not None
        candidates = np.argwhere(_frontier_mask(traversable, known))
        if len(candidates) == 0:
            return np.empty((0, 2), dtype=np.float64)
        distances = np.linalg.norm(candidates - np.asarray(start), axis=1)
        order = np.argsort(distances)[::-1]
        for index in order[: self.config.frontier_fallback_candidates]:
            goal = tuple(map(int, candidates[index]))
            cells = _a_star(traversable, start, goal)
            if len(cells) > 1:
                cells = _simplify_grid_path(cells, traversable)
                return np.vstack(
                    [self.snapshot.grid_to_world(cell) for cell in cells[1:]]
                )
        return np.empty((0, 2), dtype=np.float64)

    def _plan_semantic_viewpoint(self, robot_xy: np.ndarray) -> np.ndarray:
        assert self.snapshot is not None
        centroid = self.target_centroid
        if centroid is None:
            return np.empty((0, 2), dtype=np.float64)
        traversable, _, _, _ = self.snapshot.navigation_layers(self.config)
        start = _nearest_true_cell(
            traversable,
            self.snapshot.world_to_grid(robot_xy),
            max_radius=max(4, int(round(1.0 / self.snapshot.resolution))),
        )
        if start is None:
            return np.empty((0, 2), dtype=np.float64)

        candidate_bins = list(range(self.config.semantic_azimuth_bins))
        if self.mode == PlannerMode.HOLE_COVERAGE:
            candidate_bins.sort(
                key=lambda item: (
                    item in self.observed_azimuth_bins,
                    self.attempted_azimuth_bins.get(item, 0),
                    _circular_bin_distance(
                        item,
                        self._angle_to_bin(
                            math.atan2(
                                robot_xy[1] - centroid[1],
                                robot_xy[0] - centroid[0],
                            )
                        ),
                        self.config.semantic_azimuth_bins,
                    ),
                )
            )
        else:
            scores = self._inspection_candidate_scores(centroid)
            candidate_bins.sort(
                key=lambda item: (
                    -scores[item],
                    self.attempted_azimuth_bins.get(item, 0),
                )
            )

        search_radius = int(
            math.ceil(
                self.config.candidate_search_radius_m / self.snapshot.resolution
            )
        )
        for bin_index in candidate_bins:
            angle = self._bin_angle(bin_index)
            desired = centroid[:2] + self.config.semantic_standoff_m * np.asarray(
                [math.cos(angle), math.sin(angle)]
            )
            goal = _nearest_true_cell(
                traversable,
                self.snapshot.world_to_grid(desired),
                max_radius=search_radius,
            )
            if goal is None:
                continue
            cells = _a_star(traversable, start, goal)
            if len(cells) <= 1:
                continue
            self.goal_bin = bin_index
            cells = _simplify_grid_path(cells, traversable)
            return np.vstack(
                [self.snapshot.grid_to_world(cell) for cell in cells[1:]]
            )
        return np.empty((0, 2), dtype=np.float64)

    def _inspection_candidate_scores(self, centroid: np.ndarray) -> np.ndarray:
        points, normals, counts = self.semantic_cloud.arrays()
        scores = np.zeros(self.config.semantic_azimuth_bins, dtype=np.float64)
        if len(points) == 0:
            return scores
        weights = 1.0 / np.sqrt(np.maximum(1, counts))
        for bin_index in range(self.config.semantic_azimuth_bins):
            angle = self._bin_angle(bin_index)
            candidate = centroid.copy()
            candidate[:2] += self.config.semantic_standoff_m * np.asarray(
                [math.cos(angle), math.sin(angle)]
            )
            view_vectors = candidate - points
            distances = np.linalg.norm(view_vectors, axis=1)
            valid = distances > 1.0e-6
            unit_views = np.zeros_like(view_vectors)
            unit_views[valid] = view_vectors[valid] / distances[valid, None]
            incidence = np.sum(normals * unit_views, axis=1)
            visible = (
                valid
                & (distances >= self.config.semantic_standoff_m - self.config.semantic_standoff_tolerance_m)
                & (distances <= self.config.semantic_standoff_m + self.config.semantic_standoff_tolerance_m)
                & (incidence > math.cos(math.radians(70.0)))
            )
            scores[bin_index] = float(np.sum(weights[visible]))
            if bin_index not in self.observed_azimuth_bins:
                scores[bin_index] += max(1.0, 0.1 * len(points))
            scores[bin_index] /= 1.0 + self.attempted_azimuth_bins.get(bin_index, 0)
        return scores

    def _path_finished(self, robot_xy: np.ndarray) -> bool:
        while self.path_cursor < len(self.path_world):
            distance = np.linalg.norm(self.path_world[self.path_cursor] - robot_xy)
            if distance > self.config.waypoint_tolerance_m:
                return False
            self.path_cursor += 1
        return len(self.path_world) > 0

    def _path_control(self, robot_xy: np.ndarray, robot_yaw: float) -> tuple[float, float]:
        centroid = self.target_centroid
        if self.path_cursor >= len(self.path_world):
            if centroid is not None:
                heading = math.atan2(centroid[1] - robot_xy[1], centroid[0] - robot_xy[0])
                error = _wrap_angle(heading - robot_yaw)
                return 0.0, float(
                    np.clip(
                        self.config.angular_gain * error,
                        -self.config.max_angular_action,
                        self.config.max_angular_action,
                    )
                )
            # Rotate in place while graph construction waits for map evidence.
            return 0.0, 0.5 * self.config.max_angular_action

        waypoint = self.path_world[self.path_cursor]
        delta = waypoint - robot_xy
        distance = float(np.linalg.norm(delta))
        heading = math.atan2(delta[1], delta[0])
        error = _wrap_angle(heading - robot_yaw)
        angular = float(
            np.clip(
                self.config.angular_gain * error,
                -self.config.max_angular_action,
                self.config.max_angular_action,
            )
        )
        if abs(error) >= self.config.turn_in_place_angle_rad:
            linear = 0.0
        else:
            linear = min(
                self.config.max_linear_action,
                self.config.linear_gain * distance,
            ) * max(0.0, math.cos(error))
        return float(linear), angular

    def _camera_control(
        self,
        robot_position: np.ndarray,
        robot_yaw: float,
        pt_joint_positions: np.ndarray,
        step: int,
    ) -> tuple[float, float]:
        if len(pt_joint_positions) < 2:
            return 0.0, 0.0
        centroid = self.target_centroid
        if centroid is None:
            pan_position = float(pt_joint_positions[0])
            if pan_position >= self.config.camera_pan_limit_rad:
                self.scan_direction = -1.0
            elif pan_position <= -self.config.camera_pan_limit_rad:
                self.scan_direction = 1.0
            return self.scan_direction, 0.0

        relative = centroid - robot_position
        horizontal = max(1.0e-6, float(np.linalg.norm(relative[:2])))
        target_bearing = math.atan2(relative[1], relative[0])
        desired_pan = float(
            np.clip(
                _wrap_angle(target_bearing - robot_yaw),
                -self.config.camera_pan_limit_rad,
                self.config.camera_pan_limit_rad,
            )
        )
        elevation = math.atan2(relative[2], horizontal)
        desired_tilt = -elevation
        if self.mode == PlannerMode.INSPECTION:
            desired_tilt += self.config.inspection_tilt_scan_amplitude_rad * math.sin(
                2.0 * math.pi * step / 80.0
            )
        pan_error = desired_pan - float(pt_joint_positions[0])
        tilt_error = desired_tilt - float(pt_joint_positions[1])
        return (
            float(np.clip(self.config.camera_pan_gain * pan_error, -1.0, 1.0)),
            float(np.clip(self.config.camera_tilt_gain * tilt_error, -1.0, 1.0)),
        )

    def _record_motion(self, step: int, robot_xy: np.ndarray) -> None:
        self.pose_history.append((int(step), robot_xy.copy()))
        cutoff = int(step) - self.config.stuck_window_steps
        while len(self.pose_history) > 2 and self.pose_history[0][0] < cutoff:
            self.pose_history.pop(0)

    def _is_stuck(self, step: int) -> bool:
        if len(self.pose_history) < 2 or len(self.path_world) == 0:
            return False
        first_step, first_position = self.pose_history[0]
        if step - first_step < self.config.stuck_window_steps:
            return False
        return (
            np.linalg.norm(self.pose_history[-1][1] - first_position)
            < self.config.stuck_distance_m
        )

    def _angle_to_bin(self, angle: float) -> int:
        fraction = (_wrap_angle(angle) + math.pi) / (2.0 * math.pi)
        return int(math.floor(fraction * self.config.semantic_azimuth_bins)) % self.config.semantic_azimuth_bins

    def _bin_angle(self, bin_index: int) -> float:
        return (
            -math.pi
            + (bin_index + 0.5)
            * 2.0
            * math.pi
            / self.config.semantic_azimuth_bins
        )


def _inflate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    nx, ny = mask.shape
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    inflated = np.zeros_like(mask, dtype=bool)
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            x0 = radius + dx
            y0 = radius + dy
            inflated |= padded[x0 : x0 + nx, y0 : y0 + ny]
    return inflated


def _largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Keep the largest 4-connected semantic region.

    The configured observation noise creates isolated false-positive pixels.
    Allowing those pixels to seed a semantic cloud can trigger inspection
    before an object has actually been detected.
    """

    if mask.ndim != 2 or not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    largest: list[tuple[int, int]] = []
    for row, col in np.argwhere(mask):
        row, col = int(row), int(col)
        if visited[row, col]:
            continue
        component: list[tuple[int, int]] = []
        stack = [(row, col)]
        visited[row, col] = True
        while stack:
            current_row, current_col = stack.pop()
            component.append((current_row, current_col))
            for next_row, next_col in (
                (current_row - 1, current_col),
                (current_row + 1, current_col),
                (current_row, current_col - 1),
                (current_row, current_col + 1),
            ):
                if not (0 <= next_row < height and 0 <= next_col < width):
                    continue
                if mask[next_row, next_col] and not visited[next_row, next_col]:
                    visited[next_row, next_col] = True
                    stack.append((next_row, next_col))
        if len(component) > len(largest):
            largest = component
    filtered = np.zeros_like(mask, dtype=bool)
    if largest:
        rows, cols = zip(*largest)
        filtered[np.asarray(rows), np.asarray(cols)] = True
    return filtered


def _frontier_mask(traversable: np.ndarray, known: np.ndarray) -> np.ndarray:
    unknown = ~known
    padded = np.pad(unknown, 1, mode="constant", constant_values=False)
    adjacent_unknown = np.zeros_like(unknown)
    nx, ny = unknown.shape
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        adjacent_unknown |= padded[1 + dx : 1 + dx + nx, 1 + dy : 1 + dy + ny]
    return traversable & adjacent_unknown


def _nearest_true_cell(
    mask: np.ndarray, cell: tuple[int, int], max_radius: int
) -> tuple[int, int] | None:
    nx, ny = mask.shape
    cx, cy = cell
    if 0 <= cx < nx and 0 <= cy < ny and mask[cx, cy]:
        return cx, cy
    for radius in range(1, max_radius + 1):
        candidates = []
        for x in range(cx - radius, cx + radius + 1):
            candidates.append((x, cy - radius))
            candidates.append((x, cy + radius))
        for y in range(cy - radius + 1, cy + radius):
            candidates.append((cx - radius, y))
            candidates.append((cx + radius, y))
        valid = [
            item
            for item in candidates
            if 0 <= item[0] < nx and 0 <= item[1] < ny and mask[item]
        ]
        if valid:
            return min(valid, key=lambda item: (item[0] - cx) ** 2 + (item[1] - cy) ** 2)
    return None


def _bresenham(start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    x0, y0 = start
    x1, y1 = goal
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    cells = []
    while True:
        cells.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        doubled = 2 * error
        if doubled >= dy:
            error += dy
            x0 += sx
        if doubled <= dx:
            error += dx
            y0 += sy
    return cells


def _line_is_free(
    traversable: np.ndarray, start: tuple[int, int], goal: tuple[int, int]
) -> bool:
    return all(traversable[cell] for cell in _bresenham(start, goal))


def _sample_graph(
    nodes: np.ndarray,
    traversable: np.ndarray,
    connection_radius: float,
    max_neighbors: int,
) -> list[list[tuple[int, float]]]:
    count = len(nodes)
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(count)]
    if count <= 1:
        return adjacency
    delta = nodes[:, None, :] - nodes[None, :, :]
    distances = np.linalg.norm(delta, axis=2)
    for source in range(count):
        order = np.argsort(distances[source])
        connected = 0
        for target in order[1:]:
            distance = float(distances[source, target])
            if distance > connection_radius:
                break
            if connected >= max_neighbors:
                break
            a = tuple(map(int, nodes[source]))
            b = tuple(map(int, nodes[target]))
            if not _line_is_free(traversable, a, b):
                continue
            adjacency[source].append((int(target), distance))
            if not any(neighbor == source for neighbor, _ in adjacency[target]):
                adjacency[target].append((source, distance))
            connected += 1
    return adjacency


def _dijkstra(
    adjacency: list[list[tuple[int, float]]], root: int
) -> tuple[np.ndarray, np.ndarray]:
    distances = np.full(len(adjacency), np.inf, dtype=np.float64)
    parents = np.full(len(adjacency), -1, dtype=np.int64)
    distances[root] = 0.0
    queue: list[tuple[float, int]] = [(0.0, root)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances[node]:
            continue
        for neighbor, edge_cost in adjacency[node]:
            candidate = distance + edge_cost
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                parents[neighbor] = node
                heapq.heappush(queue, (candidate, neighbor))
    return distances, parents


def _reconstruct_node_path(parents: np.ndarray, root: int, goal: int) -> list[int]:
    path = [goal]
    while path[-1] != root:
        parent = int(parents[path[-1]])
        if parent < 0:
            return []
        path.append(parent)
    path.reverse()
    return path


def _volumetric_gain(
    unknown_voxels: np.ndarray, cell: tuple[int, int], radius: float
) -> float:
    cx, cy = cell
    r = max(1, int(math.ceil(radius)))
    x0, x1 = max(0, cx - r), min(unknown_voxels.shape[0], cx + r + 1)
    y0, y1 = max(0, cy - r), min(unknown_voxels.shape[1], cy + r + 1)
    local = unknown_voxels[x0:x1, y0:y1]
    gx, gy = np.ogrid[x0 - cx : x1 - cx, y0 - cy : y1 - cy]
    circle = gx * gx + gy * gy <= radius * radius
    return float(np.count_nonzero(local[circle]))


def _a_star(
    traversable: np.ndarray, start: tuple[int, int], goal: tuple[int, int]
) -> list[tuple[int, int]]:
    if start == goal:
        return [start]
    nx, ny = traversable.shape
    g_score = np.full((nx, ny), np.inf, dtype=np.float64)
    parents: dict[tuple[int, int], tuple[int, int]] = {}
    g_score[start] = 0.0
    queue: list[tuple[float, float, tuple[int, int]]] = [
        (math.dist(start, goal), 0.0, start)
    ]
    closed: set[tuple[int, int]] = set()
    neighbors = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    )
    while queue:
        _, current_g, current = heapq.heappop(queue)
        if current in closed:
            continue
        if current == goal:
            path = [goal]
            while path[-1] != start:
                path.append(parents[path[-1]])
            path.reverse()
            return path
        closed.add(current)
        for dx, dy, cost in neighbors:
            neighbor = current[0] + dx, current[1] + dy
            if not (0 <= neighbor[0] < nx and 0 <= neighbor[1] < ny):
                continue
            if not traversable[neighbor]:
                continue
            if dx != 0 and dy != 0:
                if not (
                    traversable[current[0] + dx, current[1]]
                    and traversable[current[0], current[1] + dy]
                ):
                    continue
            tentative = current_g + cost
            if tentative >= g_score[neighbor]:
                continue
            g_score[neighbor] = tentative
            parents[neighbor] = current
            priority = tentative + math.dist(neighbor, goal)
            heapq.heappush(queue, (priority, tentative, neighbor))
    return []


def _simplify_grid_path(
    path: list[tuple[int, int]], traversable: np.ndarray
) -> list[tuple[int, int]]:
    if len(path) <= 2:
        return path
    simplified = [path[0]]
    anchor = 0
    while anchor < len(path) - 1:
        furthest = anchor + 1
        for candidate in range(anchor + 2, len(path)):
            if _line_is_free(traversable, path[anchor], path[candidate]):
                furthest = candidate
            else:
                break
        simplified.append(path[furthest])
        anchor = furthest
    return simplified


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _circular_bin_distance(a: int, b: int, count: int) -> int:
    delta = abs(a - b)
    return min(delta, count - delta)


def yaw_from_quaternion_wxyz(quaternion: np.ndarray) -> float:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def write_ascii_ply(path: str, points: np.ndarray) -> None:
    """Write a dependency-free semantic reconstruction artifact."""

    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    with open(path, "w", encoding="utf-8") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(points)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        if len(points):
            np.savetxt(stream, points, fmt="%.7f")
