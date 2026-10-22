# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Classical planning utilities for the frontier + semantic NBV baseline.

This module deliberately has no Isaac Sim dependencies.  It turns the
inspection environment's log-odds voxel map into a collision-inflated 2-D
navigation grid, plans to exploration frontiers, and ranks reachable views on
a ring around an estimated semantic target centroid.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Iterable

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError


_NEIGHBORS = (
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, math.sqrt(2.0)),
    (-1, 1, math.sqrt(2.0)),
    (1, -1, math.sqrt(2.0)),
    (1, 1, math.sqrt(2.0)),
)


@dataclass(frozen=True)
class PlannerConfig:
    """Parameters shared by frontier and next-best-view planning."""

    resolution: float
    robot_radius: float = 0.35
    safety_margin: float = 0.15
    max_start_snap_radius: float = 0.30
    navigation_z_min: float = 0.10
    navigation_z_max: float = 0.85
    occupied_log_odds: float = 0.0
    known_log_odds: float = 0.05
    min_frontier_cells: int = 5
    frontier_gain_radius: float = 1.0
    frontier_gain_weight: float = 0.035
    frontier_cluster_weight: float = 0.08
    frontier_travel_weight: float = 0.20
    viewpoint_count: int = 12
    viewpoint_standoff: float = 1.0
    min_view_radius: float = 1.15
    max_view_radius: float = 2.50
    viewpoint_snap_radius: float = 0.65
    min_target_standoff: float = 0.65
    visited_view_separation: float = 0.55
    viewpoint_novelty_weight: float = 3.0
    viewpoint_bin_weight: float = 1.0
    viewpoint_clearance_weight: float = 0.35
    viewpoint_travel_weight: float = 0.12
    viewpoint_known_los_weight: float = 0.45


@dataclass
class GridProjection:
    """A collision-inflated ground-plane projection of a 3-D voxel map."""

    traversable: np.ndarray
    known: np.ndarray
    occupied: np.ndarray
    unknown: np.ndarray
    clearance: np.ndarray
    origin_xy: np.ndarray
    resolution: float

    def world_to_grid(self, xy: np.ndarray | Iterable[float]) -> tuple[int, int]:
        point = np.asarray(xy, dtype=np.float64)
        index = np.floor((point[:2] - self.origin_xy) / self.resolution).astype(np.int64)
        return int(index[0]), int(index[1])

    def grid_to_world(self, index: tuple[int, int] | np.ndarray) -> np.ndarray:
        cell = np.asarray(index, dtype=np.float64)
        return self.origin_xy + (cell[:2] + 0.5) * self.resolution

    def in_bounds(self, index: tuple[int, int]) -> bool:
        return 0 <= index[0] < self.traversable.shape[0] and 0 <= index[1] < self.traversable.shape[1]


@dataclass
class DijkstraTree:
    """Shortest-path tree rooted at the robot's current grid cell."""

    start: tuple[int, int]
    distances: np.ndarray
    parents: np.ndarray

    def path_to(self, goal: tuple[int, int]) -> np.ndarray | None:
        if not np.isfinite(self.distances[goal]):
            return None
        path = [goal]
        cursor = goal
        while cursor != self.start:
            parent = self.parents[cursor]
            if parent[0] < 0:
                return None
            cursor = (int(parent[0]), int(parent[1]))
            path.append(cursor)
        path.reverse()
        return np.asarray(path, dtype=np.int32)


@dataclass
class PathPlan:
    """A selected navigation goal and the path leading to it."""

    goal_grid: tuple[int, int]
    goal_world: np.ndarray
    path_grid: np.ndarray
    path_world: np.ndarray
    path_length: float
    score: float


@dataclass
class ViewCandidate(PathPlan):
    """A reachable semantic inspection viewpoint."""

    angle: float
    novelty: float
    clearance: float
    known_los_fraction: float
    desired_world: np.ndarray


class SemanticTargetModel:
    """A bounded semantic surface-volume accumulated from masked depth.

    Repeated pixels are fused into world-frame voxels.  A convex-hull volume
    centroid is used when the observations span 3-D; robust bounds provide a
    stable fallback for early planar or sparse observations.
    """

    def __init__(self, voxel_size: float = 0.05, max_voxels: int = 30_000):
        self.voxel_size = float(voxel_size)
        self.max_voxels = int(max_voxels)
        self._keys = np.empty((0, 3), dtype=np.int64)
        self.detection_frames = 0
        self.view_angles: list[float] = []
        self.view_positions: list[np.ndarray] = []
        self._centroid_cache: np.ndarray | None = None
        self._extent_cache: np.ndarray | None = None
        self._convex_volume_cache = 0.0
        self._centroid_method = "empty"

    @property
    def voxel_count(self) -> int:
        return int(self._keys.shape[0])

    @property
    def points(self) -> np.ndarray:
        return (self._keys.astype(np.float64) + 0.5) * self.voxel_size

    @property
    def centroid(self) -> np.ndarray | None:
        if self.voxel_count == 0:
            return None
        self._update_geometry_cache()
        return self._centroid_cache.copy()

    @property
    def extent(self) -> np.ndarray | None:
        if self.voxel_count == 0:
            return None
        self._update_geometry_cache()
        return self._extent_cache.copy()

    @property
    def convex_volume(self) -> float:
        if self.voxel_count == 0:
            return 0.0
        self._update_geometry_cache()
        return self._convex_volume_cache

    @property
    def centroid_method(self) -> str:
        if self.voxel_count == 0:
            return "empty"
        self._update_geometry_cache()
        return self._centroid_method

    @property
    def horizontal_radius(self) -> float:
        center = self.centroid
        if center is None:
            return 0.0
        radii = np.linalg.norm(self.points[:, :2] - center[None, :2], axis=1)
        return float(np.percentile(radii, 90.0)) if radii.size else 0.0

    def update(self, points_world: np.ndarray) -> int:
        points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        if points.size == 0:
            return 0
        new_keys = np.floor(points / self.voxel_size).astype(np.int64)
        new_keys = np.unique(new_keys, axis=0)
        self._keys = np.unique(np.concatenate((self._keys, new_keys), axis=0), axis=0)
        if self._keys.shape[0] > self.max_voxels:
            # np.unique returns lexicographic order, so an even deterministic
            # subsample preserves the full spatial span without RNG state.
            keep = np.linspace(0, self._keys.shape[0] - 1, self.max_voxels, dtype=np.int64)
            self._keys = self._keys[keep]
        self._centroid_cache = None
        self._extent_cache = None
        self.detection_frames += 1
        return int(new_keys.shape[0])

    def mark_view(self, camera_or_robot_position: np.ndarray) -> None:
        center = self.centroid
        if center is None:
            return
        position = np.asarray(camera_or_robot_position, dtype=np.float64).copy()
        angle = math.atan2(position[1] - center[1], position[0] - center[0])
        self.view_angles.append(angle)
        self.view_positions.append(position)

    def _update_geometry_cache(self) -> None:
        if self._centroid_cache is not None:
            return
        points = self.points
        lower, upper = np.percentile(points, (5.0, 95.0), axis=0)
        fallback_centroid = 0.5 * (lower + upper)
        self._extent_cache = upper - lower
        self._centroid_cache = fallback_centroid
        self._convex_volume_cache = 0.0
        self._centroid_method = "robust_bounds"
        if points.shape[0] < 4:
            return

        # ConvexHull cost depends mainly on hull complexity, but cap its input
        # so repeated semantic updates remain cheap for high-resolution masks.
        if points.shape[0] > 3000:
            sample = np.linspace(0, points.shape[0] - 1, 3000, dtype=np.int64)
            hull_points = points[sample]
        else:
            hull_points = points
        try:
            hull = ConvexHull(hull_points)
        except QhullError:
            return
        reference = np.mean(hull_points[hull.vertices], axis=0)
        weighted_centroid = np.zeros(3, dtype=np.float64)
        total_volume = 0.0
        for simplex in hull.simplices:
            a, b, c = hull_points[simplex]
            volume = abs(float(np.dot(a - reference, np.cross(b - reference, c - reference)))) / 6.0
            if volume <= 0.0:
                continue
            weighted_centroid += volume * (reference + a + b + c) / 4.0
            total_volume += volume
        minimum_volume = max(1.0e-9, 0.25 * self.voxel_size**3)
        if total_volume > minimum_volume:
            self._centroid_cache = weighted_centroid / total_volume
            self._convex_volume_cache = total_volume
            self._centroid_method = "convex_hull"


def project_voxel_map(
    occupancy: np.ndarray,
    map_origin: np.ndarray,
    config: PlannerConfig,
    robot_xy: np.ndarray | None = None,
) -> GridProjection:
    """Project log-odds voxels into known, occupied, and safe ground cells."""

    occupancy = np.asarray(occupancy)
    if occupancy.ndim != 3:
        raise ValueError(f"occupancy must have shape (X, Y, Z), got {occupancy.shape}")
    z_min = max(0, int(math.floor(config.navigation_z_min / config.resolution)))
    z_max = min(occupancy.shape[2], int(math.ceil(config.navigation_z_max / config.resolution)) + 1)
    if z_min >= z_max:
        raise ValueError("navigation height range does not intersect the voxel map")

    navigation_band = occupancy[:, :, z_min:z_max]
    occupied = np.any(navigation_band > config.occupied_log_odds, axis=2)
    free_evidence = np.any(navigation_band < -config.known_log_odds, axis=2)
    known = occupied | free_evidence
    unknown = ~known

    # Treat the finite map boundary as a wall.  It prevents a path from
    # skimming an edge whose out-of-bounds side has no occupancy evidence.
    occupied = occupied.copy()
    occupied[[0, -1], :] = True
    occupied[:, [0, -1]] = True
    known = known | occupied
    unknown = ~known

    clearance = ndimage.distance_transform_edt(~occupied) * config.resolution
    required_clearance = config.robot_radius + config.safety_margin
    traversable = free_evidence & (clearance >= required_clearance)

    projection = GridProjection(
        traversable=traversable,
        known=known,
        occupied=occupied,
        unknown=unknown,
        clearance=clearance,
        origin_xy=np.asarray(map_origin, dtype=np.float64)[:2],
        resolution=config.resolution,
    )

    if robot_xy is not None:
        start = projection.world_to_grid(robot_xy)
        if projection.in_bounds(start):
            # Mapping can label the camera/chassis voxel inconsistently.  The
            # robot's current footprint is observed free by construction, but
            # never overwrite a positively occupied cell.
            seed_radius = max(1, int(math.ceil(0.20 / config.resolution)))
            x0, y0 = start
            xs = slice(max(0, x0 - seed_radius), min(traversable.shape[0], x0 + seed_radius + 1))
            ys = slice(max(0, y0 - seed_radius), min(traversable.shape[1], y0 + seed_radius + 1))
            local_free = ~occupied[xs, ys]
            traversable[xs, ys] |= local_free
    return projection


def build_shortest_path_tree(
    projection: GridProjection,
    start_world: np.ndarray,
    max_start_snap_radius: float | None = None,
) -> DijkstraTree | None:
    """Build an 8-connected shortest-path tree over currently known free space."""

    start = _snap_to_mask(
        projection,
        projection.world_to_grid(start_world),
        projection.traversable,
        max_distance=max_start_snap_radius,
    )
    if start is None:
        return None
    shape = projection.traversable.shape
    distances = np.full(shape, np.inf, dtype=np.float64)
    parents = np.full((*shape, 2), -1, dtype=np.int32)
    distances[start] = 0.0
    queue: list[tuple[float, int, int]] = [(0.0, start[0], start[1])]

    while queue:
        distance, x, y = heapq.heappop(queue)
        if distance != distances[x, y]:
            continue
        for dx, dy, cost in _NEIGHBORS:
            nx, ny = x + dx, y + dy
            if nx < 0 or ny < 0 or nx >= shape[0] or ny >= shape[1]:
                continue
            if not projection.traversable[nx, ny]:
                continue
            if dx != 0 and dy != 0:
                # Do not cut diagonally between two occupied corners.
                if not (projection.traversable[x + dx, y] and projection.traversable[x, y + dy]):
                    continue
            candidate_distance = distance + cost * projection.resolution
            if candidate_distance < distances[nx, ny]:
                distances[nx, ny] = candidate_distance
                parents[nx, ny] = (x, y)
                heapq.heappush(queue, (candidate_distance, nx, ny))
    return DijkstraTree(start=start, distances=distances, parents=parents)


def select_frontier(
    projection: GridProjection,
    tree: DijkstraTree,
    config: PlannerConfig,
    bias_world_xy: np.ndarray | None = None,
    bias_weight: float = 0.0,
) -> PathPlan | None:
    """Select a reachable free/unknown boundary using gain minus travel cost."""

    neighbor_structure = np.ones((3, 3), dtype=bool)
    frontiers = compute_frontier_mask(projection, tree)
    labels, component_count = ndimage.label(frontiers, structure=neighbor_structure)
    if component_count == 0:
        return None

    gain_radius_cells = max(1, int(math.ceil(config.frontier_gain_radius / projection.resolution)))
    axis = np.arange(-gain_radius_cells, gain_radius_cells + 1)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    gain_kernel = (xx * xx + yy * yy <= gain_radius_cells * gain_radius_cells).astype(np.float32)
    unknown_gain = ndimage.convolve(projection.unknown.astype(np.float32), gain_kernel, mode="constant")

    component_sizes = ndimage.sum(frontiers, labels, index=np.arange(1, component_count + 1))
    eligible_labels = np.flatnonzero(component_sizes >= config.min_frontier_cells) + 1
    if eligible_labels.size == 0:
        eligible_labels = np.asarray([int(np.argmax(component_sizes)) + 1])

    best: PathPlan | None = None
    for label_id in eligible_labels:
        cells = np.argwhere(labels == label_id)
        if cells.size == 0:
            continue
        cell_scores = (
            config.frontier_gain_weight * unknown_gain[cells[:, 0], cells[:, 1]]
            - config.frontier_travel_weight * tree.distances[cells[:, 0], cells[:, 1]]
        )
        if bias_world_xy is not None and bias_weight > 0.0:
            world_cells = projection.origin_xy + (cells + 0.5) * projection.resolution
            bias_distance = np.linalg.norm(world_cells - np.asarray(bias_world_xy)[:2], axis=1)
            cell_scores -= bias_weight * bias_distance
        selected_cell = cells[int(np.argmax(cell_scores))]
        goal = (int(selected_cell[0]), int(selected_cell[1]))
        path_grid = tree.path_to(goal)
        if path_grid is None:
            continue
        cluster_bonus = config.frontier_cluster_weight * math.sqrt(float(component_sizes[label_id - 1]))
        score = float(np.max(cell_scores) + cluster_bonus)
        plan = _make_path_plan(projection, tree, goal, path_grid, score)
        if best is None or plan.score > best.score:
            best = plan
    return best


def compute_frontier_mask(projection: GridProjection, tree: DijkstraTree | None = None) -> np.ndarray:
    """Return safe free cells adjacent to unknown space.

    When a shortest-path tree is supplied, unreachable frontier components are
    removed.  Keeping this public lets the debugging dashboard display exactly
    the same frontier definition used by the planner.
    """

    neighbor_structure = np.ones((3, 3), dtype=bool)
    frontiers = projection.traversable & ndimage.binary_dilation(
        projection.unknown, structure=neighbor_structure
    )
    if tree is not None:
        frontiers &= np.isfinite(tree.distances)
    return frontiers


def plan_to_world_goal(
    projection: GridProjection,
    tree: DijkstraTree,
    goal_world: np.ndarray,
    snap_radius: float,
) -> PathPlan | None:
    """Replan to a requested world goal, snapping only to reachable free cells."""

    goal = _snap_reachable_goal(projection, tree, np.asarray(goal_world)[:2], snap_radius)
    if goal is None:
        return None
    path_grid = tree.path_to(goal)
    if path_grid is None:
        return None
    return _make_path_plan(projection, tree, goal, path_grid, score=-float(tree.distances[goal]))


def rank_viewpoints(
    projection: GridProjection,
    tree: DijkstraTree,
    target_centroid: np.ndarray,
    target_radius: float,
    visited_angles: Iterable[float],
    visited_positions: Iterable[np.ndarray],
    config: PlannerConfig,
) -> list[ViewCandidate]:
    """Generate and score a reachable ring of next-best-view candidates."""

    target = np.asarray(target_centroid, dtype=np.float64)
    target_radius = max(float(target_radius), 0.05)
    ring_radius = float(
        np.clip(target_radius + config.viewpoint_standoff, config.min_view_radius, config.max_view_radius)
    )
    previous_angles = list(visited_angles)
    previous_positions = [np.asarray(position, dtype=np.float64)[:2] for position in visited_positions]
    seen_bins = {_angle_bin(angle, config.viewpoint_count) for angle in previous_angles}
    candidates: list[ViewCandidate] = []
    used_goals: set[tuple[int, int]] = set()

    for desired_angle in np.linspace(-math.pi, math.pi, config.viewpoint_count, endpoint=False):
        direction = np.asarray([math.cos(desired_angle), math.sin(desired_angle)])
        desired_world = target[:2] + ring_radius * direction
        goal = _snap_reachable_goal(projection, tree, desired_world, config.viewpoint_snap_radius)
        if goal is None or goal in used_goals:
            continue
        actual_world = projection.grid_to_world(goal)
        actual_radius = float(np.linalg.norm(actual_world - target[:2]))
        if actual_radius < target_radius + config.min_target_standoff:
            continue
        if any(
            np.linalg.norm(actual_world - previous_position) < config.visited_view_separation
            for previous_position in previous_positions
        ):
            continue

        actual_angle = math.atan2(actual_world[1] - target[1], actual_world[0] - target[0])
        if previous_angles:
            nearest_angle = min(circular_distance(actual_angle, angle) for angle in previous_angles)
            novelty = min(1.0, nearest_angle / math.pi)
        else:
            novelty = 1.0
        unvisited_bin = float(_angle_bin(actual_angle, config.viewpoint_count) not in seen_bins)

        line = _grid_line(goal, projection.world_to_grid(target[:2]))
        if line.size == 0:
            continue
        line_world = projection.origin_xy + (line + 0.5) * projection.resolution
        distance_to_target = np.linalg.norm(line_world - target[None, :2], axis=1)
        approach_line = line[distance_to_target > target_radius + 0.10]
        if approach_line.size:
            line_occupied = projection.occupied[approach_line[:, 0], approach_line[:, 1]]
            if np.any(line_occupied):
                continue
            line_known = projection.known[approach_line[:, 0], approach_line[:, 1]]
            known_los_fraction = float(np.mean(line_known))
        else:
            known_los_fraction = 1.0

        path_grid = tree.path_to(goal)
        if path_grid is None:
            continue
        path_length = float(tree.distances[goal])
        clearance = float(projection.clearance[goal])
        normalized_clearance = min(1.0, clearance / max(config.robot_radius + config.safety_margin, 1.0e-6))
        score = (
            config.viewpoint_novelty_weight * novelty
            + config.viewpoint_bin_weight * unvisited_bin
            + config.viewpoint_clearance_weight * normalized_clearance
            + config.viewpoint_known_los_weight * known_los_fraction
            - config.viewpoint_travel_weight * path_length
        )
        base = _make_path_plan(projection, tree, goal, path_grid, score)
        candidates.append(
            ViewCandidate(
                **base.__dict__,
                angle=actual_angle,
                novelty=novelty,
                clearance=clearance,
                known_los_fraction=known_los_fraction,
                desired_world=desired_world,
            )
        )
        used_goals.add(goal)

    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    return candidates


def circular_distance(first: float, second: float) -> float:
    """Return the unsigned shortest distance between two angles."""

    return abs(math.atan2(math.sin(first - second), math.cos(first - second)))


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""

    return math.atan2(math.sin(angle), math.cos(angle))


def _make_path_plan(
    projection: GridProjection,
    tree: DijkstraTree,
    goal: tuple[int, int],
    path_grid: np.ndarray,
    score: float,
) -> PathPlan:
    path_world = projection.origin_xy + (path_grid.astype(np.float64) + 0.5) * projection.resolution
    return PathPlan(
        goal_grid=goal,
        goal_world=projection.grid_to_world(goal),
        path_grid=path_grid,
        path_world=path_world,
        path_length=float(tree.distances[goal]),
        score=float(score),
    )


def _snap_to_mask(
    projection: GridProjection,
    requested: tuple[int, int],
    mask: np.ndarray,
    max_distance: float | None = None,
) -> tuple[int, int] | None:
    if projection.in_bounds(requested) and mask[requested]:
        return requested
    cells = np.argwhere(mask)
    if cells.size == 0:
        return None
    delta = cells - np.asarray(requested)[None, :]
    squared_distances = np.einsum("ij,ij->i", delta, delta)
    selected_index = int(np.argmin(squared_distances))
    if max_distance is not None:
        selected_distance = math.sqrt(float(squared_distances[selected_index])) * projection.resolution
        if selected_distance > max_distance + 1.0e-9:
            return None
    selected = cells[selected_index]
    return int(selected[0]), int(selected[1])


def _snap_reachable_goal(
    projection: GridProjection,
    tree: DijkstraTree,
    requested_world: np.ndarray,
    snap_radius: float,
) -> tuple[int, int] | None:
    requested = projection.world_to_grid(requested_world)
    radius_cells = max(0, int(math.ceil(snap_radius / projection.resolution)))
    x0, y0 = requested
    x_min, x_max = max(0, x0 - radius_cells), min(projection.traversable.shape[0], x0 + radius_cells + 1)
    y_min, y_max = max(0, y0 - radius_cells), min(projection.traversable.shape[1], y0 + radius_cells + 1)
    if x_min >= x_max or y_min >= y_max:
        return None
    reachable = projection.traversable[x_min:x_max, y_min:y_max] & np.isfinite(
        tree.distances[x_min:x_max, y_min:y_max]
    )
    cells = np.argwhere(reachable)
    if cells.size == 0:
        return None
    cells[:, 0] += x_min
    cells[:, 1] += y_min
    world_cells = projection.origin_xy + (cells + 0.5) * projection.resolution
    distances = np.linalg.norm(world_cells - requested_world[None, :2], axis=1)
    valid = distances <= snap_radius + 0.5 * projection.resolution
    if not np.any(valid):
        return None
    valid_cells = cells[valid]
    valid_distances = distances[valid]
    selected = valid_cells[int(np.argmin(valid_distances))]
    return int(selected[0]), int(selected[1])


def _grid_line(start: tuple[int, int], end: tuple[int, int]) -> np.ndarray:
    """Integer supercover-like sampling between two grid cells."""

    delta = max(abs(end[0] - start[0]), abs(end[1] - start[1]))
    if delta == 0:
        return np.asarray([start], dtype=np.int32)
    x = np.rint(np.linspace(start[0], end[0], delta + 1)).astype(np.int32)
    y = np.rint(np.linspace(start[1], end[1], delta + 1)).astype(np.int32)
    return np.unique(np.stack((x, y), axis=1), axis=0)


def _angle_bin(angle: float, count: int) -> int:
    normalized = (wrap_angle(angle) + math.pi) / (2.0 * math.pi)
    return int(math.floor(normalized * count)) % count

