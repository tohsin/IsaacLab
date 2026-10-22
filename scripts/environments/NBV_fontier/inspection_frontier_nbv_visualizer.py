# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Live and saved diagnostics for the frontier + semantic NBV baseline."""

from __future__ import annotations

from pathlib import Path
import math

import numpy as np
from scipy.spatial import ConvexHull, QhullError


class FrontierNBVDebugVisualizer:
    """Render a six-panel view of perception, planning, and control state."""

    def __init__(
        self,
        live: bool = False,
        save_dir: Path | None = None,
        plot_interval: int = 5,
        save_interval: int = 25,
    ):
        self.live = bool(live)
        self.save_dir = None if save_dir is None else Path(save_dir).expanduser().resolve()
        self.plot_interval = max(1, int(plot_interval))
        self.save_interval = max(1, int(save_interval))
        self.enabled = self.live or self.save_dir is not None
        self._last_rendered_step = -1
        self._last_dumped_stuck_event = 0
        self.steps: list[int] = []
        self.coverage: list[float] = []
        self.goal_distance: list[float] = []
        self.shield_scale: list[float] = []
        self.linear_action: list[float] = []
        self.angular_action: list[float] = []

        self.plt = None
        self.figure = None
        self.axes = None
        if not self.enabled:
            return
        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        import matplotlib

        if not self.live:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self.plt = plt
        if self.live:
            plt.ion()
        self.figure, self.axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
        try:
            self.figure.canvas.manager.set_window_title("Inspection: Frontier + Semantic NBV Debugger")
        except AttributeError:
            pass

    def record(self, controller) -> None:
        """Record low-cost scalar diagnostics after an environment step."""

        if not self.enabled:
            return
        self.steps.append(int(controller.step_count))
        self.coverage.append(float(controller.current_coverage_percent()))
        self.goal_distance.append(float(controller.current_goal_distance()))
        shield = getattr(controller.env, "current_safety_shield_scale", None)
        self.shield_scale.append(1.0 if shield is None else float(shield[0].item()))
        action = controller.last_action
        self.linear_action.append(float(action[0]) if action.size > 0 else 0.0)
        self.angular_action.append(float(action[1]) if action.size > 1 else 0.0)

    def render(self, controller, force: bool = False) -> None:
        """Update the dashboard and optionally save a PNG/raw stuck snapshot."""

        if not self.enabled:
            return
        step = int(controller.step_count)
        new_stuck_event = controller.stuck_event_count > self._last_dumped_stuck_event
        should_draw = force or new_stuck_event or step % self.plot_interval == 0
        if not should_draw or step == self._last_rendered_step:
            return

        projection = controller.update_debug_projection()
        frontier_mask = controller.debug_frontier_mask
        if frontier_mask is None and controller.debug_tree is not None:
            if __package__:
                from .inspection_frontier_nbv_planner import compute_frontier_mask
            else:
                from inspection_frontier_nbv_planner import compute_frontier_mask

            frontier_mask = compute_frontier_mask(projection, controller.debug_tree)

        self._draw_map(controller, projection, frontier_mask)
        self._draw_clearance_and_plan(controller, projection, frontier_mask)
        self._draw_sensor(controller)
        self._draw_semantics(controller)
        self._draw_history()
        self._draw_status(controller, projection, frontier_mask)
        self.figure.suptitle(
            f"Frontier + Semantic NBV | step {step} | phase {controller.phase}", fontsize=15
        )

        if self.live:
            self.figure.canvas.draw_idle()
            self.figure.canvas.flush_events()
            self.plt.pause(0.001)
        if self.save_dir is not None and (force or new_stuck_event or step % self.save_interval == 0):
            self.figure.savefig(self.save_dir / f"dashboard_{step:05d}.png", dpi=130)
        if new_stuck_event:
            if self.save_dir is not None:
                self._dump_stuck_state(controller, projection, frontier_mask)
            self._last_dumped_stuck_event = controller.stuck_event_count
        self._last_rendered_step = step

    def close(self, controller=None) -> None:
        if not self.enabled:
            return
        if controller is not None:
            self.render(controller, force=True)
        if self.live:
            self.plt.ioff()
        self.plt.close(self.figure)

    @staticmethod
    def _extent(projection) -> tuple[float, float, float, float]:
        x_max = projection.origin_xy[0] + projection.traversable.shape[0] * projection.resolution
        y_max = projection.origin_xy[1] + projection.traversable.shape[1] * projection.resolution
        return projection.origin_xy[0], x_max, projection.origin_xy[1], y_max

    @staticmethod
    def _grid_points(projection, mask: np.ndarray) -> np.ndarray:
        cells = np.argwhere(mask)
        if cells.size == 0:
            return np.empty((0, 2), dtype=np.float64)
        return projection.origin_xy + (cells.astype(np.float64) + 0.5) * projection.resolution

    def _draw_map(self, controller, projection, frontier_mask) -> None:
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Circle, Patch

        axis = self.axes[0, 0]
        axis.clear()
        # 0 unknown, 1 known free, 2 inflated/no-go free, 3 occupied, 4 traversable.
        classes = np.zeros(projection.traversable.shape, dtype=np.uint8)
        known_free = projection.known & ~projection.occupied
        classes[known_free] = 1
        classes[known_free & ~projection.traversable] = 2
        classes[projection.occupied] = 3
        classes[projection.traversable] = 4
        colors = ["#777777", "#f7f7f7", "#f4a261", "#161616", "#bde0fe"]
        axis.imshow(
            classes.T,
            origin="lower",
            extent=self._extent(projection),
            interpolation="nearest",
            cmap=ListedColormap(colors),
            vmin=0,
            vmax=4,
        )
        if frontier_mask is not None:
            frontiers = self._grid_points(projection, frontier_mask)
            if frontiers.size:
                axis.scatter(frontiers[:, 0], frontiers[:, 1], s=5, c="#00d9ff", label="reachable frontier")
        self._draw_robot(axis, controller)
        robot = controller.robot_position()
        axis.add_patch(
            Circle(
                robot[:2],
                controller.planner_config.robot_radius + controller.planner_config.safety_margin,
                fill=False,
                edgecolor="#e63946",
                linewidth=1.5,
                label="inflated footprint",
            )
        )
        axis.set_title("Navigation map: unknown / free / inflated / occupied")
        axis.set_xlabel("world X [m]")
        axis.set_ylabel("world Y [m]")
        axis.set_aspect("equal")
        self._focus_map(axis, controller, projection)
        state_handles = [
            Patch(facecolor=colors[0], label="unknown"),
            Patch(facecolor=colors[1], edgecolor="#999999", label="known free"),
            Patch(facecolor=colors[2], label="inflated/no-go"),
            Patch(facecolor=colors[3], label="occupied"),
            Patch(facecolor=colors[4], label="traversable"),
        ]
        plot_handles, _ = axis.get_legend_handles_labels()
        axis.legend(handles=state_handles + plot_handles, loc="upper right", fontsize=6, ncol=2)

    def _draw_clearance_and_plan(self, controller, projection, frontier_mask) -> None:
        from matplotlib.patches import Circle

        axis = self.axes[0, 1]
        axis.clear()
        clearance = np.ma.masked_where(~projection.known, projection.clearance)
        axis.imshow(
            clearance.T,
            origin="lower",
            extent=self._extent(projection),
            interpolation="nearest",
            cmap="viridis",
            vmin=0.0,
            vmax=2.0,
        )
        if frontier_mask is not None and controller.phase in {"bootstrap", "frontier", "target_frontier"}:
            frontiers = self._grid_points(projection, frontier_mask)
            if frontiers.size:
                axis.scatter(frontiers[:, 0], frontiers[:, 1], s=4, c="#00ffff", alpha=0.8)
        plan = controller.plan
        if plan is not None and plan.path_world.size:
            axis.plot(plan.path_world[:, 0], plan.path_world[:, 1], "-", color="#ffd60a", linewidth=2.2)
            axis.scatter(*plan.goal_world, marker="X", s=100, c="#ff00d4", edgecolors="white")
        failed_plan = controller.last_stuck_plan
        if failed_plan is not None and failed_plan.path_world.size:
            axis.plot(
                failed_plan.path_world[:, 0],
                failed_plan.path_world[:, 1],
                "--",
                color="#ff3b30",
                linewidth=1.6,
                alpha=0.8,
                label=f"last stuck path (step {controller.last_stuck_step})",
            )

        candidates = controller.last_view_candidates
        if candidates:
            points = np.asarray([candidate.goal_world for candidate in candidates])
            scores = np.asarray([candidate.score for candidate in candidates])
            axis.scatter(
                points[:, 0],
                points[:, 1],
                c=scores,
                cmap="plasma",
                s=55,
                edgecolors="white",
                linewidths=0.5,
            )
            for index, point in enumerate(points):
                axis.text(point[0], point[1], str(index), fontsize=7, color="white", ha="center", va="center")
        centroid = controller.target.centroid
        if centroid is not None:
            axis.scatter(centroid[0], centroid[1], marker="*", s=180, c="#39ff14", edgecolors="black")
            radius = float(
                np.clip(
                    controller.target.horizontal_radius + controller.planner_config.viewpoint_standoff,
                    controller.planner_config.min_view_radius,
                    controller.planner_config.max_view_radius,
                )
            )
            axis.add_patch(Circle(centroid[:2], radius, fill=False, linestyle="--", edgecolor="#39ff14"))
        for view in controller.target.view_positions:
            axis.scatter(view[0], view[1], marker="D", s=38, c="#00ff88", edgecolors="black")
        self._draw_robot(axis, controller)
        axis.set_title("Clearance [0–2 m], path, frontier and NBV utility")
        axis.set_xlabel("world X [m]")
        axis.set_ylabel("world Y [m]")
        axis.set_aspect("equal")
        self._focus_map(axis, controller, projection)

    def _draw_sensor(self, controller) -> None:
        axis = self.axes[0, 2]
        axis.clear()
        camera = controller.env._ptz_camera
        rgb = camera.data.output.get("rgb")
        if rgb is not None:
            image = rgb[0].detach().cpu().numpy()[..., :3]
            if image.dtype != np.uint8:
                scale = 255.0 if np.nanmax(image) <= 1.0 else 1.0
                image = np.clip(image * scale, 0, 255).astype(np.uint8)
            mask = controller.env._get_semantic_mask(camera)
            if mask is not None:
                semantic = mask[0].detach().cpu().numpy().squeeze().astype(bool)
                overlay = image.copy()
                overlay[semantic] = (0, 255, 70)
                image = (0.60 * image + 0.40 * overlay).astype(np.uint8)
            axis.imshow(image)
            axis.set_title("PTZ RGB + semantic target mask")
        else:
            depth = camera.data.output["distance_to_image_plane"][0].detach().cpu().numpy().squeeze()
            axis.imshow(np.clip(depth, 0.0, 10.0), cmap="turbo", vmin=0.0, vmax=10.0)
            axis.set_title("PTZ depth [m]")
        axis.axis("off")

    def _draw_semantics(self, controller) -> None:
        axis = self.axes[1, 0]
        axis.clear()
        points = controller.target.points
        if points.shape[0] > 4000:
            selection = np.linspace(0, points.shape[0] - 1, 4000, dtype=np.int64)
            display_points = points[selection]
        else:
            display_points = points
        if display_points.size:
            axis.scatter(
                display_points[:, 0],
                display_points[:, 1],
                c=display_points[:, 2],
                cmap="turbo",
                s=5,
                alpha=0.65,
            )
            try:
                hull = ConvexHull(display_points[:, :2])
                polygon = np.append(hull.vertices, hull.vertices[0])
                axis.plot(display_points[polygon, 0], display_points[polygon, 1], "-", color="#ff006e")
            except QhullError:
                pass
        centroid = controller.target.centroid
        if centroid is not None:
            axis.scatter(centroid[0], centroid[1], marker="*", s=180, c="#39ff14", edgecolors="black")
        for position in controller.target.view_positions:
            if centroid is not None:
                axis.plot(
                    [position[0], centroid[0]],
                    [position[1], centroid[1]],
                    "--",
                    color="#00a6fb",
                    alpha=0.5,
                )
            axis.scatter(position[0], position[1], marker="D", s=35, c="#00a6fb")
        self._draw_robot(axis, controller)
        axis.set_title("Semantic volume top-down (color = height) + convex hull")
        axis.set_xlabel("world X [m]")
        axis.set_ylabel("world Y [m]")
        axis.set_aspect("equal")
        if not display_points.size:
            axis.text(
                0.5,
                0.92,
                "No semantic target voxels yet",
                transform=axis.transAxes,
                ha="center",
                color="#555555",
            )
        if controller.debug_projection is not None:
            self._focus_map(axis, controller, controller.debug_projection)

    def _draw_history(self) -> None:
        axis = self.axes[1, 1]
        axis.clear()
        if self.steps:
            goal_normalized = np.clip(np.asarray(self.goal_distance) / 5.0, 0.0, 1.0)
            axis.plot(self.steps, np.asarray(self.coverage) / 100.0, label="coverage / 100", color="#4361ee")
            axis.plot(self.steps, goal_normalized, label="goal distance / 5 m", color="#f77f00")
            axis.plot(self.steps, self.shield_scale, label="safety-shield scale", color="#d00000")
            axis.plot(self.steps, np.abs(self.linear_action), ":", label="|linear action|", color="#2a9d8f")
            axis.plot(self.steps, np.abs(self.angular_action), ":", label="|angular action|", color="#8338ec")
        axis.set_ylim(-0.05, 1.05)
        axis.set_xlabel("control step")
        axis.set_title("Progress and blockage diagnostics")
        axis.grid(alpha=0.25)
        axis.legend(loc="upper left", fontsize=7)

    def _draw_status(self, controller, projection, frontier_mask) -> None:
        axis = self.axes[1, 2]
        axis.clear()
        axis.axis("off")
        known_fraction = 100.0 * float(np.mean(projection.known))
        occupied_fraction = 100.0 * float(np.mean(projection.occupied))
        traversable_fraction = 100.0 * float(np.mean(projection.traversable))
        frontier_count = 0 if frontier_mask is None else int(np.count_nonzero(frontier_mask))
        plan = controller.plan
        goal = "none" if plan is None else f"({plan.goal_world[0]:.2f}, {plan.goal_world[1]:.2f})"
        plan_length = float("nan") if plan is None else plan.path_length
        centroid = controller.target.centroid
        centroid_text = "none" if centroid is None else f"({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})"
        shield = getattr(controller.env, "current_safety_shield_scale", None)
        shield_scale = 1.0 if shield is None else float(shield[0].item())
        actual_action = getattr(controller.env, "actions", None)
        if actual_action is None:
            actual_text = "n/a"
        else:
            values = actual_action[0, :2].detach().cpu().numpy()
            actual_text = f"v={values[0]:+.2f}, w={values[1]:+.2f}"
        candidate_scores = [candidate.score for candidate in controller.last_view_candidates[:5]]
        map_text = (
            f"map known / occupied / safe: {known_fraction:.1f}% / "
            f"{occupied_fraction:.2f}% / {traversable_fraction:.1f}%\n"
        )
        text = (
            f"phase: {controller.phase}\n"
            f"{map_text}"
            f"reachable frontier cells: {frontier_count}\n"
            f"goal: {goal} | remaining={controller.current_goal_distance():.2f} m\n"
            f"planned path length: {plan_length:.2f} m\n"
            f"policy action: v={controller.last_action[0]:+.2f}, w={controller.last_action[1]:+.2f}\n"
            f"applied action: {actual_text}\n"
            f"safety-shield scale: {shield_scale:.2f}\n"
            f"no-progress counter / events: {controller.stuck_steps} / {controller.stuck_event_count}\n\n"
            f"shield-blocked counter: {controller._shield_blocked_steps}\n"
            f"recovery attempts: {controller.recovery_attempts}\n"
            f"recovery backoff / turn: {controller._recovery_backoff_remaining} / "
            f"{controller._recovery_turn_remaining}\n\n"
            f"semantic voxels: {controller.target.voxel_count}\n"
            f"centroid: {centroid_text}\n"
            f"centroid method: {controller.target.centroid_method}\n"
            f"convex volume: {controller.target.convex_volume:.3f} m³\n"
            f"completed / candidate views: {controller.completed_viewpoints} / {len(controller.last_view_candidates)}\n"
            f"top NBV scores: {np.round(candidate_scores, 2).tolist()}"
        )
        axis.text(
            0.02,
            0.98,
            text,
            transform=axis.transAxes,
            va="top",
            ha="left",
            family="monospace",
            fontsize=9,
            bbox={"facecolor": "#f6f6f6", "edgecolor": "#777777", "alpha": 0.95},
        )
        axis.set_title("Planner state")

    @staticmethod
    def _draw_robot(axis, controller) -> None:
        position = controller.robot_position()
        yaw = controller.robot_yaw()
        axis.scatter(position[0], position[1], marker="o", s=55, c="#e63946", edgecolors="white", zorder=8)
        axis.arrow(
            position[0],
            position[1],
            0.55 * math.cos(yaw),
            0.55 * math.sin(yaw),
            width=0.035,
            head_width=0.18,
            color="#e63946",
            length_includes_head=True,
            zorder=8,
        )

    @staticmethod
    def _focus_map(axis, controller, projection) -> None:
        robot = controller.robot_position()
        points = [robot[:2]]
        if controller.plan is not None:
            points.append(controller.plan.goal_world)
        centroid = controller.target.centroid
        if centroid is not None:
            points.append(centroid[:2])
        points = np.asarray(points)
        margin = 2.0
        full_extent = FrontierNBVDebugVisualizer._extent(projection)
        axis.set_xlim(
            max(full_extent[0], points[:, 0].min() - margin),
            min(full_extent[1], points[:, 0].max() + margin),
        )
        axis.set_ylim(
            max(full_extent[2], points[:, 1].min() - margin),
            min(full_extent[3], points[:, 1].max() + margin),
        )

    def _dump_stuck_state(self, controller, projection, frontier_mask) -> None:
        failed_plan = controller.last_stuck_plan
        plan_path = np.empty((0, 2)) if failed_plan is None else failed_plan.path_world
        candidate_points = np.asarray(
            [candidate.goal_world for candidate in controller.last_view_candidates], dtype=np.float64
        ).reshape(-1, 2)
        candidate_scores = np.asarray(
            [candidate.score for candidate in controller.last_view_candidates], dtype=np.float64
        )
        np.savez_compressed(
            self.save_dir / f"stuck_{controller.last_stuck_step:05d}.npz",
            traversable=projection.traversable,
            known=projection.known,
            occupied=projection.occupied,
            clearance=projection.clearance,
            frontier=np.zeros_like(projection.known) if frontier_mask is None else frontier_mask,
            map_origin_xy=projection.origin_xy,
            resolution=projection.resolution,
            robot_position=controller.robot_position(),
            robot_yaw=controller.robot_yaw(),
            path_world=plan_path,
            candidate_points=candidate_points,
            candidate_scores=candidate_scores,
            semantic_points=controller.target.points,
            policy_action=controller.last_action,
        )
