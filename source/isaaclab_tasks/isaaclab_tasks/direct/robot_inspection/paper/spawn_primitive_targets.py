# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Display every procedural inspection target in the training warehouse.

The meshes are built directly from ``primitive_data_set``, using the same
spawner used by ``InspectionEnv``.  The warehouse is loaded from the task's
``env_parameters``, so ``ISAACLAB_INSPECTION_ENV_*`` overrides are honored in
the same way as training.  The GUI remains open after setup so the viewport can
be adjusted before taking a screenshot.  A deterministic RGB capture is also
written by default.

Examples:

.. code-block:: bash

    # Open the scene and take a screenshot manually from the viewport.
    ./isaaclab.sh -p \
        source/isaaclab_tasks/isaaclab_tasks/direct/robot_inspection/paper/spawn_primitive_targets.py

    # Render the default paper image without opening a window.
    ./isaaclab.sh -p \
        source/isaaclab_tasks/isaaclab_tasks/direct/robot_inspection/paper/spawn_primitive_targets.py \
        --headless --save_path /tmp/primitive_targets.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


_DEFAULT_OUTPUT = Path(__file__).with_name("primitive_inspection_targets.png")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--columns", type=int, default=5, help="Number of targets in each row.")
parser.add_argument("--spacing_x", type=float, default=2.1, help="Horizontal target spacing in metres.")
parser.add_argument("--spacing_y", type=float, default=2.2, help="Row spacing in metres.")
parser.add_argument("--width", type=int, default=1920, help="Saved image width in pixels.")
parser.add_argument("--height", type=int, default=1080, help="Saved image height in pixels.")
parser.add_argument(
    "--save_path",
    type=Path,
    default=_DEFAULT_OUTPUT,
    help="Path for the automatic RGB capture.",
)
parser.add_argument("--no_save", action="store_true", help="Do not make the automatic RGB capture.")
parser.add_argument(
    "--exit_after_capture",
    action="store_true",
    help="Close the GUI after saving instead of leaving the scene open.",
)
parser.add_argument("--warmup_steps", type=int, default=30, help="Render steps before saving the image.")
parser.add_argument(
    "--camera_eye",
    type=float,
    nargs=3,
    default=(-10.0, 5.0, 8.4),
    metavar=("X", "Y", "Z"),
    help="World position of both the viewport and capture camera.",
)
parser.add_argument(
    "--camera_target",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.55),
    metavar=("X", "Y", "Z"),
    help="World-space point viewed by both cameras.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# An RTX camera is used for the reproducible file capture.  Enabling cameras
# is harmless in the interactive case and required for a headless capture.
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Simulator imports must follow AppLauncher construction."""

import math

import numpy as np
import torch
from PIL import Image

import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab_tasks.direct.robot_inspection.configs.config_ import env_parameters
from isaaclab_tasks.direct.robot_inspection.configs.primitive_dataset import primitive_data_set
from isaaclab_tasks.direct.robot_inspection.utils.tessellated_primitives import (
    build_tessellated_primitive_cfg,
    create_tessellated_composite_mesh,
)


# A restrained palette gives adjacent silhouettes enough contrast while
# keeping the result suitable for a paper figure.
_COLORS = (
    (0.16, 0.42, 0.67),
    (0.20, 0.55, 0.69),
    (0.25, 0.62, 0.58),
    (0.38, 0.59, 0.37),
    (0.63, 0.58, 0.25),
    (0.77, 0.48, 0.22),
    (0.73, 0.34, 0.27),
    (0.63, 0.31, 0.48),
    (0.49, 0.34, 0.64),
    (0.33, 0.39, 0.67),
    (0.18, 0.48, 0.62),
    (0.29, 0.58, 0.48),
    (0.58, 0.54, 0.31),
    (0.72, 0.42, 0.25),
    (0.58, 0.35, 0.54),
)

_COMPOSITE_TYPES = {
    "tessellated_c_housing",
    "tessellated_u_housing",
    "tessellated_low_arm",
    "tessellated_stepped_block",
    "tessellated_thin_legged_body",
    "tessellated_overhang",
}


def _root_height(primitive: dict) -> float:
    """Return the translation that places an authored primitive on Z=0."""
    primitive_type = str(primitive["type"])
    axis = str(primitive.get("axis", "Z")).upper()

    if primitive_type in _COMPOSITE_TYPES:
        mesh_cfg = {key: value for key, value in primitive.items() if key != "domain_randomization"}
        return -float(create_tessellated_composite_mesh(mesh_cfg).bounds[0, 2])
    if primitive_type == "tessellated_sphere":
        return float(primitive["radius"])
    if primitive_type == "tessellated_cone" and axis == "Z":
        # The structured upright cone is authored from z=0 to z=height.
        return 0.0
    if primitive_type in ("tessellated_cylinder", "tessellated_cone"):
        return float(primitive["height"]) / 2.0 if axis == "Z" else float(primitive["radius"])
    if primitive_type in ("tessellated_t_block_flat", "tessellated_shell_side"):
        return float(primitive["size"][1]) / 2.0
    return float(primitive["size"][2]) / 2.0


def _grid_position(index: int, count: int) -> tuple[float, float]:
    columns = min(args_cli.columns, count)
    rows = math.ceil(count / columns)
    row, column = divmod(index, columns)
    row_length = min(columns, count - row * columns)
    x = (column - 0.5 * (row_length - 1)) * args_cli.spacing_x
    y = (row - 0.5 * (rows - 1)) * args_cli.spacing_y
    return x, y


def _spawn_targets() -> list[str]:
    target_names = list(primitive_data_set)
    for index, target_name in enumerate(target_names):
        primitive = primitive_data_set[target_name]["primitive"]
        x, y = _grid_position(index, len(target_names))
        color = _COLORS[index % len(_COLORS)]
        # The authored flat cone points along +X, which presents its circular
        # base to the warehouse camera. Turn it around world Z so the tip and
        # side profile remain visible while the cone stays horizontal.
        orientation = (0.0, 0.0, 0.0, 1.0) if target_name == "cone_flat" else (1.0, 0.0, 0.0, 0.0)
        spawn_cfg = build_tessellated_primitive_cfg(
            primitive,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color,
                roughness=0.62,
                metallic=0.0,
            ),
        )
        spawn_cfg.func(
            f"/World/InspectionTargets/{target_name}",
            spawn_cfg,
            translation=(x, y, _root_height(primitive)),
            orientation=orientation,
        )
    return target_names


def _spawn_training_warehouse() -> None:
    """Spawn the same warehouse asset and transform configured for training."""
    warehouse_cfg = sim_utils.UsdFileCfg(
        usd_path=env_parameters.usd_path,
        scale=env_parameters.scale,
    )
    warehouse_cfg.func(
        "/World/TrainingWarehouse",
        warehouse_cfg,
        translation=env_parameters.position,
        orientation=env_parameters.orientation,
    )
    print(f"[INFO] Training warehouse: {env_parameters.usd_path}")


def _make_capture_camera() -> Camera:
    return Camera(
        CameraCfg(
            prim_path="/World/PaperCamera",
            update_period=0.0,
            height=args_cli.height,
            width=args_cli.width,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=17.5,
                focus_distance=15.0,
                horizontal_aperture=22.5,
                clipping_range=(0.1, 100.0),
            ),
        )
    )


def _print_layout(target_names: list[str]) -> None:
    columns = min(args_cli.columns, len(target_names))
    print("[INFO] Primitive layout (front row to back row, left to right):")
    for row_start in range(0, len(target_names), columns):
        row = target_names[row_start : row_start + columns]
        print(f"  row {row_start // columns + 1}: " + " | ".join(row))


def _save_rgb(camera: Camera) -> None:
    save_path = args_cli.save_path.expanduser().resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = camera.data.output["rgb"][0].detach().cpu().numpy()
    rgb = np.asarray(rgb[..., :3], dtype=np.uint8)
    Image.fromarray(rgb, mode="RGB").save(save_path)
    print(f"[INFO] Saved {rgb.shape[1]}x{rgb.shape[0]} paper capture to: {save_path}")


def main() -> None:
    if args_cli.columns < 1:
        raise ValueError("--columns must be at least 1")
    if args_cli.width < 1 or args_cli.height < 1:
        raise ValueError("--width and --height must be positive")
    if args_cli.warmup_steps < 1:
        raise ValueError("--warmup_steps must be at least 1")

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 60.0, device=args_cli.device))
    sim.set_camera_view(eye=args_cli.camera_eye, target=args_cli.camera_target)

    _spawn_training_warehouse()
    # Match the light authored by InspectionEnv after scene cloning.
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    target_names = _spawn_targets()
    capture_camera = _make_capture_camera()

    sim.reset()
    capture_camera.set_world_poses_from_view(
        eyes=torch.tensor([args_cli.camera_eye], device=sim.device, dtype=torch.float32),
        targets=torch.tensor([args_cli.camera_target], device=sim.device, dtype=torch.float32),
    )
    _print_layout(target_names)

    for _ in range(args_cli.warmup_steps):
        sim.step()
        capture_camera.update(sim.get_physics_dt())

    if not args_cli.no_save:
        _save_rgb(capture_camera)

    if not args_cli.headless and not args_cli.exit_after_capture:
        print("[INFO] Scene is ready. Adjust the viewport if desired, then use the viewport screenshot tool.")
        while simulation_app.is_running():
            sim.step()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
