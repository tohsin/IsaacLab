# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Capture reproducible depth figures for the inspection paper.

This utility renders two real Isaac Sim scenes:

* navigation: a robot-height camera viewing a forklift and warehouse obstacles;
* inspection: a close camera viewing the UR10 mount inspection target.

For each scene it saves the metric depth tensor, a 16-bit millimetre depth
image, a grayscale visualization, a Turbo-color visualization, and a paired
grayscale/color panel. For the inspection scene it also saves a binary semantic
mask in which the UR10 mount is green and all non-target pixels are black. The
depth and mask come from the same camera frame. The visualizations are
deterministic transformations of simulator sensor outputs; they are not
generated images.

Example:

    ./isaaclab.sh -p scripts/environments/capture_paper_depth_figures.py \
        --headless --enable_cameras
"""

"""Launch Isaac Sim before importing simulator modules."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--output_dir",
    type=Path,
    default=Path(
        "source/isaaclab_tasks/isaaclab_tasks/direct/robot_inspection/paper/"
        "depth_captures"
    ),
    help="Directory in which to save depth arrays and figures.",
)
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=540)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Simulator imports."""

import numpy as np
import torch
from PIL import Image

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


def _spawn_usd(
    prim_path: str,
    usd_path: str,
    translation: tuple[float, float, float],
    orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    scale: tuple[float, float, float] | None = None,
    semantic_tags: list[tuple[str, str]] | None = None,
) -> None:
    cfg = sim_utils.UsdFileCfg(
        usd_path=usd_path,
        scale=scale,
        semantic_tags=semantic_tags,
    )
    cfg.func(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
    )


def _spawn_box(
    prim_path: str,
    translation: tuple[float, float, float],
    size: tuple[float, float, float],
    color: tuple[float, float, float],
) -> None:
    cfg = sim_utils.CuboidCfg(
        size=size,
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=color,
            roughness=0.75,
        ),
    )
    cfg.func(prim_path, cfg, translation=translation)


def _make_camera(prim_path: str, include_semantics: bool = False) -> Camera:
    parent_path = str(Path(prim_path).parent)
    if not prim_utils.is_prim_path_valid(parent_path):
        prim_utils.create_prim(parent_path, "Xform")
    return Camera(
        cfg=CameraCfg(
            prim_path=prim_path,
            update_period=0.0,
            height=args_cli.height,
            width=args_cli.width,
            data_types=["distance_to_image_plane"]
            + (["semantic_segmentation"] if include_semantics else []),
            colorize_semantic_segmentation=False,
            semantic_filter="inspection_target:ur10_mount",
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 100.0),
            ),
        )
    )


def design_scene() -> tuple[Camera, Camera]:
    # Use the same warehouse asset as the inspection task.
    print("[INFO] Spawning warehouse...", flush=True)
    _spawn_usd(
        "/World/Warehouse",
        f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd",
        translation=(0.0, 0.0, 0.0),
    )

    light_cfg = sim_utils.DomeLightCfg(
        intensity=1800.0,
        color=(0.82, 0.85, 0.90),
    )
    light_cfg.func("/World/DomeLight", light_cfg)

    # Navigation scene: place real USD geometry at several ranges from a
    # robot-height camera.  The boxes stand in for ordinary movable clutter.
    print("[INFO] Spawning forklift...", flush=True)
    _spawn_usd(
        "/World/PaperNavigation/Forklift",
        f"{ISAAC_NUCLEUS_DIR}/Props/Forklift/forklift.usd",
        translation=(0.50, 3.70, 0.0),
        orientation=(0.7071068, 0.0, 0.0, 0.7071068),
        scale=(0.6, 0.6, 0.6),
    )
    print("[INFO] Spawning navigation obstacles...", flush=True)
    _spawn_box(
        "/World/PaperNavigation/NearPallet",
        translation=(-0.15, -2.60, 0.15),
        size=(1.00, 0.65, 0.30),
        color=(0.34, 0.20, 0.08),
    )
    _spawn_box(
        "/World/PaperNavigation/MidCrate",
        translation=(-2.75, 0.80, 0.45),
        size=(0.75, 0.75, 0.90),
        color=(0.55, 0.30, 0.08),
    )
    _spawn_box(
        "/World/PaperNavigation/SafetyBarrier",
        translation=(1.10, 0.65, 0.32),
        size=(1.20, 0.16, 0.64),
        color=(0.90, 0.45, 0.04),
    )

    # Inspection scene: use the exact target asset and scale from the
    # evaluation dataset, but place it in a separate part of the warehouse.
    print("[INFO] Spawning UR10 mount...", flush=True)
    _spawn_usd(
        "/World/PaperInspection/UR10Mount",
        f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ur10_mount.usd",
        translation=(3.25, 7.0, 0.405),
        orientation=(0.7071068, 0.7071068, 0.0, 0.0),
        scale=(3.0, 3.0, 2.2),
        semantic_tags=[("inspection_target", "ur10_mount")],
    )

    print("[INFO] Creating RTX cameras...", flush=True)
    navigation_camera = _make_camera("/World/PaperCameras/NavigationCamera")
    inspection_camera = _make_camera(
        "/World/PaperCameras/InspectionCamera", include_semantics=True
    )
    print("[INFO] Scene definition complete.", flush=True)
    return navigation_camera, inspection_camera


def _finite_depth(depth: np.ndarray, far_m: float) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32).squeeze()
    return np.nan_to_num(depth, nan=far_m, posinf=far_m, neginf=0.0)


def _turbo(values: np.ndarray) -> np.ndarray:
    """Small dependency-free approximation of the Google Turbo colormap."""
    # Polynomial coefficients from the public Turbo colormap approximation.
    x = np.clip(values, 0.0, 1.0)[..., None]
    k = np.array(
        [
            [0.13572138, 4.61539260, -42.66032258, 132.13108234, -152.94239396, 59.28637943],
            [0.09140261, 2.19418839, 4.84296658, -14.18503333, 4.27729857, 2.82956604],
            [0.10667330, 12.64194608, -60.58204836, 110.36276771, -89.90310912, 27.34824973],
        ],
        dtype=np.float32,
    )
    powers = np.concatenate([np.ones_like(x), x, x**2, x**3, x**4, x**5], axis=-1)
    rgb = np.einsum("...k,ck->...c", powers, k)
    return np.uint8(np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5)


def save_depth_products(
    name: str,
    depth_tensor: torch.Tensor,
    near_m: float,
    far_m: float,
    output_dir: Path,
) -> None:
    depth_m = _finite_depth(depth_tensor.detach().cpu().numpy(), far_m)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / f"{name}_depth_m.npy", depth_m)
    depth_mm = np.uint16(np.clip(depth_m * 1000.0, 0.0, 65535.0))
    Image.fromarray(depth_mm, mode="I;16").save(output_dir / f"{name}_depth_mm.png")

    normalized = np.clip((depth_m - near_m) / (far_m - near_m), 0.0, 1.0)
    proximity = 1.0 - normalized  # near=bright/warm, far=dark/cool
    gray = np.uint8(proximity * 255.0 + 0.5)
    color = _turbo(proximity)

    Image.fromarray(gray, mode="L").save(output_dir / f"{name}_depth_gray.png")
    Image.fromarray(color, mode="RGB").save(output_dir / f"{name}_depth_turbo.png")

    gutter = np.full((gray.shape[0], 8, 3), 255, dtype=np.uint8)
    pair = np.concatenate([np.repeat(gray[..., None], 3, axis=-1), gutter, color], axis=1)
    Image.fromarray(pair, mode="RGB").save(output_dir / f"{name}_depth_pair.png")

    finite = depth_m[np.isfinite(depth_m)]
    print(
        f"[INFO] Saved {name}: shape={depth_m.shape}, "
        f"range=[{finite.min():.3f}, {finite.max():.3f}] m, "
        f"visualized over [{near_m:.2f}, {far_m:.2f}] m"
    )


def save_semantic_mask(camera: Camera, output_dir: Path) -> None:
    """Save a green target mask from the inspection camera's semantic IDs."""
    segmentation = camera.data.output.get("semantic_segmentation")
    if segmentation is None:
        raise RuntimeError("Inspection camera did not return semantic segmentation")

    semantic_ids = segmentation[0].detach().cpu().numpy().squeeze()
    # Camera stores one info dictionary per camera prim (unlike TiledCamera,
    # whose info is batched differently).
    camera_info = camera.data.info[0] if isinstance(camera.data.info, list) else camera.data.info
    info = camera_info.get("semantic_segmentation", {}) or {}
    id_to_labels = info.get("idToLabels", {})
    target_ids = [
        int(semantic_id)
        for semantic_id, labels in id_to_labels.items()
        if labels.get("inspection_target") == "ur10_mount"
    ]
    if not target_ids:
        raise RuntimeError(
            "UR10 semantic ID was not found; available labels: "
            f"{id_to_labels}"
        )

    target_mask = np.isin(semantic_ids, target_ids)
    target_pixels = int(target_mask.sum())
    if target_pixels == 0:
        raise RuntimeError("UR10 semantic mask is empty")

    output_dir.mkdir(parents=True, exist_ok=True)
    binary = np.uint8(target_mask) * 255
    green = np.zeros((*target_mask.shape, 3), dtype=np.uint8)
    green[target_mask] = (0, 255, 0)
    Image.fromarray(binary, mode="L").save(
        output_dir / "inspection_semantic_mask_binary.png"
    )
    Image.fromarray(green, mode="RGB").save(
        output_dir / "inspection_semantic_mask_green.png"
    )

    turbo = np.asarray(Image.open(output_dir / "inspection_depth_turbo.png").convert("RGB"))
    gutter = np.full((green.shape[0], 8, 3), 255, dtype=np.uint8)
    pair = np.concatenate((turbo, gutter, green), axis=1)
    Image.fromarray(pair, mode="RGB").save(
        output_dir / "inspection_depth_semantic_pair.png"
    )
    print(
        f"[INFO] Saved inspection semantic mask: IDs={target_ids}, "
        f"target_pixels={target_pixels} ({100.0 * target_pixels / target_mask.size:.2f}%)",
        flush=True,
    )


def main() -> None:
    print("[INFO] Creating simulation context...", flush=True)
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    navigation_camera, inspection_camera = design_scene()

    print("[INFO] Resetting simulation...", flush=True)
    sim.reset()
    print("[INFO] Simulation reset complete.", flush=True)

    device = sim.device
    navigation_camera.set_world_poses_from_view(
        eyes=torch.tensor([[-2.8, -3.6, 0.72]], device=device),
        targets=torch.tensor([[0.25, 3.0, 0.90]], device=device),
    )
    inspection_camera.set_world_poses_from_view(
        eyes=torch.tensor([[2.40, 5.40, 1.25]], device=device),
        targets=torch.tensor([[3.25, 7.0, 1.00]], device=device),
    )

    # Allow USD references, RTX sensors, and materials to finish initializing.
    print("[INFO] Warming up RTX cameras...", flush=True)
    for _ in range(30):
        sim.step()
        navigation_camera.update(dt=sim.get_physics_dt())
        inspection_camera.update(dt=sim.get_physics_dt())

    print("[INFO] Saving depth products...", flush=True)
    save_depth_products(
        "navigation",
        navigation_camera.data.output["distance_to_image_plane"][0],
        near_m=0.5,
        far_m=12.0,
        output_dir=args_cli.output_dir,
    )
    save_depth_products(
        "inspection",
        inspection_camera.data.output["distance_to_image_plane"][0],
        near_m=0.25,
        far_m=5.0,
        output_dir=args_cli.output_dir,
    )
    save_semantic_mask(inspection_camera, args_cli.output_dir)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
