"""Shared startup domain randomization for procedural inspection primitives."""

from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np
from pxr import Gf, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.sim.utils import bind_visual_material


@dataclass(frozen=True)
class PrimitiveRandomizationResult:
    """Per-environment values produced by primitive domain randomization."""

    sizes: np.ndarray
    """Axis-aligned world extents in metres, shaped ``(num_envs, 3)``."""

    root_heights: np.ndarray
    """Root Z positions that place the primitive directly on the floor."""

    colors: np.ndarray
    """Assigned linear RGB colors, shaped ``(num_envs, 3)``."""

    size_variant_indices: np.ndarray
    """Index into the bounded bank of unique Warp-mesh size variants."""


def apply_primitive_domain_randomization(
    *,
    stage,
    target_name: str,
    prim_path_template: str,
    primitive_cfg: dict,
    num_envs: int,
    seed: int | None,
) -> PrimitiveRandomizationResult:
    """Apply fixed size and color randomization before Warp meshes are cached.

    Each environment receives one variant for its entire lifetime. A bounded
    size bank allows environments with identical geometry to share a Warp
    mesh. Scaling changes vertices but not topology or face IDs.
    """
    randomization_cfg = primitive_cfg.get("domain_randomization", {})
    worker_rank = int(os.environ.get("REAL_LOCAL_RANK", os.environ.get("LOCAL_RANK", "0")))
    target_seed = int(seed or 42) + sum(target_name.encode("utf-8")) + 1009 * worker_rank
    rng = np.random.default_rng(target_seed)

    scales, sizes, root_heights, size_variant_indices = _sample_geometry(
        primitive_cfg=primitive_cfg,
        randomization_cfg=randomization_cfg,
        num_envs=num_envs,
        rng=rng,
    )
    material_paths, color_indices, colors = _create_color_materials(
        stage=stage,
        target_name=target_name,
        randomization_cfg=randomization_cfg,
        num_envs=num_envs,
        rng=rng,
    )

    for env_id in range(num_envs):
        obj_prim_path = _resolve_env_prim_path(prim_path_template, env_id)
        obj_prim = stage.GetPrimAtPath(obj_prim_path)
        if not obj_prim.IsValid():
            raise RuntimeError(f"Cannot randomize missing primitive: {obj_prim_path}")

        xformable = UsdGeom.Xformable(obj_prim)
        scale_op = next(
            (op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeScale),
            None,
        )
        if scale_op is None:
            scale_op = xformable.AddScaleOp()
        scale_op.Set(Gf.Vec3d(*scales[env_id].tolist()))

        _set_translation_z(xformable, float(root_heights[env_id]))
        bind_visual_material(
            f"{obj_prim_path}/geometry/mesh",
            material_paths[int(color_indices[env_id])],
        )

    return PrimitiveRandomizationResult(
        sizes=sizes,
        root_heights=root_heights,
        colors=colors,
        size_variant_indices=size_variant_indices,
    )


def _sample_geometry(
    *,
    primitive_cfg: dict,
    randomization_cfg: dict,
    num_envs: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    primitive_type = primitive_cfg["type"]
    if primitive_type == "tessellated_cylinder_flat":
        # Backward-compatible normalization. New configurations use one
        # cylinder type and select the X axis for the flat category.
        primitive_type = "tessellated_cylinder"
        axis = "X"
    else:
        axis = str(primitive_cfg.get("axis", "Z")).upper()

    requested_variants = int(randomization_cfg.get("num_size_variants", 1))
    num_variants = max(1, min(requested_variants, num_envs))
    variant_indices = np.arange(num_envs) % num_variants
    rng.shuffle(variant_indices)

    if primitive_type == "tessellated_cuboid":
        base_size = _vector3(primitive_cfg["size"], "cuboid size")
        size_min = _vector3(randomization_cfg.get("size_min", base_size), "cuboid size_min")
        size_max = _vector3(randomization_cfg.get("size_max", base_size), "cuboid size_max")
        _validate_range(size_min, size_max, "cuboid size")
        variants = rng.uniform(size_min, size_max, size=(num_variants, 3))
        variants[0] = size_min
        sizes = variants[variant_indices]
        scales = sizes / base_size
        root_heights = sizes[:, 2] / 2.0
        return scales, sizes, root_heights, variant_indices

    if primitive_type == "tessellated_sphere":
        base_radius = float(primitive_cfg.get("radius", 0.4))
        radius_min = float(randomization_cfg.get("radius_min", base_radius))
        radius_max = float(randomization_cfg.get("radius_max", base_radius))
        _validate_range(radius_min, radius_max, "sphere radius")
        variants = rng.uniform(radius_min, radius_max, size=num_variants)
        variants[0] = radius_min
        radii = variants[variant_indices]
        scales = np.repeat((radii / base_radius)[:, None], 3, axis=1)
        sizes = np.repeat((2.0 * radii)[:, None], 3, axis=1)
        return scales, sizes, radii, variant_indices

    if primitive_type in ("tessellated_cylinder", "tessellated_cone"):
        if axis not in ("X", "Y", "Z"):
            raise ValueError(f"Unsupported {primitive_type} axis: {axis!r}")
        base_radius = float(primitive_cfg.get("radius", 0.4))
        base_height = float(primitive_cfg.get("height", 0.8))
        radius_min = float(randomization_cfg.get("radius_min", base_radius))
        radius_max = float(randomization_cfg.get("radius_max", base_radius))
        height_min = float(randomization_cfg.get("height_min", base_height))
        height_max = float(randomization_cfg.get("height_max", base_height))
        _validate_range(radius_min, radius_max, f"{primitive_type} radius")
        _validate_range(height_min, height_max, f"{primitive_type} height")

        # Radius and height belong to the same variant. This guarantees that
        # num_size_variants is the actual upper bound on unique Warp meshes.
        variants = np.column_stack(
            (
                rng.uniform(radius_min, radius_max, size=num_variants),
                rng.uniform(height_min, height_max, size=num_variants),
            )
        )
        variants[0] = (radius_min, height_min)
        radii = variants[variant_indices, 0]
        heights = variants[variant_indices, 1]
        radius_scale = radii / base_radius
        height_scale = heights / base_height

        if axis == "X":
            scales = np.column_stack((height_scale, radius_scale, radius_scale))
            sizes = np.column_stack((heights, 2.0 * radii, 2.0 * radii))
        elif axis == "Y":
            scales = np.column_stack((radius_scale, height_scale, radius_scale))
            sizes = np.column_stack((2.0 * radii, heights, 2.0 * radii))
        else:
            scales = np.column_stack((radius_scale, radius_scale, height_scale))
            sizes = np.column_stack((2.0 * radii, 2.0 * radii, heights))

        if primitive_type == "tessellated_cone" and axis == "Z":
            # trimesh cones use local Z bounds [0, height], so the root is
            # already on the base. This prevents an initial airborne frame.
            root_heights = np.zeros(num_envs, dtype=np.float64)
        else:
            root_heights = sizes[:, 2] / 2.0
        return scales, sizes, root_heights, variant_indices

    raise ValueError(f"Unsupported primitive domain-randomization type: {primitive_type!r}")


def _create_color_materials(
    *,
    stage,
    target_name: str,
    randomization_cfg: dict,
    num_envs: int,
    rng: np.random.Generator,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    if "colors" in randomization_cfg:
        color_variants = np.asarray(randomization_cfg["colors"], dtype=np.float64)
        if color_variants.ndim != 2 or color_variants.shape[1] != 3 or len(color_variants) == 0:
            raise ValueError("Domain-randomization colors must be a non-empty RGB list")
    else:
        num_colors = max(1, min(int(randomization_cfg.get("num_color_variants", num_envs)), num_envs))
        color_min = _vector3(randomization_cfg.get("color_min", (0.1, 0.1, 0.1)), "color_min")
        color_max = _vector3(randomization_cfg.get("color_max", (0.9, 0.9, 0.9)), "color_max")
        if np.any(color_min < 0.0) or np.any(color_max < color_min):
            raise ValueError(f"Invalid color range: min={color_min}, max={color_max}")
        color_variants = rng.uniform(color_min, color_max, size=(num_colors, 3))

    if np.any(color_variants < 0.0) or np.any(color_variants > 1.0):
        raise ValueError("RGB values must be within [0, 1]")

    material_paths = []
    for color_idx, color in enumerate(color_variants):
        material_path = f"/World/Looks/{target_name}_color_{color_idx}"
        if not stage.GetPrimAtPath(material_path).IsValid():
            material_cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(color.tolist()))
            material_cfg.func(material_path, material_cfg)
        material_paths.append(material_path)

    color_indices = np.arange(num_envs) % len(color_variants)
    rng.shuffle(color_indices)
    return material_paths, color_indices, color_variants[color_indices]


def _resolve_env_prim_path(prim_path_template: str, env_id: int) -> str:
    if "env_.*" in prim_path_template:
        return prim_path_template.replace("env_.*", f"env_{env_id}")
    return f"{prim_path_template}_{env_id}"


def _set_translation_z(xformable: UsdGeom.Xformable, z_value: float) -> None:
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() != UsdGeom.XformOp.TypeTranslate:
            continue
        translation = op.Get()
        try:
            op.Set(Gf.Vec3d(float(translation[0]), float(translation[1]), z_value))
        except Exception:
            op.Set(Gf.Vec3f(float(translation[0]), float(translation[1]), z_value))
        return
    xformable.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, z_value))


def _vector3(value, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError(f"{name} must contain exactly three values, got {value!r}")
    return vector


def _validate_range(minimum, maximum, name: str) -> None:
    if np.any(np.asarray(minimum) <= 0.0) or np.any(np.asarray(maximum) < np.asarray(minimum)):
        raise ValueError(f"Invalid {name} range: min={minimum}, max={maximum}")
