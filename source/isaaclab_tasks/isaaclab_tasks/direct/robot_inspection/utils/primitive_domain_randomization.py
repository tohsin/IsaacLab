"""Shared startup domain randomization for procedural inspection primitives."""

from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np
from pxr import Gf, UsdGeom, Vt

import isaaclab.sim as sim_utils
from isaaclab.sim.utils import bind_visual_material

from .tessellated_primitives import create_tessellated_composite_mesh


_COMPOSITE_PRIMITIVE_TYPES = {
    "tessellated_c_housing",
    "tessellated_u_housing",
    "tessellated_low_arm",
    "tessellated_stepped_block",
    "tessellated_thin_legged_body",
    "tessellated_overhang",
}


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

    footprint_radii: np.ndarray
    """Maximum horizontal distance from the root to any mesh point."""


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
    geometry bank allows environments with identical geometry to share a Warp
    mesh. Composite point arrays change shape but retain fixed topology and
    face counts.
    """
    randomization_cfg = primitive_cfg.get("domain_randomization", {})
    worker_rank = int(os.environ.get("REAL_LOCAL_RANK", os.environ.get("LOCAL_RANK", "0")))
    target_seed = int(seed or 42) + sum(target_name.encode("utf-8")) + 1009 * worker_rank
    rng = np.random.default_rng(target_seed)

    primitive_type = str(primitive_cfg["type"])
    geometry_variants = None
    geometry_meshes = None
    if primitive_type in _COMPOSITE_PRIMITIVE_TYPES:
        (
            scales,
            sizes,
            root_heights,
            size_variant_indices,
            footprint_radii,
            geometry_variants,
        ) = _sample_composite_geometry(
            primitive_cfg=primitive_cfg,
            randomization_cfg=randomization_cfg,
            num_envs=num_envs,
            rng=rng,
        )
        # Build each member of the bounded bank once. Authoring the same point
        # array into several cloned environments is much cheaper than running
        # subdivision again for every environment.
        geometry_meshes = [
            create_tessellated_composite_mesh(variant) for variant in geometry_variants
        ]
    else:
        scales, sizes, root_heights, size_variant_indices = _sample_geometry(
            primitive_cfg=primitive_cfg,
            randomization_cfg=randomization_cfg,
            num_envs=num_envs,
            rng=rng,
        )
        footprint_radii = 0.5 * np.linalg.norm(sizes[:, :2], axis=1)
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

        if geometry_meshes is not None:
            _set_composite_mesh_geometry(
                stage=stage,
                obj_prim_path=obj_prim_path,
                mesh=geometry_meshes[int(size_variant_indices[env_id])],
            )

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
        footprint_radii=footprint_radii,
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

    if primitive_type in ("tessellated_cuboid", "tessellated_shell", "tessellated_shell_side", "tessellated_t_block", "tessellated_t_block_flat"):
        is_flat = primitive_type in ("tessellated_t_block_flat", "tessellated_shell_side")
        base_type = "tessellated_t_block" if primitive_type == "tessellated_t_block_flat" else "tessellated_shell" if primitive_type == "tessellated_shell_side" else primitive_type
        base_size = _vector3(primitive_cfg["size"], f"{base_type} size")
        size_min = _vector3(randomization_cfg.get("size_min", base_size), f"{base_type} size_min")
        size_max = _vector3(randomization_cfg.get("size_max", base_size), f"{base_type} size_max")
        _validate_range(size_min, size_max, f"{base_type} size")
        
        variants = rng.uniform(size_min, size_max, size=(num_variants, 3))
        variants[0] = size_min
        sizes = variants[variant_indices]
        scales = sizes / base_size
        
        if is_flat:
            # The unscaled mesh was rotated around X by -pi/2.
            # Original geometry: X=width, Y=thickness, Z=height
            # Rotated geometry: X=width, Y=height, Z=thickness
            # The sampled 'sizes' are (width, thickness, height).
            # We map the scales to the rotated geometry axes: X->X, Y->Z, Z->Y
            scales = np.column_stack((scales[:, 0], scales[:, 2], scales[:, 1]))
            sizes = np.column_stack((sizes[:, 0], sizes[:, 2], sizes[:, 1]))

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
        
        # If the object is flat, users intuitively expect 'height' to mean world Z height,
        # and 'length' to mean the longitudinal length (geometric height).
        if axis in ("X", "Y"):
            if "length_min" in randomization_cfg or "length_max" in randomization_cfg:
                # 'length' overrides the geometric height
                geom_height_min = float(randomization_cfg.get("length_min", base_height))
                geom_height_max = float(randomization_cfg.get("length_max", base_height))
            else:
                # Fallback to swap: if they provided 'height' but no 'length', 
                # assume they meant world height. But to preserve backward compatibility if 
                # they only provided radius, we'll map radius to world width/height.
                geom_height_min = height_min
                geom_height_max = height_max
            
            if "height_min" in randomization_cfg or "height_max" in randomization_cfg:
                # If they explicitly specify 'height' for a flat object, treat it as World Z height.
                # Since world height = 2 * radius, geometric radius = height / 2.
                radius_min = float(randomization_cfg.get("height_min", base_radius * 2)) / 2.0
                radius_max = float(randomization_cfg.get("height_max", base_radius * 2)) / 2.0
            
            # Assign the resolved geometric height
            height_min = geom_height_min
            height_max = geom_height_max

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


def _sample_composite_geometry(
    *,
    primitive_cfg: dict,
    randomization_cfg: dict,
    num_envs: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Sample a bounded bank of genuine composite-geometry variants."""
    requested_variants = int(randomization_cfg.get("num_size_variants", 1))
    num_variants = max(1, min(requested_variants, num_envs))
    variant_indices = np.arange(num_envs) % num_variants
    rng.shuffle(variant_indices)
    primitive_type = str(primitive_cfg["type"])
    base_cfg = {key: value for key, value in primitive_cfg.items() if key != "domain_randomization"}
    variants: list[dict] = []

    if primitive_type == "tessellated_c_housing":
        size_min, size_max = _vector_range(randomization_cfg, "size", base_cfg["size"])
        opening_min, opening_max = _scalar_range(
            randomization_cfg,
            "opening_width_fraction",
            float(base_cfg.get("opening_width_fraction", 0.55)),
        )
        pocket_min, pocket_max = _scalar_range(
            randomization_cfg,
            "pocket_depth_fraction",
            float(base_cfg.get("pocket_depth_fraction", 0.7)),
        )
        if not 0.0 < opening_min <= opening_max < 1.0:
            raise ValueError("C-housing opening fractions must lie inside (0, 1)")
        if not 0.0 < pocket_min <= pocket_max < 1.0:
            raise ValueError("C-housing pocket-depth fractions must lie inside (0, 1)")
        sampled_sizes = rng.uniform(size_min, size_max, size=(num_variants, 3))
        openings = rng.uniform(opening_min, opening_max, size=num_variants)
        pockets = rng.uniform(pocket_min, pocket_max, size=num_variants)
        for index in range(num_variants):
            variants.append(
                {
                    **base_cfg,
                    "size": tuple(sampled_sizes[index]),
                    "opening_width_fraction": float(openings[index]),
                    "pocket_depth_fraction": float(pockets[index]),
                }
            )

    elif primitive_type == "tessellated_u_housing":
        size_min, size_max = _vector_range(randomization_cfg, "size", base_cfg["size"])
        opening_min, opening_max = _scalar_range(
            randomization_cfg,
            "opening_width_fraction",
            float(base_cfg.get("opening_width_fraction", 0.55)),
        )
        floor_min, floor_max = _scalar_range(
            randomization_cfg,
            "pocket_floor_height",
            float(base_cfg.get("pocket_floor_height", 0.25)),
        )
        if not 0.0 < opening_min <= opening_max < 1.0:
            raise ValueError("U-housing opening fractions must lie inside (0, 1)")
        if not 0.0 < floor_min <= floor_max:
            raise ValueError("U-housing pocket_floor_height must be positive")
        sampled_sizes = rng.uniform(size_min, size_max, size=(num_variants, 3))
        openings = rng.uniform(opening_min, opening_max, size=num_variants)
        floors = rng.uniform(floor_min, floor_max, size=num_variants)
        for index in range(num_variants):
            actual_floor = min(float(floors[index]), float(sampled_sizes[index][2]) - 0.01)
            variants.append(
                {
                    **base_cfg,
                    "size": tuple(sampled_sizes[index]),
                    "opening_width_fraction": float(openings[index]),
                    "pocket_floor_height": actual_floor,
                }
            )

    elif primitive_type == "tessellated_low_arm":
        sampled = {
            name: rng.uniform(*_scalar_range(randomization_cfg, name, float(base_cfg[name])), num_variants)
            for name in (
                "body_radius",
                "body_height",
                "arm_length",
                "arm_width",
                "arm_thickness",
                "arm_clearance",
            )
        }
        for index in range(num_variants):
            body_radius = float(sampled["body_radius"][index])
            arm_width = min(float(sampled["arm_width"][index]), 1.9 * body_radius)
            body_height = float(sampled["body_height"][index])
            arm_clearance = float(sampled["arm_clearance"][index])
            arm_thickness = min(
                float(sampled["arm_thickness"][index]),
                body_height - arm_clearance - 0.05,
            )
            variants.append(
                {
                    **base_cfg,
                    "body_radius": body_radius,
                    "body_height": body_height,
                    "arm_length": float(sampled["arm_length"][index]),
                    "arm_width": arm_width,
                    "arm_thickness": arm_thickness,
                    "arm_clearance": arm_clearance,
                }
            )

    elif primitive_type == "tessellated_stepped_block":
        size_min, size_max = _vector_range(randomization_cfg, "size", base_cfg["size"])
        sampled_sizes = rng.uniform(size_min, size_max, size=(num_variants, 3))
        first_x = rng.uniform(*_scalar_range(randomization_cfg, "first_step_x_fraction", 0.34), num_variants)
        second_x = rng.uniform(*_scalar_range(randomization_cfg, "second_step_x_fraction", 0.68), num_variants)
        # The legacy configuration keys retain "height" in their names, but
        # after the axis flip these fractions shape the horizontal depth axis.
        first_depth = rng.uniform(
            *_scalar_range(randomization_cfg, "low_step_height_fraction", 0.4), num_variants
        )
        second_depth = rng.uniform(
            *_scalar_range(randomization_cfg, "middle_step_height_fraction", 0.7), num_variants
        )
        if np.any(first_x >= second_x) or np.any(first_depth >= second_depth):
            raise ValueError("Stepped-block fraction ranges overlap or are reversed")
        for index in range(num_variants):
            variants.append(
                {
                    **base_cfg,
                    "size": tuple(sampled_sizes[index]),
                    "step_x_fractions": (float(first_x[index]), float(second_x[index])),
                    "step_height_fractions": (
                        float(first_depth[index]),
                        float(second_depth[index]),
                    ),
                }
            )

    elif primitive_type == "tessellated_thin_legged_body":
        parameter_names = (
            "depth",
            "gap_width",
            "left_leg_width",
            "right_leg_width",
            "underside_height",
            "body_thickness",
        )
        sampled = {
            name: rng.uniform(*_scalar_range(randomization_cfg, name, float(base_cfg[name])), num_variants)
            for name in parameter_names
        }
        for index in range(num_variants):
            variants.append(
                {**base_cfg, **{name: float(sampled[name][index]) for name in parameter_names}}
            )

    elif primitive_type == "tessellated_overhang":
        support_min, support_max = _vector_range(
            randomization_cfg, "support_size", base_cfg["support_size"]
        )
        plate_min, plate_max = _vector_range(
            randomization_cfg, "plate_size", base_cfg["plate_size"]
        )
        offset_min, offset_max = _scalar_range(
            randomization_cfg, "offset_fraction", 0.0, positive=False
        )
        if offset_min < -0.95 or offset_max > 0.95:
            raise ValueError("Overhang offset fractions must stay within [-0.95, 0.95]")
        supports = rng.uniform(support_min, support_max, size=(num_variants, 3))
        plates = rng.uniform(plate_min, plate_max, size=(num_variants, 3))
        if np.any(supports[:, :2] >= plates[:, :2]):
            raise ValueError("Every sampled overhang plate must exceed its support in X and Y")
        offset_fractions = rng.uniform(offset_min, offset_max, size=(num_variants, 2))
        offset_fractions[0] = 0.0
        maximum_offset = float(randomization_cfg.get("plate_offset_max", np.inf))
        if maximum_offset <= 0.0:
            raise ValueError(f"plate_offset_max must be positive, got {maximum_offset}")
        for index in range(num_variants):
            clearance = 0.5 * (plates[index, :2] - supports[index, :2])
            plate_offset = np.clip(
                offset_fractions[index] * clearance,
                -maximum_offset,
                maximum_offset,
            )
            variants.append(
                {
                    **base_cfg,
                    "support_size": tuple(supports[index]),
                    "plate_size": tuple(plates[index]),
                    "plate_offset": tuple(plate_offset),
                }
            )
    else:
        raise ValueError(f"Unsupported composite primitive type: {primitive_type!r}")

    # Most banks keep one easy authored reference member. Targets can disable
    # this when every generated mesh must strictly obey the configured bounds.
    if bool(randomization_cfg.get("include_authored_variant", True)):
        variants[0] = base_cfg

    variant_sizes = np.empty((num_variants, 3), dtype=np.float64)
    variant_root_heights = np.empty(num_variants, dtype=np.float64)
    variant_footprint_radii = np.empty(num_variants, dtype=np.float64)
    expected_faces = None
    for index, variant in enumerate(variants):
        mesh = create_tessellated_composite_mesh(variant)
        if expected_faces is None:
            expected_faces = mesh.faces
        elif mesh.faces.shape != expected_faces.shape:
            raise ValueError(
                f"{primitive_type} variants changed topology: "
                f"{mesh.faces.shape} != {expected_faces.shape}"
            )
        bounds = np.asarray(mesh.bounds, dtype=np.float64)
        variant_sizes[index] = bounds[1] - bounds[0]
        variant_root_heights[index] = -bounds[0, 2]
        variant_footprint_radii[index] = np.linalg.norm(mesh.vertices[:, :2], axis=1).max()

    sizes = variant_sizes[variant_indices]
    root_heights = variant_root_heights[variant_indices]
    footprint_radii = variant_footprint_radii[variant_indices]
    scales = np.ones((num_envs, 3), dtype=np.float64)
    return scales, sizes, root_heights, variant_indices, footprint_radii, variants


def _set_composite_mesh_geometry(*, stage, obj_prim_path: str, mesh) -> None:
    """Author one sampled point/topology array onto an already-cloned USD mesh."""
    mesh_prim_path = f"{obj_prim_path}/geometry/mesh"
    mesh_prim = stage.GetPrimAtPath(mesh_prim_path)
    if not mesh_prim.IsValid():
        raise RuntimeError(f"Cannot randomize missing composite mesh: {mesh_prim_path}")
    usd_mesh = UsdGeom.Mesh(mesh_prim)
    points = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
    face_indices = np.ascontiguousarray(mesh.faces.reshape(-1), dtype=np.int32)
    face_counts = np.full(len(mesh.faces), 3, dtype=np.int32)
    usd_mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
    usd_mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(face_indices))
    usd_mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(face_counts))


def _scalar_range(
    randomization_cfg: dict, name: str, default: float, *, positive: bool = True
) -> tuple[float, float]:
    minimum = float(randomization_cfg.get(f"{name}_min", default))
    maximum = float(randomization_cfg.get(f"{name}_max", default))
    if maximum < minimum or (positive and minimum <= 0.0):
        raise ValueError(f"Invalid {name} range: min={minimum}, max={maximum}")
    return minimum, maximum


def _vector_range(
    randomization_cfg: dict, name: str, default
) -> tuple[np.ndarray, np.ndarray]:
    default_vector = _vector3(default, name)
    minimum = _vector3(randomization_cfg.get(f"{name}_min", default_vector), f"{name}_min")
    maximum = _vector3(randomization_cfg.get(f"{name}_max", default_vector), f"{name}_max")
    _validate_range(minimum, maximum, name)
    return minimum, maximum


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
