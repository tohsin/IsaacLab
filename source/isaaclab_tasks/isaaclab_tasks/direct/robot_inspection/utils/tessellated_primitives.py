"""Task-local mesh primitives with controllable face density.

The stock :class:`isaaclab.sim.MeshCuboidCfg` creates a box with twelve
triangles.  That is sufficient for rendering and collision, but too coarse for
an inspection reward that treats triangle IDs as surface samples.  The
spawner below recursively subdivides those triangles while leaving the shape
of the cuboid unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING

import numpy as np
import trimesh
import isaacsim.core.utils.prims as prim_utils
from pxr import Usd, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.meshes.meshes import _spawn_mesh_geom_from_mesh
from isaaclab.sim.utils import clone
from isaaclab.utils import configclass


@clone
def spawn_tessellated_cuboid(
    prim_path: str,
    cfg: "TessellatedCuboidCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn a cuboid whose planar surfaces contain many triangle faces.

    A normal cuboid starts with 12 triangles.  Every subdivision splits each
    triangle into four, so the final count is ``12 * 4**subdivisions``.
    Subdivision only adds coplanar vertices and therefore does not change the
    cuboid's physical shape.
    """
    if cfg.subdivisions < 0:
        raise ValueError(f"subdivisions must be non-negative, got {cfg.subdivisions}")

    mesh = trimesh.creation.box(extents=cfg.size)
    for _ in range(cfg.subdivisions):
        vertices, faces = trimesh.remesh.subdivide(mesh.vertices, mesh.faces)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    _spawn_mesh_geom_from_mesh(prim_path, cfg, mesh, translation, orientation)
    return prim_utils.get_prim_at_path(prim_path)


@configclass
class TessellatedCuboidCfg(sim_utils.MeshCuboidCfg):
    """Configuration for a cuboid with explicitly subdivided triangle faces."""

    func: Callable = spawn_tessellated_cuboid
    subdivisions: int = 3
    """Recursive triangle subdivisions. Three produces 768 triangles."""


@clone
def spawn_tessellated_shell(
    prim_path: str,
    cfg: "TessellatedShellCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn a shell (a box with a rectangular pocket) whose surfaces are highly tessellated."""
    if cfg.subdivisions < 0:
        raise ValueError(f"subdivisions must be non-negative, got {cfg.subdivisions}")

    W, T, H = cfg.size
    wall = cfg.wall_thickness
    
    if wall * 2 >= W or wall * 2 >= T or wall >= H:
        raise ValueError(f"wall_thickness {wall} is too large for size {cfg.size}")

    w, t, h = W - 2*wall, T - 2*wall, -H/2.0 + wall

    vertices = np.array([
        # Outer box
        [-W/2, -T/2, -H/2], [W/2, -T/2, -H/2], [W/2, T/2, -H/2], [-W/2, T/2, -H/2],  # 0,1,2,3 (Bottom)
        [-W/2, -T/2,  H/2], [W/2, -T/2,  H/2], [W/2, T/2,  H/2], [-W/2, T/2,  H/2],  # 4,5,6,7 (Top)
        # Inner pocket
        [-w/2, -t/2, h], [w/2, -t/2, h], [w/2, t/2, h], [-w/2, t/2, h],  # 8,9,10,11 (Inner bottom)
        [-w/2, -t/2, H/2], [w/2, -t/2, H/2], [w/2, t/2, H/2], [-w/2, t/2, H/2],  # 12,13,14,15 (Inner top)
    ])

    faces = [
        # Outer Bottom (normal -Z)
        (0, 3, 2), (0, 2, 1),
        # Outer Front (normal -Y)
        (0, 1, 5), (0, 5, 4),
        # Outer Right (normal +X)
        (1, 2, 6), (1, 6, 5),
        # Outer Back (normal +Y)
        (2, 3, 7), (2, 7, 6),
        # Outer Left (normal -X)
        (3, 0, 4), (3, 4, 7),
        
        # Top Rims (normal +Z)
        (4, 5, 13), (4, 13, 12),  # Front rim
        (5, 6, 14), (5, 14, 13),  # Right rim
        (6, 7, 15), (6, 15, 14),  # Back rim
        (7, 4, 12), (7, 12, 15),  # Left rim
        
        # Inner Bottom (normal +Z)
        (8, 9, 10), (8, 10, 11),
        
        # Inner Front (normal +Y)
        (12, 13, 9), (12, 9, 8),
        # Inner Right (normal -X)
        (13, 14, 10), (13, 10, 9),
        # Inner Back (normal -Y)
        (14, 15, 11), (14, 11, 10),
        # Inner Left (normal +X)
        (15, 12, 8), (15, 8, 11),
    ]

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    _orient_axial_mesh(mesh, cfg.axis)
    
    for _ in range(cfg.subdivisions):
        v, f = trimesh.remesh.subdivide(mesh.vertices, mesh.faces)
        mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)

    _spawn_mesh_geom_from_mesh(prim_path, cfg, mesh, translation, orientation)
    return prim_utils.get_prim_at_path(prim_path)


@configclass
class TessellatedShellCfg(sim_utils.MeshCuboidCfg):
    """Configuration for a shell with explicitly subdivided triangle faces."""

    func: Callable = spawn_tessellated_shell
    subdivisions: int = 3
    """Recursive triangle subdivisions. Three produces 1792 triangles."""
    wall_thickness: float = 0.1
    """Thickness of the walls separating the inner pocket from the outer bounds."""
    axis: str = "Z"
    """The upward-pointing axis (e.g. Z for upright, Y for sideways)."""


@clone
def spawn_tessellated_t_block(
    prim_path: str,
    cfg: "TessellatedTBlockCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn a T-block whose surfaces contain many triangle faces."""
    if cfg.subdivisions < 0:
        raise ValueError(f"subdivisions must be non-negative, got {cfg.subdivisions}")

    width, thickness, height = cfg.size
    bar_h = height * cfg.bar_height_fraction
    stem_w = width * cfg.stem_width_fraction
    
    z_bottom = -height / 2.0
    z_mid = height / 2.0 - bar_h
    z_top = height / 2.0
    
    x_left_bar = -width / 2.0
    x_left_stem = -stem_w / 2.0
    x_right_stem = stem_w / 2.0
    x_right_bar = width / 2.0
    
    y_front = -thickness / 2.0
    y_back = thickness / 2.0
    
    profile = [
        (x_left_stem, z_bottom),  # 0
        (x_right_stem, z_bottom), # 1
        (x_right_stem, z_mid),    # 2
        (x_right_bar, z_mid),     # 3
        (x_right_bar, z_top),     # 4
        (x_left_bar, z_top),      # 5
        (x_left_bar, z_mid),      # 6
        (x_left_stem, z_mid),     # 7
    ]
    
    vertices = []
    for y in [y_front, y_back]:
        for x, z in profile:
            vertices.append((x, y, z))
            
    faces = [
        # Front faces (normal -Y)
        (0, 1, 2), (0, 2, 7), (6, 7, 5), (7, 2, 5), (2, 4, 5), (2, 3, 4),
        # Back faces (normal +Y)
        (8, 10, 9), (8, 15, 10), (14, 13, 15), (15, 13, 10), (10, 13, 12), (10, 12, 11)
    ]
    
    # Side faces
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 0)]
    for a, b in edges:
        faces.extend([(a, a + 8, b + 8), (a, b + 8, b)])
        
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)
    _orient_axial_mesh(mesh, cfg.axis)
    
    for _ in range(cfg.subdivisions):
        v, f = trimesh.remesh.subdivide(mesh.vertices, mesh.faces)
        mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)

    _spawn_mesh_geom_from_mesh(prim_path, cfg, mesh, translation, orientation)
    return prim_utils.get_prim_at_path(prim_path)


@configclass
class TessellatedTBlockCfg(sim_utils.MeshCuboidCfg):
    """Configuration for a T-block with explicitly subdivided triangle faces."""

    func: Callable = spawn_tessellated_t_block
    subdivisions: int = 3
    """Recursive triangle subdivisions. Three produces 1536 triangles."""
    bar_height_fraction: float = 0.3
    """Fraction of total height occupied by the horizontal bar."""
    stem_width_fraction: float = 0.4
    """Fraction of total width occupied by the vertical stem."""
    axis: str = "Z"
    """The upward-pointing axis (e.g. Z for upright, Y for flat)."""


@clone
def spawn_tessellated_composite(
    prim_path: str,
    cfg: "TessellatedCompositeCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn one of the fixed-topology composite inspection targets."""
    mesh = create_tessellated_composite_mesh(cfg.primitive_cfg)
    _spawn_mesh_geom_from_mesh(prim_path, cfg, mesh, translation, orientation)

    # These meshes are deliberately concave.  A convex hull would close the
    # U-housing pocket, leg gap, and overhang clearance that they are intended
    # to teach.  Convex decomposition also remains valid if a run makes the
    # target dynamic instead of using the normal kinematic configuration.
    mesh_prim = prim_utils.get_prim_at_path(f"{prim_path}/geometry/mesh")
    collision_api = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
    collision_api.GetApproximationAttr().Set(cfg.collision_approximation)
    return prim_utils.get_prim_at_path(prim_path)


@configclass
class TessellatedCompositeCfg(sim_utils.MeshCfg):
    """Configuration for a randomized, concave composite target."""

    func: Callable = spawn_tessellated_composite
    primitive_cfg: dict = MISSING
    """Shape parameters passed to :func:`create_tessellated_composite_mesh`."""
    collision_approximation: str = "convexDecomposition"
    """PhysX approximation that preserves the important concavities."""


@clone
def spawn_tessellated_sphere(
    prim_path: str,
    cfg: "TessellatedSphereCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn a sphere whose surface contains many triangle faces.

    An icosphere is used to ensure relatively uniform triangle sizes.
    subdivisions=3 produces 1280 triangles.
    """
    if cfg.subdivisions < 0:
        raise ValueError(f"subdivisions must be non-negative, got {cfg.subdivisions}")

    mesh = trimesh.creation.icosphere(subdivisions=cfg.subdivisions, radius=cfg.radius)

    _spawn_mesh_geom_from_mesh(prim_path, cfg, mesh, translation, orientation)
    return prim_utils.get_prim_at_path(prim_path)


@configclass
class TessellatedSphereCfg(sim_utils.MeshSphereCfg):
    """Configuration for a sphere with explicitly subdivided triangle faces."""

    func: Callable = spawn_tessellated_sphere
    subdivisions: int = 3
    """Recursive triangle subdivisions. Three produces 1280 triangles."""


@clone
def spawn_tessellated_cylinder(
    prim_path: str,
    cfg: "TessellatedCylinderCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn a cylinder with structured circumferential and axial faces."""
    mesh = _create_structured_cylinder_mesh(
        radius=cfg.radius,
        height=cfg.height,
        angular_segments=cfg.angular_segments,
        height_segments=cfg.height_segments,
        cap_radial_segments=cfg.cap_radial_segments,
    )
    _orient_axial_mesh(mesh, cfg.axis)

    _spawn_mesh_geom_from_mesh(prim_path, cfg, mesh, translation, orientation)
    return prim_utils.get_prim_at_path(prim_path)


@configclass
class TessellatedCylinderCfg(sim_utils.MeshCylinderCfg):
    """Configuration for a cylinder with a structured surface grid."""

    func: Callable = spawn_tessellated_cylinder
    angular_segments: int = 32
    """Segments around the circumference."""
    height_segments: int = 16
    """Segments along the cylinder axis."""
    cap_radial_segments: int = 8
    """Concentric segments from each cap center to its rim."""


@clone
def spawn_tessellated_cone(
    prim_path: str,
    cfg: "TessellatedConeCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn a cone with structured circumferential and height faces."""
    mesh = _create_structured_cone_mesh(
        radius=cfg.radius,
        height=cfg.height,
        angular_segments=cfg.angular_segments,
        height_segments=cfg.height_segments,
        cap_radial_segments=cfg.cap_radial_segments,
    )
    _orient_axial_mesh(mesh, cfg.axis)

    _spawn_mesh_geom_from_mesh(prim_path, cfg, mesh, translation, orientation)
    return prim_utils.get_prim_at_path(prim_path)


@configclass
class TessellatedConeCfg(sim_utils.MeshConeCfg):
    """Configuration for a cone with a structured surface grid."""

    func: Callable = spawn_tessellated_cone
    angular_segments: int = 32
    """Segments around the circumference."""
    height_segments: int = 16
    """Segments from the base to the apex."""
    cap_radial_segments: int = 8
    """Concentric segments from the base center to its rim."""


def build_tessellated_primitive_cfg(primitive_cfg: dict, **spawn_kwargs):
    """Build the appropriate Isaac Lab spawner config from one data entry."""
    primitive_type = primitive_cfg["type"]
    if primitive_type == "tessellated_cuboid":
        return TessellatedCuboidCfg(
            size=tuple(primitive_cfg["size"]),
            subdivisions=int(primitive_cfg.get("subdivisions", 3)),
            **spawn_kwargs,
        )
    if primitive_type in ("tessellated_shell", "tessellated_shell_side"):
        axis = "Y" if primitive_type == "tessellated_shell_side" else primitive_cfg.get("axis", "Z")
        return TessellatedShellCfg(
            size=tuple(primitive_cfg["size"]),
            subdivisions=int(primitive_cfg.get("subdivisions", 3)),
            wall_thickness=float(primitive_cfg.get("wall_thickness", 0.1)),
            axis=str(axis).upper(),
            **spawn_kwargs,
        )
    if primitive_type in ("tessellated_t_block", "tessellated_t_block_flat"):
        axis = "Y" if primitive_type == "tessellated_t_block_flat" else primitive_cfg.get("axis", "Z")
        return TessellatedTBlockCfg(
            size=tuple(primitive_cfg["size"]),
            subdivisions=int(primitive_cfg.get("subdivisions", 3)),
            bar_height_fraction=float(primitive_cfg.get("bar_height_fraction", 0.3)),
            stem_width_fraction=float(primitive_cfg.get("stem_width_fraction", 0.4)),
            axis=str(axis).upper(),
            **spawn_kwargs,
        )
    if primitive_type in (
        "tessellated_c_housing",
        "tessellated_u_housing",
        "tessellated_low_arm",
        "tessellated_stepped_block",
        "tessellated_thin_legged_body",
        "tessellated_overhang",
    ):
        return TessellatedCompositeCfg(
            primitive_cfg={
                key: value for key, value in primitive_cfg.items() if key != "domain_randomization"
            },
            collision_approximation=str(
                primitive_cfg.get("collision_approximation", "convexDecomposition")
            ),
            **spawn_kwargs,
        )
    if primitive_type == "tessellated_sphere":
        return TessellatedSphereCfg(
            radius=float(primitive_cfg.get("radius", 0.5)),
            subdivisions=int(primitive_cfg.get("subdivisions", 3)),
            **spawn_kwargs,
        )
    if primitive_type in ("tessellated_cylinder", "tessellated_cylinder_flat"):
        axis = "X" if primitive_type == "tessellated_cylinder_flat" else primitive_cfg.get("axis", "Z")
        angular_segments, height_segments, cap_radial_segments = _structured_segment_counts(primitive_cfg)
        return TessellatedCylinderCfg(
            radius=float(primitive_cfg.get("radius", 0.5)),
            height=float(primitive_cfg.get("height", 1.0)),
            axis=str(axis).upper(),
            angular_segments=angular_segments,
            height_segments=height_segments,
            cap_radial_segments=cap_radial_segments,
            **spawn_kwargs,
        )
    if primitive_type == "tessellated_cone":
        angular_segments, height_segments, cap_radial_segments = _structured_segment_counts(primitive_cfg)
        return TessellatedConeCfg(
            radius=float(primitive_cfg.get("radius", 0.5)),
            height=float(primitive_cfg.get("height", 1.0)),
            axis=str(primitive_cfg.get("axis", "Z")).upper(),
            angular_segments=angular_segments,
            height_segments=height_segments,
            cap_radial_segments=cap_radial_segments,
            **spawn_kwargs,
        )
    raise ValueError(f"Unsupported inspection primitive configuration: {primitive_cfg}")


def create_tessellated_composite_mesh(primitive_cfg: dict) -> trimesh.Trimesh:
    """Create a composite mesh from absolute, per-variant dimensions.

    Every family keeps a fixed topology while its dimensions change.  This is
    important because face IDs and the coverage denominator are fixed per
    target, while a bounded bank of point arrays can still represent genuinely
    different pockets, gaps, offsets, and appendages.
    """
    primitive_type = str(primitive_cfg["type"])
    subdivisions = int(primitive_cfg.get("subdivisions", 3))
    if subdivisions < 0:
        raise ValueError(f"subdivisions must be non-negative, got {subdivisions}")

    if primitive_type == "tessellated_c_housing":
        mesh = _create_c_housing_mesh(
            size=tuple(primitive_cfg["size"]),
            opening_width_fraction=float(primitive_cfg.get("opening_width_fraction", 0.55)),
            pocket_depth_fraction=float(primitive_cfg.get("pocket_depth_fraction", 0.7)),
        )
    elif primitive_type == "tessellated_u_housing":
        mesh = _create_u_housing_mesh(
            size=tuple(primitive_cfg["size"]),
            opening_width_fraction=float(primitive_cfg.get("opening_width_fraction", 0.55)),
            pocket_floor_height=float(primitive_cfg.get("pocket_floor_height", 0.25)),
        )
    elif primitive_type == "tessellated_low_arm":
        mesh = _create_low_arm_mesh(
            body_radius=float(primitive_cfg.get("body_radius", 0.4)),
            body_height=float(primitive_cfg.get("body_height", 1.0)),
            arm_length=float(primitive_cfg.get("arm_length", 0.65)),
            arm_width=float(primitive_cfg.get("arm_width", 0.2)),
            arm_thickness=float(primitive_cfg.get("arm_thickness", 0.16)),
            arm_clearance=float(primitive_cfg.get("arm_clearance", 0.08)),
            angular_segments=int(primitive_cfg.get("angular_segments", 32)),
        )
    elif primitive_type == "tessellated_stepped_block":
        mesh = _create_stepped_block_mesh(
            size=tuple(primitive_cfg["size"]),
            step_x_fractions=tuple(primitive_cfg.get("step_x_fractions", (0.34, 0.68))),
            step_height_fractions=tuple(
                primitive_cfg.get("step_height_fractions", (0.4, 0.7))
            ),
        )
    elif primitive_type == "tessellated_thin_legged_body":
        mesh = _create_thin_legged_body_mesh(
            depth=float(primitive_cfg.get("depth", 0.8)),
            gap_width=float(primitive_cfg.get("gap_width", 0.55)),
            left_leg_width=float(primitive_cfg.get("left_leg_width", 0.18)),
            right_leg_width=float(primitive_cfg.get("right_leg_width", 0.18)),
            underside_height=float(primitive_cfg.get("underside_height", 1.05)),
            body_thickness=float(primitive_cfg.get("body_thickness", 0.3)),
        )
    elif primitive_type == "tessellated_overhang":
        mesh = _create_overhang_mesh(
            support_size=tuple(primitive_cfg.get("support_size", (0.5, 0.5, 1.1))),
            plate_size=tuple(primitive_cfg.get("plate_size", (1.2, 1.2, 0.22))),
            plate_offset=tuple(primitive_cfg.get("plate_offset", (0.0, 0.0))),
        )
    else:
        raise ValueError(f"Unsupported composite primitive type: {primitive_type!r}")

    for _ in range(subdivisions):
        vertices, faces = trimesh.remesh.subdivide(mesh.vertices, mesh.faces)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return mesh


def _triangulate_simple_polygon(points: list[tuple[float, float]]) -> list[tuple[int, int, int]]:
    """Triangulate a counter-clockwise simple polygon with deterministic ear clipping."""
    coordinates = np.asarray(points, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[0] < 3 or coordinates.shape[1] != 2:
        raise ValueError("A polygon needs at least three 2-D points")
    signed_area = 0.5 * np.sum(
        coordinates[:, 0] * np.roll(coordinates[:, 1], -1)
        - np.roll(coordinates[:, 0], -1) * coordinates[:, 1]
    )
    if signed_area <= 0.0:
        raise ValueError("Composite profiles must be counter-clockwise")

    def cross(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        return float(np.cross(b - a, c - a))

    def inside_triangle(point: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
        epsilon = 1.0e-10
        return (
            cross(a, b, point) >= -epsilon
            and cross(b, c, point) >= -epsilon
            and cross(c, a, point) >= -epsilon
        )

    remaining = list(range(len(points)))
    triangles: list[tuple[int, int, int]] = []
    while len(remaining) > 3:
        for position, current in enumerate(remaining):
            previous = remaining[position - 1]
            following = remaining[(position + 1) % len(remaining)]
            if cross(coordinates[previous], coordinates[current], coordinates[following]) <= 1.0e-10:
                continue
            if any(
                inside_triangle(
                    coordinates[candidate],
                    coordinates[previous],
                    coordinates[current],
                    coordinates[following],
                )
                for candidate in remaining
                if candidate not in (previous, current, following)
            ):
                continue
            triangles.append((previous, current, following))
            del remaining[position]
            break
        else:
            raise ValueError(f"Could not triangulate composite profile: {points!r}")
    triangles.append(tuple(remaining))
    return triangles


def _create_xy_extrusion_mesh(
    profile: list[tuple[float, float]], height: float
) -> trimesh.Trimesh:
    """Extrude an XY profile symmetrically along Z."""
    if height <= 0.0:
        raise ValueError(f"Extrusion height must be positive, got {height}")
    cap_faces = _triangulate_simple_polygon(profile)
    count = len(profile)
    vertices = [(x, y, -height / 2.0) for x, y in profile]
    vertices.extend((x, y, height / 2.0) for x, y in profile)
    faces = [tuple(reversed(face)) for face in cap_faces]
    faces.extend(tuple(index + count for index in face) for face in cap_faces)
    for index in range(count):
        following = (index + 1) % count
        faces.extend(
            ((index, following, following + count), (index, following + count, index + count))
        )
    return trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)


def _create_xz_extrusion_mesh(
    profile: list[tuple[float, float]], depth: float
) -> trimesh.Trimesh:
    """Extrude an XZ profile symmetrically along Y."""
    if depth <= 0.0:
        raise ValueError(f"Extrusion depth must be positive, got {depth}")
    cap_faces = _triangulate_simple_polygon(profile)
    count = len(profile)
    vertices = [(x, -depth / 2.0, z) for x, z in profile]
    vertices.extend((x, depth / 2.0, z) for x, z in profile)
    # A CCW XZ cap points toward -Y on the front and must be reversed on the back.
    faces = list(cap_faces)
    faces.extend(tuple(reversed(tuple(index + count for index in face))) for face in cap_faces)
    for index in range(count):
        following = (index + 1) % count
        faces.extend(
            ((index, index + count, following + count), (index, following + count, following))
        )
    return trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)


def _create_c_housing_mesh(
    size: tuple[float, float, float],
    opening_width_fraction: float,
    pocket_depth_fraction: float,
) -> trimesh.Trimesh:
    width, depth, height = map(float, size)
    if min(width, depth, height) <= 0.0:
        raise ValueError(f"C-housing size must be positive, got {size}")
    if not 0.0 < opening_width_fraction < 1.0:
        raise ValueError("opening_width_fraction must lie in (0, 1)")
    if not 0.0 < pocket_depth_fraction < 1.0:
        raise ValueError("pocket_depth_fraction must lie in (0, 1)")
    opening_half_width = 0.5 * width * opening_width_fraction
    front = -0.5 * depth
    back = 0.5 * depth
    pocket_back = front + depth * pocket_depth_fraction
    profile = [
        (-0.5 * width, front),
        (-opening_half_width, front),
        (-opening_half_width, pocket_back),
        (opening_half_width, pocket_back),
        (opening_half_width, front),
        (0.5 * width, front),
        (0.5 * width, back),
        (-0.5 * width, back),
    ]
    return _create_xy_extrusion_mesh(profile, height)


def _create_u_housing_mesh(
    size: tuple[float, float, float],
    opening_width_fraction: float,
    pocket_floor_height: float,
) -> trimesh.Trimesh:
    width, depth, height = map(float, size)
    if min(width, depth, height) <= 0.0:
        raise ValueError(f"U-housing size must be positive, got {size}")
    if not 0.0 < opening_width_fraction < 1.0:
        raise ValueError("opening_width_fraction must lie in (0, 1)")
    if not 0.0 < pocket_floor_height < height:
        raise ValueError(f"pocket_floor_height ({pocket_floor_height}) must be strictly between 0 and height ({height})")
    opening_half_width = 0.5 * width * opening_width_fraction
    bottom = -0.5 * height
    top = 0.5 * height
    pocket_bottom = bottom + pocket_floor_height
    profile = [
        (0.5 * width, bottom),
        (0.5 * width, top),
        (opening_half_width, top),
        (opening_half_width, pocket_bottom),
        (-opening_half_width, pocket_bottom),
        (-opening_half_width, top),
        (-0.5 * width, top),
        (-0.5 * width, bottom),
    ]
    return _create_xz_extrusion_mesh(profile, depth)


def _create_stepped_block_mesh(
    size: tuple[float, float, float],
    step_x_fractions: tuple[float, float],
    step_height_fractions: tuple[float, float],
) -> trimesh.Trimesh:
    """Create a vertically extruded prism with a stepped XY footprint.

    ``size`` always means world-space ``(width, depth, height)``. The
    ``step_height_fractions`` name is retained for configuration compatibility,
    but those fractions now locate turns along the horizontal depth axis.
    """
    width, depth, height = map(float, size)
    first_x, second_x = map(float, step_x_fractions)
    first_depth, second_depth = map(float, step_height_fractions)
    if min(width, depth, height) <= 0.0:
        raise ValueError(f"Stepped-block size must be positive, got {size}")
    if not 0.0 < first_x < second_x < 1.0:
        raise ValueError("step_x_fractions must be strictly increasing inside (0, 1)")
    if not 0.0 < first_depth < second_depth < 1.0:
        raise ValueError("step_height_fractions must be strictly increasing inside (0, 1)")
    x0 = -0.5 * width
    x1 = x0 + width * first_x
    x2 = x0 + width * second_x
    x3 = 0.5 * width
    y0 = -0.5 * depth
    y1 = y0 + depth * first_depth
    y2 = y0 + depth * second_depth
    y3 = 0.5 * depth
    profile = [
        (x0, y0),
        (x3, y0),
        (x3, y3),
        (x2, y3),
        (x2, y2),
        (x1, y2),
        (x1, y1),
        (x0, y1),
    ]
    return _create_xy_extrusion_mesh(profile, height)


def _create_thin_legged_body_mesh(
    depth: float,
    gap_width: float,
    left_leg_width: float,
    right_leg_width: float,
    underside_height: float,
    body_thickness: float,
) -> trimesh.Trimesh:
    dimensions = (
        depth,
        gap_width,
        left_leg_width,
        right_leg_width,
        underside_height,
        body_thickness,
    )
    if min(dimensions) <= 0.0:
        raise ValueError(f"Thin-legged-body dimensions must be positive, got {dimensions}")
    width = left_leg_width + gap_width + right_leg_width
    height = underside_height + body_thickness
    x0 = -0.5 * width
    x1 = x0 + left_leg_width
    x2 = x1 + gap_width
    x3 = 0.5 * width
    z0 = -0.5 * height
    z1 = z0 + underside_height
    z2 = 0.5 * height
    profile = [
        (x0, z0),
        (x1, z0),
        (x1, z1),
        (x2, z1),
        (x2, z0),
        (x3, z0),
        (x3, z2),
        (x0, z2),
    ]
    return _create_xz_extrusion_mesh(profile, depth)


def _create_low_arm_mesh(
    body_radius: float,
    body_height: float,
    arm_length: float,
    arm_width: float,
    arm_thickness: float,
    arm_clearance: float,
    angular_segments: int = 32,
) -> trimesh.Trimesh:
    """Create a structured cylindrical pedestal union with one thin, elevated arm."""
    if min(body_radius, body_height, arm_length, arm_width, arm_thickness) <= 0.0:
        raise ValueError("Low-arm dimensions must be positive")
    if arm_clearance < 0.0 or arm_clearance + arm_thickness >= body_height:
        raise ValueError("The low arm must fit strictly below the pedestal top")

    # Determine how many sectors the arm covers
    ratio = (arm_width / 2.0) / body_radius
    if ratio >= 1.0:
        raise ValueError("The arm is too wide for the cylinder body")
    theta = float(np.arcsin(ratio))
    sector_angle = 2.0 * np.pi / angular_segments
    half_hidden = max(1, int(round(theta / sector_angle)))
    if half_hidden * 2 >= angular_segments:
        raise ValueError("The arm is too wide, it hides the entire cylinder")

    # Snap the arm width and position to the sector boundaries
    actual_theta = half_hidden * sector_angle
    actual_arm_width = 2.0 * body_radius * float(np.sin(actual_theta))
    arm_attach_x = body_radius * float(np.cos(actual_theta))
    arm_outer_x = arm_attach_x + arm_length

    z_bottom = -0.5 * body_height
    z_arm_bottom = z_bottom + arm_clearance
    z_arm_top = z_arm_bottom + arm_thickness
    z_top = 0.5 * body_height

    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []

    # 4 cylinder rings
    rings = []
    for z in (z_bottom, z_arm_bottom, z_arm_top, z_top):
        ring = []
        for i in range(angular_segments):
            angle = i * sector_angle
            ring.append(len(vertices))
            vertices.append((body_radius * float(np.cos(angle)), body_radius * float(np.sin(angle)), z))
        rings.append(ring)
    R0, R1, R2, R3 = rings

    # Arm outer corners at bottom and top
    A0_bot = len(vertices)
    vertices.append((arm_outer_x, -actual_arm_width / 2.0, z_arm_bottom))
    A1_bot = len(vertices)
    vertices.append((arm_outer_x, actual_arm_width / 2.0, z_arm_bottom))

    A0_top = len(vertices)
    vertices.append((arm_outer_x, -actual_arm_width / 2.0, z_arm_top))
    A1_top = len(vertices)
    vertices.append((arm_outer_x, actual_arm_width / 2.0, z_arm_top))

    # Top and bottom caps (radial segments = 1 for base mesh)
    _append_cap(vertices, faces, R0, body_radius, z_bottom, 1, top=False)
    _append_cap(vertices, faces, R3, body_radius, z_top, 1, top=True)

    # Lower and upper cylinder walls
    _append_quad_band(faces, R0, R1, 0)
    _append_quad_band(faces, R2, R3, 2)

    # Middle cylinder wall (only the exposed part)
    for i in range(half_hidden, angular_segments - half_hidden):
        lower = R1[i]
        lower_next = R1[(i + 1) % angular_segments]
        upper = R2[i]
        upper_next = R2[(i + 1) % angular_segments]
        if (1 + i) % 2 == 0:
            faces.extend(((lower, lower_next, upper_next), (lower, upper_next, upper)))
        else:
            faces.extend(((lower, lower_next, upper), (lower_next, upper_next, upper)))

    # Middle arm vertical walls
    idx_start = (angular_segments - half_hidden) % angular_segments
    idx_end = half_hidden % angular_segments
    faces.extend((
        (R1[idx_start], A0_bot, A0_top),
        (R1[idx_start], A0_top, R2[idx_start])
    ))
    faces.extend((
        (A0_bot, A1_bot, A1_top),
        (A0_bot, A1_top, A0_top)
    ))
    faces.extend((
        (A1_bot, R1[idx_end], R2[idx_end]),
        (A1_bot, R2[idx_end], A1_top)
    ))

    # Arm bottom and top shelves
    arm_poly_bot = [A0_bot, A1_bot]
    arm_poly_top = [A0_top, A1_top]
    
    for i in range(half_hidden, -half_hidden - 1, -1):
        idx = (i + angular_segments) % angular_segments
        arm_poly_bot.append(R1[idx])
        arm_poly_top.append(R2[idx])

    points2d = [(vertices[idx][0], vertices[idx][1]) for idx in arm_poly_bot]
    triangles = _triangulate_simple_polygon(points2d)

    for tri in triangles:
        faces.append(tuple(reversed(tuple(arm_poly_bot[i] for i in tri))))

    for tri in triangles:
        faces.append(tuple(arm_poly_top[i] for i in tri))

    return trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)


def _create_overhang_mesh(
    support_size: tuple[float, float, float],
    plate_size: tuple[float, float, float],
    plate_offset: tuple[float, float],
) -> trimesh.Trimesh:
    support_width, support_depth, support_height = map(float, support_size)
    plate_width, plate_depth, plate_thickness = map(float, plate_size)
    offset_x, offset_y = map(float, plate_offset)
    if min((*support_size, *plate_size)) <= 0.0:
        raise ValueError("Overhang dimensions must be positive")
    if support_width >= plate_width or support_depth >= plate_depth:
        raise ValueError("The overhang plate must be wider and deeper than its support")
    if abs(offset_x) + 0.5 * support_width >= 0.5 * plate_width:
        raise ValueError("The support must remain strictly inside the plate in X")
    if abs(offset_y) + 0.5 * support_depth >= 0.5 * plate_depth:
        raise ValueError("The support must remain strictly inside the plate in Y")

    total_height = support_height + plate_thickness
    bottom = -0.5 * total_height
    interface = bottom + support_height
    top = 0.5 * total_height
    support = (
        (-0.5 * support_width, -0.5 * support_depth, bottom),
        (0.5 * support_width, 0.5 * support_depth, interface),
    )
    plate = (
        (offset_x - 0.5 * plate_width, offset_y - 0.5 * plate_depth, interface),
        (offset_x + 0.5 * plate_width, offset_y + 0.5 * plate_depth, top),
    )
    return _create_box_union_mesh((support, plate))


def _create_box_union_mesh(
    boxes: tuple[
        tuple[tuple[float, float, float], tuple[float, float, float]], ...
    ]
) -> trimesh.Trimesh:
    """Mesh the exterior of an axis-aligned box union without internal faces."""
    x_values = sorted({coordinate for box in boxes for coordinate in (box[0][0], box[1][0])})
    y_values = sorted({coordinate for box in boxes for coordinate in (box[0][1], box[1][1])})
    z_values = sorted({coordinate for box in boxes for coordinate in (box[0][2], box[1][2])})
    occupied = np.zeros((len(x_values) - 1, len(y_values) - 1, len(z_values) - 1), dtype=bool)
    for i in range(len(x_values) - 1):
        for j in range(len(y_values) - 1):
            for k in range(len(z_values) - 1):
                midpoint = (
                    0.5 * (x_values[i] + x_values[i + 1]),
                    0.5 * (y_values[j] + y_values[j + 1]),
                    0.5 * (z_values[k] + z_values[k + 1]),
                )
                occupied[i, j, k] = any(
                    all(lower[axis] < midpoint[axis] < upper[axis] for axis in range(3))
                    for lower, upper in boxes
                )

    vertices: list[tuple[float, float, float]] = []
    vertex_lookup: dict[tuple[float, float, float], int] = {}
    faces: list[tuple[int, int, int]] = []

    def add_quad(points: tuple[tuple[float, float, float], ...]) -> None:
        indices = []
        for point in points:
            if point not in vertex_lookup:
                vertex_lookup[point] = len(vertices)
                vertices.append(point)
            indices.append(vertex_lookup[point])
        faces.extend(((indices[0], indices[1], indices[2]), (indices[0], indices[2], indices[3])))

    for i, j, k in np.argwhere(occupied):
        x0, x1 = x_values[i], x_values[i + 1]
        y0, y1 = y_values[j], y_values[j + 1]
        z0, z1 = z_values[k], z_values[k + 1]
        if i == 0 or not occupied[i - 1, j, k]:
            add_quad(((x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0)))
        if i == occupied.shape[0] - 1 or not occupied[i + 1, j, k]:
            add_quad(((x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)))
        if j == 0 or not occupied[i, j - 1, k]:
            add_quad(((x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)))
        if j == occupied.shape[1] - 1 or not occupied[i, j + 1, k]:
            add_quad(((x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0)))
        if k == 0 or not occupied[i, j, k - 1]:
            add_quad(((x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0)))
        if k == occupied.shape[2] - 1 or not occupied[i, j, k + 1]:
            add_quad(((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)))
    return trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)


def _orient_axial_mesh(mesh: trimesh.Trimesh, axis: str) -> None:
    """Rotate a Z-axis trimesh cylinder/cone onto the requested local axis."""
    axis = str(axis).upper()
    if axis == "X":
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2.0, [0, 1, 0]))
    elif axis == "Y":
        mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2.0, [1, 0, 0]))
    elif axis != "Z":
        raise ValueError(f"Axis must be one of X, Y, or Z, got {axis!r}")


def _structured_segment_counts(primitive_cfg: dict) -> tuple[int, int, int]:
    """Read structured-grid counts, with a fallback for older subdivision configs."""
    subdivisions = int(primitive_cfg.get("subdivisions", 2))
    if subdivisions < 0:
        raise ValueError(f"subdivisions must be non-negative, got {subdivisions}")
    return (
        int(primitive_cfg.get("angular_segments", 32)),
        int(primitive_cfg.get("height_segments", 4 * 2**subdivisions)),
        int(primitive_cfg.get("cap_radial_segments", 2 * 2**subdivisions)),
    )


def _validate_structured_mesh_dimensions(
    radius: float,
    height: float,
    angular_segments: int,
    height_segments: int,
    cap_radial_segments: int,
) -> None:
    if radius <= 0.0:
        raise ValueError(f"radius must be positive, got {radius}")
    if height <= 0.0:
        raise ValueError(f"height must be positive, got {height}")
    if angular_segments < 3:
        raise ValueError(f"angular_segments must be at least 3, got {angular_segments}")
    if height_segments < 1:
        raise ValueError(f"height_segments must be at least 1, got {height_segments}")
    if cap_radial_segments < 1:
        raise ValueError(f"cap_radial_segments must be at least 1, got {cap_radial_segments}")


def _append_quad_band(
    faces: list[tuple[int, int, int]],
    lower_ring: list[int],
    upper_ring: list[int],
    band_index: int,
) -> None:
    """Triangulate a ring-to-ring band while alternating the quad diagonals."""
    angular_segments = len(lower_ring)
    for angular_index in range(angular_segments):
        next_index = (angular_index + 1) % angular_segments
        lower = lower_ring[angular_index]
        lower_next = lower_ring[next_index]
        upper = upper_ring[angular_index]
        upper_next = upper_ring[next_index]
        if (band_index + angular_index) % 2 == 0:
            faces.extend(((lower, lower_next, upper_next), (lower, upper_next, upper)))
        else:
            faces.extend(((lower, lower_next, upper), (lower_next, upper_next, upper)))


def _append_cap(
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, int, int]],
    outer_ring: list[int],
    radius: float,
    z: float,
    cap_radial_segments: int,
    top: bool,
) -> None:
    """Add concentric cap rings, sharing the outer rim with the curved surface."""
    angular_segments = len(outer_ring)
    center_index = len(vertices)
    vertices.append((0.0, 0.0, z))

    rings: list[list[int]] = []
    for radial_index in range(1, cap_radial_segments):
        ring_radius = radius * radial_index / cap_radial_segments
        ring = []
        for angular_index in range(angular_segments):
            theta = 2.0 * np.pi * angular_index / angular_segments
            ring.append(len(vertices))
            vertices.append((ring_radius * np.cos(theta), ring_radius * np.sin(theta), z))
        rings.append(ring)
    rings.append(outer_ring)

    first_ring = rings[0]
    for angular_index in range(angular_segments):
        next_index = (angular_index + 1) % angular_segments
        triangle = (center_index, first_ring[angular_index], first_ring[next_index])
        faces.append(triangle if top else tuple(reversed(triangle)))

    for radial_index, (inner_ring, outer_cap_ring) in enumerate(zip(rings[:-1], rings[1:])):
        cap_faces: list[tuple[int, int, int]] = []
        _append_quad_band(cap_faces, inner_ring, outer_cap_ring, radial_index)
        # _append_quad_band winds an inner-to-outer planar band toward -Z.
        if top:
            faces.extend(tuple(reversed(face)) for face in cap_faces)
        else:
            faces.extend(cap_faces)


def _create_structured_cylinder_mesh(
    radius: float,
    height: float,
    angular_segments: int,
    height_segments: int,
    cap_radial_segments: int,
) -> trimesh.Trimesh:
    """Create a closed Z-axis cylinder with a two-dimensional surface grid."""
    _validate_structured_mesh_dimensions(
        radius, height, angular_segments, height_segments, cap_radial_segments
    )
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    side_rings: list[list[int]] = []

    for height_index in range(height_segments + 1):
        z = -height / 2.0 + height * height_index / height_segments
        ring = []
        for angular_index in range(angular_segments):
            theta = 2.0 * np.pi * angular_index / angular_segments
            ring.append(len(vertices))
            vertices.append((radius * np.cos(theta), radius * np.sin(theta), z))
        side_rings.append(ring)

    for height_index in range(height_segments):
        _append_quad_band(faces, side_rings[height_index], side_rings[height_index + 1], height_index)

    _append_cap(vertices, faces, side_rings[0], radius, -height / 2.0, cap_radial_segments, top=False)
    _append_cap(vertices, faces, side_rings[-1], radius, height / 2.0, cap_radial_segments, top=True)
    return trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)


def _create_structured_cone_mesh(
    radius: float,
    height: float,
    angular_segments: int,
    height_segments: int,
    cap_radial_segments: int,
) -> trimesh.Trimesh:
    """Create a closed Z-axis cone with rings from its base toward its apex."""
    _validate_structured_mesh_dimensions(
        radius, height, angular_segments, height_segments, cap_radial_segments
    )
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    side_rings: list[list[int]] = []

    for height_index in range(height_segments):
        fraction = height_index / height_segments
        ring_radius = radius * (1.0 - fraction)
        z = height * fraction
        ring = []
        for angular_index in range(angular_segments):
            theta = 2.0 * np.pi * angular_index / angular_segments
            ring.append(len(vertices))
            vertices.append((ring_radius * np.cos(theta), ring_radius * np.sin(theta), z))
        side_rings.append(ring)

    for height_index in range(height_segments - 1):
        _append_quad_band(faces, side_rings[height_index], side_rings[height_index + 1], height_index)

    apex_index = len(vertices)
    vertices.append((0.0, 0.0, height))
    final_ring = side_rings[-1]
    for angular_index in range(angular_segments):
        next_index = (angular_index + 1) % angular_segments
        faces.append((final_ring[angular_index], final_ring[next_index], apex_index))

    _append_cap(vertices, faces, side_rings[0], radius, 0.0, cap_radial_segments, top=False)
    return trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)
