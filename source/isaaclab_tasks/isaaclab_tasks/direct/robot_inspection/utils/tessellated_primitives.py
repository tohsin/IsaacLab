"""Task-local mesh primitives with controllable face density.

The stock :class:`isaaclab.sim.MeshCuboidCfg` creates a box with twelve
triangles.  That is sufficient for rendering and collision, but too coarse for
an inspection reward that treats triangle IDs as surface samples.  The
spawner below recursively subdivides those triangles while leaving the shape
of the cuboid unchanged.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import trimesh
import isaacsim.core.utils.prims as prim_utils
from pxr import Usd

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
