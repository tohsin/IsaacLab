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
    """Spawn a cylinder whose surface contains many triangle faces."""
    if cfg.subdivisions < 0:
        raise ValueError(f"subdivisions must be non-negative, got {cfg.subdivisions}")

    mesh = trimesh.creation.cylinder(radius=cfg.radius, height=cfg.height, sections=32)
    _orient_axial_mesh(mesh, cfg.axis)
    for _ in range(cfg.subdivisions):
        vertices, faces = trimesh.remesh.subdivide(mesh.vertices, mesh.faces)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    _spawn_mesh_geom_from_mesh(prim_path, cfg, mesh, translation, orientation)
    return prim_utils.get_prim_at_path(prim_path)


@configclass
class TessellatedCylinderCfg(sim_utils.MeshCylinderCfg):
    """Configuration for a cylinder with explicitly subdivided triangle faces."""

    func: Callable = spawn_tessellated_cylinder
    subdivisions: int = 2
    """Recursive triangle subdivisions. Two produces 2048 triangles."""


@clone
def spawn_tessellated_cone(
    prim_path: str,
    cfg: "TessellatedConeCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn a cone whose surface contains many triangle faces."""
    if cfg.subdivisions < 0:
        raise ValueError(f"subdivisions must be non-negative, got {cfg.subdivisions}")

    mesh = trimesh.creation.cone(radius=cfg.radius, height=cfg.height, sections=32)
    _orient_axial_mesh(mesh, cfg.axis)
    for _ in range(cfg.subdivisions):
        vertices, faces = trimesh.remesh.subdivide(mesh.vertices, mesh.faces)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    _spawn_mesh_geom_from_mesh(prim_path, cfg, mesh, translation, orientation)
    return prim_utils.get_prim_at_path(prim_path)


@configclass
class TessellatedConeCfg(sim_utils.MeshConeCfg):
    """Configuration for a cone with explicitly subdivided triangle faces."""

    func: Callable = spawn_tessellated_cone
    subdivisions: int = 2
    """Recursive triangle subdivisions."""


def build_tessellated_primitive_cfg(primitive_cfg: dict, **spawn_kwargs):
    """Build the appropriate Isaac Lab spawner config from one data entry."""
    primitive_type = primitive_cfg["type"]
    if primitive_type == "tessellated_cuboid":
        return TessellatedCuboidCfg(
            size=tuple(primitive_cfg["size"]),
            subdivisions=int(primitive_cfg.get("subdivisions", 3)),
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
        return TessellatedCylinderCfg(
            radius=float(primitive_cfg.get("radius", 0.5)),
            height=float(primitive_cfg.get("height", 1.0)),
            axis=str(axis).upper(),
            subdivisions=int(primitive_cfg.get("subdivisions", 2)),
            **spawn_kwargs,
        )
    if primitive_type == "tessellated_cone":
        return TessellatedConeCfg(
            radius=float(primitive_cfg.get("radius", 0.5)),
            height=float(primitive_cfg.get("height", 1.0)),
            axis=str(primitive_cfg.get("axis", "Z")).upper(),
            subdivisions=int(primitive_cfg.get("subdivisions", 2)),
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
