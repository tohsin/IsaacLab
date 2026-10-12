#!/usr/bin/env python3

"""Export and visualize evaluation-target visual meshes and collision proxies.

The robot-inspection environment applies collision to every mesh descendant of
an imported target. If a mesh does not already have ``MeshCollisionAPI``, the
environment requests a ``convexHull`` approximation. This utility reconstructs
that setup without modifying the source USD and produces OBJ/GLB/PNG artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--targets",
    nargs="+",
    default=("dolly", "caster"),
    help="Evaluation target names to inspect.",
)
parser.add_argument(
    "--output_dir",
    type=Path,
    default=Path("data/collision_mesh_diagnostics"),
    help="Directory for exported geometry, images, and measurements.",
)
parser.add_argument(
    "--max_plot_faces",
    type=int,
    default=30_000,
    help="Maximum visual triangles drawn in each Matplotlib panel.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from isaaclab_tasks.direct.robot_inspection.configs.eval_dataset import evaluation_data_set


def _triangulate_mesh(mesh_prim: UsdGeom.Mesh) -> tuple[np.ndarray, np.ndarray]:
    """Read and fan-triangulate one USD mesh in its local frame."""
    points = mesh_prim.GetPointsAttr().Get()
    counts = mesh_prim.GetFaceVertexCountsAttr().Get()
    indices = mesh_prim.GetFaceVertexIndicesAttr().Get()
    if not points or not counts or not indices:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.int64)

    vertices = np.asarray(points, dtype=np.float64)
    triangles: list[tuple[int, int, int]] = []
    offset = 0
    for count in counts:
        count = int(count)
        polygon = indices[offset : offset + count]
        offset += count
        if count < 3:
            continue
        first = int(polygon[0])
        triangles.extend(
            (first, int(polygon[index]), int(polygon[index + 1]))
            for index in range(1, count - 1)
        )
    return vertices, np.asarray(triangles, dtype=np.int64)


def _transform_vertices(vertices: np.ndarray, transform: Gf.Matrix4d) -> np.ndarray:
    """Apply a USD transform without relying on matrix storage conventions."""
    return np.asarray(
        [
            tuple(transform.Transform(Gf.Vec3d(float(point[0]), float(point[1]), float(point[2]))))
            for point in vertices
        ],
        dtype=np.float64,
    )


def _apply_dataset_transform(mesh: trimesh.Trimesh, target_cfg: dict) -> None:
    """Apply the scale and spawn orientation used by the evaluation environment."""
    scale = target_cfg.get("scale", 10.0)
    if isinstance(scale, (int, float)):
        scale = (float(scale),) * 3
    mesh.vertices *= np.asarray(scale, dtype=np.float64)

    orientation = target_cfg.get("orientation", (1.0, 0.0, 0.0, 0.0))
    w, x, y, z = (float(value) for value in orientation)
    rotation = Rotation.from_quat((x, y, z, w)).as_matrix()
    mesh.vertices = mesh.vertices @ rotation.T


def _collision_proxy(mesh: trimesh.Trimesh, approximation: str) -> tuple[trimesh.Trimesh, str]:
    """Reconstruct the requested collision approximation for one mesh child."""
    if approximation in {"", "none"}:
        return mesh.copy(), "none"
    if approximation == "convexHull":
        return mesh.convex_hull, "convexHull"

    # PhysX cooks these approximations internally. A convex hull is a useful
    # conservative diagnostic, but the summary records that it is approximate.
    if approximation in {"convexDecomposition", "meshSimplification"}:
        return mesh.convex_hull, f"{approximation} (visualized as conservative convex hull)"
    if approximation == "boundingCube":
        return trimesh.creation.box(extents=mesh.extents, transform=mesh.bounding_box.transform), approximation
    if approximation == "boundingSphere":
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=float(mesh.bounding_sphere.primitive.radius))
        sphere.apply_translation(mesh.bounding_sphere.primitive.center)
        return sphere, approximation
    return mesh.convex_hull, f"{approximation} (unknown; visualized as conservative convex hull)"


def _plot_triangles(
    axis,
    mesh: trimesh.Trimesh,
    color: tuple[float, float, float],
    alpha: float,
    max_faces: int,
    edges: bool = False,
) -> None:
    triangles = mesh.triangles
    if len(triangles) > max_faces:
        sample_indices = np.linspace(0, len(triangles) - 1, max_faces, dtype=np.int64)
        triangles = triangles[sample_indices]
    collection = Poly3DCollection(
        triangles,
        facecolor=color,
        edgecolor=(0.45, 0.12, 0.02, min(0.5, alpha)) if edges else "none",
        linewidth=0.12 if edges else 0.0,
        alpha=alpha,
    )
    axis.add_collection3d(collection)


def _configure_axis(axis, bounds: np.ndarray, top_down: bool) -> None:
    center = bounds.mean(axis=0)
    span = np.maximum(bounds[1] - bounds[0], 1.0e-6)
    half_range = float(span.max()) * 0.55
    axis.set_xlim(center[0] - half_range, center[0] + half_range)
    axis.set_ylim(center[1] - half_range, center[1] + half_range)
    axis.set_zlim(center[2] - half_range, center[2] + half_range)
    axis.set_box_aspect((1.0, 1.0, 1.0))
    axis.set_proj_type("ortho")
    axis.view_init(elev=90.0, azim=-90.0) if top_down else axis.view_init(elev=24.0, azim=-52.0)
    axis.set_axis_off()


def _surface_distance_summary(
    visual_mesh: trimesh.Trimesh, collision_mesh: trimesh.Trimesh
) -> dict[str, float]:
    """Approximate collision-surface distance from dense surface samples."""
    np.random.seed(42)
    visual_sample_count = min(200_000, max(20_000, len(visual_mesh.faces) * 5))
    collision_sample_count = min(60_000, max(10_000, len(collision_mesh.faces) * 20))
    visual_points, _ = trimesh.sample.sample_surface(visual_mesh, visual_sample_count)
    collision_points, _ = trimesh.sample.sample_surface(collision_mesh, collision_sample_count)
    distances, _ = cKDTree(visual_points).query(collision_points, k=1, workers=-1)
    return {
        "mean": float(np.mean(distances)),
        "p95": float(np.percentile(distances, 95)),
        "p99": float(np.percentile(distances, 99)),
        "max": float(np.max(distances)),
    }


def _analyze_target(target_name: str, target_cfg: dict, output_dir: Path, max_plot_faces: int) -> dict:
    stage = Usd.Stage.Open(target_cfg["usd_path"])
    if stage is None:
        raise RuntimeError(f"Could not open USD for {target_name!r}: {target_cfg['usd_path']}")

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    visual_parts: list[trimesh.Trimesh] = []
    collision_parts: list[trimesh.Trimesh] = []
    child_summaries: list[dict] = []

    # UsdFileCfg references the source stage's default prim beneath the target
    # prim. Sibling roots (for example a source-stage test plane) are not part
    # of the spawned asset and must not appear in this diagnostic.
    default_prim = stage.GetDefaultPrim()
    prim_range = Usd.PrimRange(default_prim) if default_prim.IsValid() else stage.Traverse()
    for prim in prim_range:
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh_prim = UsdGeom.Mesh(prim)
        vertices, faces = _triangulate_mesh(mesh_prim)
        if len(vertices) == 0 or len(faces) == 0:
            continue
        vertices = _transform_vertices(vertices, xform_cache.GetLocalToWorldTransform(prim))
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        _apply_dataset_transform(mesh, target_cfg)

        imageable = UsdGeom.Imageable(prim)
        visibility = str(imageable.ComputeVisibility())
        purpose = str(imageable.ComputePurpose())
        rendered = visibility != "invisible" and purpose in {"default", "render"}
        if rendered:
            visual_parts.append(mesh)

        had_authored_mesh_collision = prim.HasAPI(UsdPhysics.MeshCollisionAPI)
        if had_authored_mesh_collision:
            approximation = str(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() or "none")
        else:
            approximation = "convexHull"
        proxy, displayed_approximation = _collision_proxy(mesh, approximation)
        collision_parts.append(proxy)
        child_summaries.append(
            {
                "path": str(prim.GetPath()),
                "rendered": rendered,
                "visibility": visibility,
                "purpose": purpose,
                "visual_triangles": int(len(mesh.faces)),
                "authored_mesh_collision_api": had_authored_mesh_collision,
                "requested_approximation": approximation,
                "displayed_approximation": displayed_approximation,
                "collision_triangles": int(len(proxy.faces)),
            }
        )

    if not visual_parts or not collision_parts:
        raise RuntimeError(f"No usable mesh geometry found for {target_name!r}")

    visual_mesh = trimesh.util.concatenate(visual_parts)
    collision_mesh = trimesh.util.concatenate(collision_parts)
    combined_bounds = np.vstack((visual_mesh.bounds, collision_mesh.bounds))
    plot_bounds = np.vstack((combined_bounds.min(axis=0), combined_bounds.max(axis=0)))

    target_dir = output_dir / target_name
    target_dir.mkdir(parents=True, exist_ok=True)
    visual_path = target_dir / f"{target_name}_visual.obj"
    collision_path = target_dir / f"{target_name}_collision_proxy.obj"
    overlay_path = target_dir / f"{target_name}_overlay.glb"
    image_path = target_dir / f"{target_name}_collision_diagnostic.png"
    summary_path = target_dir / f"{target_name}_collision_summary.json"

    visual_mesh.export(visual_path)
    collision_mesh.export(collision_path)

    visual_overlay = visual_mesh.copy()
    collision_overlay = collision_mesh.copy()
    visual_overlay.visual.face_colors = (55, 120, 190, 210)
    collision_overlay.visual.face_colors = (245, 100, 30, 95)
    trimesh.Scene(
        {
            "visual_mesh": visual_overlay,
            "collision_proxy": collision_overlay,
        }
    ).export(overlay_path)

    figure = plt.figure(figsize=(15, 9), constrained_layout=True)
    panels = (
        ("Visual mesh — isometric", visual_mesh, None, False),
        ("Collision proxy — isometric", None, collision_mesh, False),
        ("Overlay — isometric", visual_mesh, collision_mesh, False),
        ("Visual mesh — top", visual_mesh, None, True),
        ("Collision proxy — top", None, collision_mesh, True),
        ("Overlay — top", visual_mesh, collision_mesh, True),
    )
    for panel_index, (title, visual, collision, top_down) in enumerate(panels, start=1):
        axis = figure.add_subplot(2, 3, panel_index, projection="3d")
        if visual is not None:
            _plot_triangles(axis, visual, (0.22, 0.48, 0.78), 0.78, max_plot_faces)
        if collision is not None:
            _plot_triangles(axis, collision, (0.96, 0.35, 0.08), 0.28, max_plot_faces, edges=True)
        _configure_axis(axis, plot_bounds, top_down)
        axis.set_title(title, fontsize=11)
    figure.legend(
        handles=(
            Patch(facecolor=(0.22, 0.48, 0.78, 0.78), label="Rendered visual surface"),
            Patch(facecolor=(0.96, 0.35, 0.08, 0.28), edgecolor=(0.45, 0.12, 0.02), label="Collision proxy"),
        ),
        loc="lower center",
        ncol=2,
        frameon=False,
    )
    figure.suptitle(f"{target_name}: visual geometry versus requested collision proxy", fontsize=15)
    figure.savefig(image_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    collision_approximations: dict[str, int] = {}
    for child in child_summaries:
        approximation = child["displayed_approximation"]
        collision_approximations[approximation] = collision_approximations.get(approximation, 0) + 1

    summary = {
        "target": target_name,
        "source_usd": target_cfg["usd_path"],
        "source_default_prim": str(default_prim.GetPath()) if default_prim.IsValid() else None,
        "dataset_scale": target_cfg.get("scale", 10.0),
        "dataset_orientation_wxyz": target_cfg.get("orientation", (1.0, 0.0, 0.0, 0.0)),
        "mesh_children": len(child_summaries),
        "rendered_mesh_children": int(sum(child["rendered"] for child in child_summaries)),
        "visual_triangles": int(len(visual_mesh.faces)),
        "collision_proxy_triangles": int(len(collision_mesh.faces)),
        "collision_approximations": collision_approximations,
        "visual_extents": visual_mesh.extents.tolist(),
        "collision_extents": collision_mesh.extents.tolist(),
        "collision_to_visual_surface_distance": _surface_distance_summary(visual_mesh, collision_mesh),
        "children": child_summaries,
        "artifacts": {
            "visual_obj": str(visual_path),
            "collision_obj": str(collision_path),
            "overlay_glb": str(overlay_path),
            "diagnostic_png": str(image_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"[INFO] {target_name}: {len(child_summaries)} mesh children, "
        f"{len(visual_mesh.faces)} visual triangles, {len(collision_mesh.faces)} collision triangles"
    )
    print(f"[INFO] Wrote {image_path}")
    return summary


def main() -> None:
    output_dir = args_cli.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    unknown_targets = sorted(set(args_cli.targets) - set(evaluation_data_set))
    if unknown_targets:
        raise ValueError(
            f"Unknown evaluation targets {unknown_targets}; choose from {sorted(evaluation_data_set)}"
        )

    summaries = [
        _analyze_target(
            target_name,
            evaluation_data_set[target_name],
            output_dir,
            args_cli.max_plot_faces,
        )
        for target_name in args_cli.targets
    ]
    combined_path = output_dir / "collision_geometry_summary.json"
    combined_path.write_text(json.dumps(summaries, indent=2) + "\n")
    print(f"[INFO] Wrote {combined_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
