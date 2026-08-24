"""Procedural primitive targets used for multi-object training."""


def _subdivided_faces(base_faces: int, subdivisions: int) -> int:
    return base_faces * 4**subdivisions


def _target(name: str, reachable_faces: int, mesh_faces: int, primitive: dict) -> dict:
    """Build the common dataset fields for one procedural target."""
    return {
        "num_faces": reachable_faces,
        "mesh_num_faces": mesh_faces,
        "prim_path": f"/World/envs/env_.*/{name}",
        "primitive": primitive,
        "scale": 1.0,
    }


_CUBE_FACES = _subdivided_faces(12, 3)  # 768
_SPHERE_FACES = 20 * 4**3  # 1,280-face icosphere
_CYLINDER_FACES = _subdivided_faces(128, 2)  # 2,048
_CONE_FACES = _subdivided_faces(64, 2)  # 1,024

# These settings are applied once at environment startup. Geometry is sampled
# from a bounded bank so identical variants can share cached Warp meshes.
_COMMON_RANDOMIZATION = {
    "num_size_variants": 16,
    "num_color_variants": 16,
    "color_min": (0.1, 0.1, 0.1),
    "color_max": (0.9, 0.9, 0.9),
}

_CUBOID_RANDOMIZATION = {
    **_COMMON_RANDOMIZATION,
    "size_min": (0.8, 0.8, 0.8),
    "size_max": (1.2, 1.2, 1.6),


    # A cuboid's appearance changes under yaw, unlike a sphere or an upright
    # axial primitive. Sample the full circle at every randomized reset.
    "yaw_range": (-3.141592653589793, 3.141592653589793),
}

_SPHERE_RANDOMIZATION = {
    **_COMMON_RANDOMIZATION,
    "radius_min": 0.5,
    "radius_max": 0.8,
}

_AXIAL_RANDOMIZATION = {
    **_COMMON_RANDOMIZATION,
    "radius_min": 0.5,
    "radius_max": 0.8,
    "height_min": 0.8,
    "height_max": 1.8,
}


primitive_data_set = {
    "tessellated_cube": _target(
        "tessellated_cube",
        # Four vertical sides; top and bottom are excluded.
        reachable_faces=_CUBE_FACES * 4 // 6,
        mesh_faces=_CUBE_FACES,
        primitive={
            "type": "tessellated_cuboid",
            "size": (0.8, 0.8, 0.8),
            "subdivisions": 3,
            "domain_randomization": {**_CUBOID_RANDOMIZATION},
        },
    ),
    "sphere": _target(
        "sphere",
        # A sphere has no finite-area bottom cap.
        reachable_faces=_SPHERE_FACES,
        mesh_faces=_SPHERE_FACES,
        primitive={
            "type": "tessellated_sphere",
            "radius": 0.4,
            "subdivisions": 3,
            "domain_randomization": {**_SPHERE_RANDOMIZATION},
        },
    ),
    "cylinder_upright": _target(
        "cylinder_upright",
        # Curved band only under the no-top/no-bottom assumption.
        reachable_faces=_CYLINDER_FACES // 2,
        mesh_faces=_CYLINDER_FACES,
        primitive={
            "type": "tessellated_cylinder",
            "axis": "Z",
            "radius": 0.4,
            "height": 0.8,
            "subdivisions": 2,
            "domain_randomization": {**_AXIAL_RANDOMIZATION},
        },
    ),
    "cylinder_flat": _target(
        "cylinder_flat",
        # Provisional reachable estimate until evaluation calibrates it.
        reachable_faces=int(_CYLINDER_FACES * 0.9),
        mesh_faces=_CYLINDER_FACES,
        primitive={
            # This is the same generator as the upright cylinder; only its
            # authored local axis and reachable denominator differ.
            "type": "tessellated_cylinder",
            "axis": "X",
            "radius": 0.4,
            "height": 0.8,
            "subdivisions": 2,
            "domain_randomization": {
                **_AXIAL_RANDOMIZATION,
                # Rotate only around world Z so the cylinder remains flat.
                "yaw_range": (-3.141592653589793, 3.141592653589793),
            },
        },
    ),
    "cone": _target(
        "cone",
        # Curved side only; the base's 512 triangles rest on the floor.
        reachable_faces=_CONE_FACES // 2,
        mesh_faces=_CONE_FACES,
        primitive={
            "type": "tessellated_cone",
            "axis": "Z",
            "radius": 0.4,
            "height": 0.8,
            "subdivisions": 2,
            "domain_randomization": {**_AXIAL_RANDOMIZATION},
        },
    ),
    "cone_flat": _target(
        "cone_flat",
        # A sideways cone exposes its base and nearly all of its curved side.
        # Keep this provisional until the reachability diagnostic calibrates it.
        reachable_faces=int(_CONE_FACES * 0.9),
        mesh_faces=_CONE_FACES,
        primitive={
            "type": "tessellated_cone",
            "axis": "X",
            "radius": 0.4,
            "height": 0.8,
            "subdivisions": 2,
            "domain_randomization": {
                **_AXIAL_RANDOMIZATION,
                # Rotate around world Z while preserving the horizontal axis.
                "yaw_range": (-3.141592653589793, 3.141592653589793),
            },
        },
    ),
}

# Preserve the name used while the primitives were first being tested.
primitive_test_data_set = primitive_data_set
