"""Procedural primitive targets used for multi-object training."""


def _subdivided_faces(base_faces: int, subdivisions: int) -> int:
    return base_faces * 4**subdivisions


def _cap_faces(angular_segments: int, radial_segments: int) -> int:
    return angular_segments * (2 * radial_segments - 1)


def _cylinder_faces(angular_segments: int, height_segments: int, radial_segments: int) -> int:
    return 2 * angular_segments * height_segments + 2 * _cap_faces(angular_segments, radial_segments)


def _cone_faces(angular_segments: int, height_segments: int, radial_segments: int) -> int:
    side_faces = angular_segments * (2 * height_segments - 1)
    return side_faces + _cap_faces(angular_segments, radial_segments)


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
_T_BLOCK_FACES = _subdivided_faces(28, 3)  # 1792
_SHELL_FACES = _subdivided_faces(28, 3)  # 1792
_SPHERE_FACES = 20 * 4**3  # 1,280-face icosphere
_ANGULAR_SEGMENTS = 64
_HEIGHT_SEGMENTS = 32
_CAP_RADIAL_SEGMENTS = 8
_CYLINDER_SIDE_FACES = 2 * _ANGULAR_SEGMENTS * _HEIGHT_SEGMENTS  # 1,024
_CYLINDER_FACES = _cylinder_faces(_ANGULAR_SEGMENTS, _HEIGHT_SEGMENTS, _CAP_RADIAL_SEGMENTS)  # 1,984
_CONE_SIDE_FACES = _ANGULAR_SEGMENTS * (2 * _HEIGHT_SEGMENTS - 1)  # 992
_CONE_FACES = _cone_faces(_ANGULAR_SEGMENTS, _HEIGHT_SEGMENTS, _CAP_RADIAL_SEGMENTS)  # 1,472

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
    "size_min": (1.0, 1.0, 1.3),
    "size_max": (1.0, 1.0, 1.3),


    # A cuboid's appearance changes under yaw, unlike a sphere or an upright
    # axial primitive. Sample the full circle at every randomized reset.
    "yaw_range": (-3.141592653589793, 3.141592653589793),
}

_LOW_CUBOID_RANDOMIZATION = {
    **_COMMON_RANDOMIZATION,
    "size_min": (1.0, 1.0, 0.2),
    "size_max": (1.0, 1.0, 0.2),
    "yaw_range": (-3.141592653589793, 3.141592653589793),
}

_HIGH_CUBOID_RANDOMIZATION = {
    **_COMMON_RANDOMIZATION,
    "size_min": (1.0, 1.0, 1.2),
    "size_max": (1.0, 1.0, 1.2),
    "yaw_range": (-3.141592653589793, 3.141592653589793),
}

_SPHERE_RANDOMIZATION = {
    **_COMMON_RANDOMIZATION,
    "radius_min": 0.65,
    "radius_max": 0.65,
}

_AXIAL_RANDOMIZATION = {
    **_COMMON_RANDOMIZATION,
    "radius_min": 0.65,
    "radius_max": 0.65,
    "height_min": 1.3,
    "height_max": 1.3,
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
    "tessellated_t_block": _target(
        "tessellated_t_block",
        # Exclude bottom stem (2 triangles) and top of the bar (2 triangles)
        reachable_faces=_T_BLOCK_FACES * 24 // 28,
        mesh_faces=_T_BLOCK_FACES,
        primitive={
            "type": "tessellated_t_block",
            "size": (0.8, 0.8, 0.8),
            "subdivisions": 3,
            "bar_height_fraction": 0.3,
            "stem_width_fraction": 0.4,
            "domain_randomization": {**_CUBOID_RANDOMIZATION},
        },
    ),
    "tessellated_t_block_flat": _target(
        "tessellated_t_block_flat",
        # Rests flat (excludes 6 bottom triangles) and robot is too short to see the top (excludes 6 top triangles).
        reachable_faces=_T_BLOCK_FACES * 16 // 28,


        mesh_faces=_T_BLOCK_FACES,
        primitive={
            "type": "tessellated_t_block_flat",
            "size": (0.8, 0.8, 0.8),
            "subdivisions": 3,
            "bar_height_fraction": 0.3,
            "stem_width_fraction": 0.4,
            "domain_randomization": {
                **_CUBOID_RANDOMIZATION,
                "yaw_range": (-3.141592653589793, 3.141592653589793),
            },
        },
    ),
    # "tessellated_shell": _target(
    #     "tessellated_shell",
    #     # Rests on the outer bottom face (2 base triangles out of 28)
    #     reachable_faces=_SHELL_FACES * 26 // 28,
    #     mesh_faces=_SHELL_FACES,
    #     primitive={
    #         "type": "tessellated_shell",
    #         "size": (1.0, 1.0, 0.6),
    #         "subdivisions": 3,
    #         "wall_thickness": 0.1,
    #         "domain_randomization": {**_LOW_CUBOID_RANDOMIZATION},
    #     },
    # ),
    "tessellated_shell_side": _target(
        "tessellated_shell_side",
        # Rests on the new bottom (2 triangles), and we exclude the new top (2 triangles)
        reachable_faces=_SHELL_FACES * 24 // 28,
        mesh_faces=_SHELL_FACES,
        primitive={
            "type": "tessellated_shell_side",
            "size": (1.0, 1.0, 0.6),
            "subdivisions": 3,
            "wall_thickness": 0.1,
            "domain_randomization": {**_HIGH_CUBOID_RANDOMIZATION},
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
        reachable_faces=_CYLINDER_SIDE_FACES,
        mesh_faces=_CYLINDER_FACES,
        primitive={
            "type": "tessellated_cylinder",
            "axis": "Z",
            "radius": 0.4,
            "height": 0.8,
            "angular_segments": _ANGULAR_SEGMENTS,
            "height_segments": _HEIGHT_SEGMENTS,
            "cap_radial_segments": _CAP_RADIAL_SEGMENTS,
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
            "angular_segments": _ANGULAR_SEGMENTS,
            "height_segments": _HEIGHT_SEGMENTS,
            "cap_radial_segments": _CAP_RADIAL_SEGMENTS,
            "domain_randomization": {
                **_AXIAL_RANDOMIZATION,
                # Rotate only around world Z so the cylinder remains flat.
                "yaw_range": (-3.141592653589793, 3.141592653589793),
            },
        },
    ),
    "cone": _target(
        "cone",
        # Curved side only; the concentric base rests on the floor.
        reachable_faces=_CONE_SIDE_FACES,
        mesh_faces=_CONE_FACES,
        primitive={
            "type": "tessellated_cone",
            "axis": "Z",
            "radius": 0.4,
            "height": 0.8,
            "angular_segments": _ANGULAR_SEGMENTS,
            "height_segments": _HEIGHT_SEGMENTS,
            "cap_radial_segments": _CAP_RADIAL_SEGMENTS,
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
            "angular_segments": _ANGULAR_SEGMENTS,
            "height_segments": _HEIGHT_SEGMENTS,
            "cap_radial_segments": _CAP_RADIAL_SEGMENTS,
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
