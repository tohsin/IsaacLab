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
_C_HOUSING_FACES = _subdivided_faces(28, 3)  # 1,792
_U_HOUSING_FACES = _subdivided_faces(28, 3)  # 1,792
_STEPPED_BLOCK_FACES = _subdivided_faces(28, 3)  # 1,792
_THIN_LEGGED_BODY_FACES = _subdivided_faces(28, 3)  # 1,792
# The low-arm generator preserves 264 base triangles as its snapped arm width
# changes: hidden middle-cylinder faces are replaced by arm shelf faces.
_LOW_ARM_FACES = _subdivided_faces(264, 1)  # 1,056
# The exact box union has 34 exposed quads before subdivision.
_OVERHANG_FACES = _subdivided_faces(68, 2)  # 1,088
_SPHERE_FACES = 20 * 4**3  # 1,280-face icosphere
_ANGULAR_SEGMENTS = 64
_HEIGHT_SEGMENTS = 32
_CAP_RADIAL_SEGMENTS = 8
_CYLINDER_SIDE_FACES = 2 * _ANGULAR_SEGMENTS * _HEIGHT_SEGMENTS  # 4,096
_CYLINDER_FACES = _cylinder_faces(_ANGULAR_SEGMENTS, _HEIGHT_SEGMENTS, _CAP_RADIAL_SEGMENTS)  # 6,016
_CONE_SIDE_FACES = _ANGULAR_SEGMENTS * (2 * _HEIGHT_SEGMENTS - 1)  # 4,032
_CONE_FACES = _cone_faces(_ANGULAR_SEGMENTS, _HEIGHT_SEGMENTS, _CAP_RADIAL_SEGMENTS)  # 4,992
PI = 3.141592653589793
yaw_range = (-PI, PI)

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
    "size_min": (0.8, 0.8, 1.0),
    "size_max": (1.2, 1.2, 1.5),


    # A cuboid's appearance changes under yaw, unlike a sphere or an upright
    # axial primitive. Sample the full circle at every randomized reset.
    "yaw_range": yaw_range,
}

# Upright T-block: requires higher minimum height so the underside faces of
# the horizontal bar can be inspected without ground occlusion or collision.
_T_BLOCK_RANDOMIZATION = {
    **_COMMON_RANDOMIZATION,
    "size_min": (0.8, 0.8, 1.5),
    "size_max": (1.2, 1.2, 1.7),
    "yaw_range": yaw_range,
}

_C_HOUSING_RANDOMIZATION = {
    **_COMMON_RANDOMIZATION,
    "size_min": (0.9, 0.8, 1.0),
    "size_max": (1.6, 1.5, 1.5),
    "opening_width_fraction_min": 0.35,
    "opening_width_fraction_max": 0.70,
    "pocket_depth_fraction_min": 0.40,
    "pocket_depth_fraction_max": 0.85,
    "yaw_range":yaw_range,
}

_U_HOUSING_RANDOMIZATION = {
    **_COMMON_RANDOMIZATION,

    "size_min": (0.9, 0.8, 1.0),
    "size_max": (1.6, 1.5, 1.45),
    "opening_width_fraction_min": 0.35,
    "opening_width_fraction_max": 0.70,
    "pocket_floor_height_min": 0.15,
    "pocket_floor_height_max": 0.30,
    "yaw_range":yaw_range,
}

_LOW_ARM_RANDOMIZATION = {
    **_COMMON_RANDOMIZATION,
    "body_radius_min": 0.30,
    "body_radius_max": 0.55,
    "body_height_min": 1.0,
    "body_height_max": 1.5,
    "arm_length_min": 0.4,
    "arm_length_max": 1.0,
    # Width and thickness deliberately cross the 0.2 m occupancy resolution.
    "arm_width_min": 0.10,
    "arm_width_max": 0.32,
    "arm_thickness_min": 0.08,
    "arm_thickness_max": 0.20,
    # "arm_clearance_min": 0.03,
    # The upper arm face remains below the roughly 0.42 m PTZ optical center,
    # so it is legitimately part of the reachable-face denominator.
    "arm_clearance_max": 0.14,
    "yaw_range": yaw_range,
}

_STEPPED_BLOCK_RANDOMIZATION = {
    **_COMMON_RANDOMIZATION,
    # Size remains world-space (width, depth, height). The staircase is cut
    # into the horizontal XY footprint, so the final component is still Z.
    # Unlike the other composite targets, honor these bounds in every variant
    # so a one-environment geometry check does not fall back to the authored size.
    "include_authored_variant": False,
    "size_min": (0.9, 0.7, 1.0),
    "size_max": (1.7, 1.3, 1.5),
    "first_step_x_fraction_min": 0.22,
    "first_step_x_fraction_max": 0.42,
    "second_step_x_fraction_min": 0.60,
    "second_step_x_fraction_max": 0.80,
    # These legacy "height" names now position the two turns along footprint
    # depth. Keeping the keys avoids invalidating existing experiment configs.
    "low_step_height_fraction_min": 0.50,
    "low_step_height_fraction_max": 0.62,
    "middle_step_height_fraction_min": 0.68,
    "middle_step_height_fraction_max": 0.82,
    "yaw_range": yaw_range,
}

_THIN_LEGGED_BODY_RANDOMIZATION = {
    **_COMMON_RANDOMIZATION,
    "depth_min": 0.55,
    "depth_max": 1.10,
    "gap_width_min": 0.30,
    "gap_width_max": 0.90,
    "left_leg_width_min": 0.10,
    "left_leg_width_max": 0.30,
    "right_leg_width_min": 0.10,
    "right_leg_width_max": 0.30,
    # Match the upright T's 1.05 m minimum inspectable underside clearance.
    "underside_height_min": 1.05,
    "underside_height_max": 1.40,
    "body_thickness_min": 0.22,
    "body_thickness_max": 0.38,
    "yaw_range": yaw_range,
}

_OVERHANG_RANDOMIZATION = {
    **_COMMON_RANDOMIZATION,
    # As with the T, the plate underside is included only because its minimum
    # clearance is kept at 1.10 m.
    "support_size_min": (0.35, 0.35, 1.10),
    "support_size_max": (0.65, 0.65, 1.5),
    "plate_size_min": (1.0, 1.0, 0.14),
    "plate_size_max": (1.7, 1.7, 0.30),
    # Fraction of the available travel before a support edge reaches a plate edge.
    "offset_fraction_min": -0.80,
    "offset_fraction_max": 0.80,
    "plate_offset_max": 0.35,
    "yaw_range": yaw_range,
}


_HIGH_CUBOID_RANDOMIZATION = {
    **_COMMON_RANDOMIZATION,
    "size_min": (0.8, 1.2, 0.8),
    "size_max": (1.3, 1.5, 1.3),
    "yaw_range": yaw_range,
}

_SPHERE_RANDOMIZATION = {
    **_COMMON_RANDOMIZATION,
    "radius_min": 0.6,
    "radius_max": 1.0,
}

_AXIAL_RANDOMIZATION = {
    **_COMMON_RANDOMIZATION,
    "radius_min": 1.1, # 
    "radius_max": 1.1,
    "height_min": 1.3,
    "height_max": 2.0, # 2.0
}
_AXIAL_RANDOMIZATION_FLAT = {
    **_COMMON_RANDOMIZATION,
    # The backend now interprets 'height' as the vertical World Z height for flat objects,
    # and 'length' as the longitudinal length (X-axis).
    "height_min": 0.8,
    "height_max": 1.5,
    "length_min": 1.0,
    "length_max": 1.8, # 1.8
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
            "domain_randomization": {**_T_BLOCK_RANDOMIZATION},
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
                "yaw_range": yaw_range,
            },
        },
    ),

    "tessellated_c_housing": _target(
        "tessellated_c_housing",
        # Six bottom-cap and six top-cap base triangles are excluded.  The
        # sixteen vertical triangles include the outer walls and pocket walls.
        reachable_faces=_C_HOUSING_FACES * 16 // 28,  # 1,024
        mesh_faces=_C_HOUSING_FACES,
        primitive={
            "type": "tessellated_c_housing",
            "size": (1.2, 1.15, 1.1),
            "subdivisions": 3,
            "opening_width_fraction": 0.55,
            "pocket_depth_fraction": 0.68,
            "domain_randomization": {**_C_HOUSING_RANDOMIZATION},
        },
    ),

    "tessellated_u_housing": _target(
        "tessellated_u_housing",
        # Rests on the bottom face (2 base triangles) and the robot camera is too low
        # to see the two top strips (2 base triangles each). The pocket floor is visible
        # from the open front/back of the trench. Total excluded: 6 base triangles.
        reachable_faces=_U_HOUSING_FACES * 22 // 28,
        mesh_faces=_U_HOUSING_FACES,
        primitive={
            "type": "tessellated_u_housing",
            "size": (1.2, 1.15, 1.1),
            "subdivisions": 3,
            "opening_width_fraction": 0.55,
            "pocket_floor_height": 0.25,
            "domain_randomization": {**_U_HOUSING_RANDOMIZATION},
        },
    ),

    "tessellated_low_arm": _target(
        "tessellated_low_arm",
        # The mesh has 264 base triangles. For the authored half_hidden=1 shape,
        # excluding both 32-triangle pedestal caps and the 3-triangle arm
        # underside leaves 197 base triangles, or 788 after one subdivision.
        # Wider randomized arms use half_hidden=2 or 3 and yield 780 or 772.
        reachable_faces=788,
        mesh_faces=_LOW_ARM_FACES,
        primitive={
            "type": "tessellated_low_arm",
            "subdivisions": 1,
            "body_radius": 0.4,
            "body_height": 1.0,
            "arm_length": 0.65,
            "arm_width": 0.2,
            "arm_thickness": 0.16,
            "arm_clearance": 0.08,
            "angular_segments": 32,
            "domain_randomization": {**_LOW_ARM_RANDOMIZATION},
        },
    ),

    "tessellated_stepped_block": _target(
        "tessellated_stepped_block",
        # The staircase is an XY footprint extruded along world Z. Excluding
        # its two six-triangle caps leaves 16 * 4**3 = 1,024 reachable faces.
        reachable_faces=_STEPPED_BLOCK_FACES * 16 // 28,
        mesh_faces=_STEPPED_BLOCK_FACES,
        primitive={
            "type": "tessellated_stepped_block",
            "size": (1.25, 0.9, 1.2),
            "subdivisions": 3,
            "step_x_fractions": (0.34, 0.68),
            "step_height_fractions": (0.55, 0.75),
            "domain_randomization": {**_STEPPED_BLOCK_RANDOMIZATION},
        },
    ),

    "tessellated_thin_legged_body": _target(
        "tessellated_thin_legged_body",
        # Exclude four base triangles beneath the feet and two top triangles.
        # The elevated two-triangle underside is counted because its clearance
        # is constrained to the same minimum used by the upright T.
        reachable_faces=_THIN_LEGGED_BODY_FACES * 22 // 28,  # 1,408
        mesh_faces=_THIN_LEGGED_BODY_FACES,
        primitive={
            "type": "tessellated_thin_legged_body",
            "subdivisions": 3,
            "depth": 0.8,
            "gap_width": 0.55,
            "left_leg_width": 0.18,
            "right_leg_width": 0.22,
            "underside_height": 1.05,
            "body_thickness": 0.30,
            "domain_randomization": {**_THIN_LEGGED_BODY_RANDOMIZATION},
        },
    ),

    "tessellated_overhang": _target(
        "tessellated_overhang",
        # Of 34 exterior quads, exclude one support floor quad and the nine
        # subdivided top-plate quads.  The other eight interface quads form the
        # inspectable underside around the support.
        reachable_faces=_subdivided_faces(48, 2),
        mesh_faces=_OVERHANG_FACES,
        primitive={
            "type": "tessellated_overhang",
            "subdivisions": 2,
            "support_size": (0.5, 0.5, 1.10),
            "plate_size": (1.25, 1.25, 0.22),
            "plate_offset": (0.0, 0.0),
            "domain_randomization": {**_OVERHANG_RANDOMIZATION},
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
        # Provisional discrete-face estimate for a ground robot whose camera
        # remains below the center of the randomized sphere.
        reachable_faces=int(_SPHERE_FACES * 0.9),
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
                **_AXIAL_RANDOMIZATION_FLAT,
                # Rotate only around world Z so the cylinder remains flat.
                "yaw_range": yaw_range,
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
                **_AXIAL_RANDOMIZATION_FLAT,
                # Rotate around world Z while preserving the horizontal axis.
                "yaw_range": (-3.141592653589793, 3.141592653589793),
            },
        },
    ),
}

# Preserve the name used while the primitives were first being tested.
primitive_test_data_set = primitive_data_set
