from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


evaluation_data_set = {
    # ConveyorBelt_A08 is intentionally excluded: it contains independent
    # root and Rollers rigid bodies, while inspection targets are represented
    # and reset as one RigidObject.
    'extinguisher': {
        'num_faces': 50_000,
        "usd_path": f"{ISAAC_NUCLEUS_DIR}/Environments/Office/Props/SM_Extinguisher.usd",
        "prim_path": "/World/envs/env_.*/extinguisher",
        "scale": 1.5,
    },
    'barrel':{
        'num_faces': 50_000,
        "usd_path": f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/Props/SM_BarelPlastic_B_02.usd",
        "prim_path": "/World/envs/env_.*/barrel",
        "scale": (2.0, 2.0, 2.0),
    },
    'dolly':{
        'num_faces': 50_000,
        # Use the composed/visual prop rather than its multi-body physics
        # layer. Inspection targets are intentionally wrapped as one rigid
        # object by inspection_env.py.
        "usd_path": f"{ISAAC_NUCLEUS_DIR}/Props/Dolly/dolly.usd",
        "prim_path": "/World/envs/env_.*/dolly",
        "scale": 1.0,
    },
    "small_corner_bracket_physics": {
        "num_faces": 2442,
        "usd_path": f"{ISAAC_NUCLEUS_DIR}/Props/Flip_Stack/small_corner_bracket_physics.usd",
        "prim_path": "/World/envs/env_.*/small_corner_bracket_physics",
        "scale": 50.0,
        "orientation": (0.7071068, 0.7071068, 0.0, 0.0),
    },

    "ur10_mount": {
            "num_faces": 50_000,#12000, #12494
            "usd_path": f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ur10_mount.usd",
            "prim_path": "/World/envs/env_.*/ur10_mount",
            "scale": (3.0, 3.0, 2.2),
            "orientation": (0.7071068, 0.7071068, 0.0, 0.0),
            # The transformed mesh extends 0.405 m below its root. Spawn at this
            # height so a kinematic mount rests on, rather than intersects, z=0.
            "root_height": 0.405,
        },
    "pallet": {
        "num_faces": 50_000, #10054,
        "usd_path": f"{ISAAC_NUCLEUS_DIR}/Props/Pallet/o3dyn_pallet.usd",
        "prim_path": "/World/envs/env_.*/pallet",
        "scale": 1.0,
        # This asset's root is authored at its base. Without an explicit value,
        # imported targets inherit the generic 0.3 m curriculum spawn height.
        "root_height": 0.0,
    },
    "caster": {
            "num_faces": 11397,
            "usd_path": f"{ISAAC_NUCLEUS_DIR}/Props/Flip_Stack/caster.usd",
            "prim_path": "/World/envs/env_.*/caster",
            "scale": (20.0, 20.0, 20.0),
            "orientation": (0.7071068, 0.7071068, 0.0, 0.0),
        },
}

    # "potted_meat_can": {
    #     "num_faces": 10763,
    #     "usd_path": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/010_potted_meat_can.usd",
    #     "prim_path": "/World/envs/env_.*/potted_meat_can",
    #     "scale": 10.0,
    #      "orientation": (0.7071068, -0.7071068, 0.0, 0.0),
    # },
    # "wood_block": {
    #     "num_faces": 12319,
    #     "usd_path": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/036_wood_block.usd",
    #     "prim_path": "/World/envs/env_.*/wood_block",
    #     "scale": 6.0,
    # },
    # "conveyor_belt": {
    #     "num_faces": 50_000,
    #     "usd_path": f"{ISAAC_NUCLEUS_DIR}/Props/Conveyors/ConveyorBelt_A08.usd",
    #     "prim_path": "/World/envs/env_.*/conveyor_belt",
    #     "scale": (0.5, 1.0, 1.0),
    # },
    # "forklift": {
    #     "num_faces": 38695,
    #     "usd_path": f"{ISAAC_NUCLEUS_DIR}/Props/Forklift/forklift.usd",
    #     "prim_path": "/World/envs/env_.*/forklift",
    #     "scale": 0.6,
    # },
    # "blue_cup": {
    #     "num_faces": 10803,
    #     "usd_path": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/019_pitcher_base.usd",
    #     "prim_path": "/World/envs/env_.*/blue_cup",
    #     "scale": 4.0,
    #     "orientation": (-0.7071068, 0.7071068, 0.0, 0.0),
    # },
    #     "tuna_fish_can": {
    #     "num_faces": 15000, #15000
    #     "usd_path": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/007_tuna_fish_can.usd",
    #     "prim_path": "/World/envs/env_.*/tuna_fish_can",
    # },
    # "tuna_fish_can_flat": {
    #     "num_faces": 4643,
    #     "usd_path": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/007_tuna_fish_can.usd",
    #     "prim_path": "/World/envs/env_.*/tuna_fish_can_flat",
    #     "orientation": (0.7071068, 0.7071068, 0.0, 0.0),
    # },
    #     "red_bowl": {
    #     "num_faces": 5831,
    #     "usd_path": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/024_bowl.usd",
    #     "prim_path": "/World/envs/env_.*/red_bowl",
    #     "scale": 10.0,
    #     "orientation": (0.7071068, 0.7071068, 0.0, 0.0),
    # },
    # "sortbot_housing": {
    #         "num_faces": 1779,
    #         "usd_path": f"{ISAAC_NUCLEUS_DIR}/Props/Sortbot_Housing/sortbot_housing.usd",
    #         "prim_path": "/World/envs/env_.*/sortbot_housing",
    #         "scale": [1.0, 1.0, 0.7],
    #     },
    #     "rubiks_cube": {
    #         "num_faces": 3800,
    #         "usd_path": f"{ISAAC_NUCLEUS_DIR}/Props/Rubiks_Cube/rubiks_cube.usd",
    #         "prim_path": "/World/envs/env_.*/rubiks_cube",
    #     },
