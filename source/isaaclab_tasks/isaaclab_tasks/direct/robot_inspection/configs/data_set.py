from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
usd_data_set_pre_train = {
     "rubiks_cube": {
        "num_faces": 3800,
        "usd_path": f"{ISAAC_NUCLEUS_DIR}/Props/Rubiks_Cube/rubiks_cube.usd",
        "prim_path": "/World/envs/env_.*/rubiks_cube",
    },
     "tuna_fish_can_flat": {
        "num_faces": 4643,
        "usd_path": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/007_tuna_fish_can.usd",
        "prim_path": "/World/envs/env_.*/tuna_fish_can_flat",
        "orientation": (0.7071068, 0.7071068, 0.0, 0.0),
    },
    "wood_block": {
        "num_faces": 12319,
        "usd_path": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/036_wood_block.usd",
        "prim_path": "/World/envs/env_.*/wood_block",
        "scale": 6.0,
    }, 
    "blue_cup": {
        "num_faces": 10803,
        "usd_path": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/019_pitcher_base.usd",
        "prim_path": "/World/envs/env_.*/blue_cup",
        "scale": 4.0,
        "orientation": (-0.7071068, 0.7071068, 0.0, 0.0),
    },
}
usd_data_set_finetune = {
    "rubiks_cube": {
        "num_faces": 3800,
        "usd_path": f"{ISAAC_NUCLEUS_DIR}/Props/Rubiks_Cube/rubiks_cube.usd",
        "prim_path": "/World/envs/env_.*/rubiks_cube",
    },
    "tuna_fish_can": {
        "num_faces": 15000, 
        "usd_path": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/007_tuna_fish_can.usd",
        "prim_path": "/World/envs/env_.*/tuna_fish_can",
    },
    "tuna_fish_can_flat": {
        "num_faces": 4643,
        "usd_path": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/007_tuna_fish_can.usd",
        "prim_path": "/World/envs/env_.*/tuna_fish_can_flat",
        "orientation": (0.7071068, 0.7071068, 0.0, 0.0),
    },
    "blue_cup": {
        "num_faces": 10803,
        "usd_path": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/019_pitcher_base.usd",
        "prim_path": "/World/envs/env_.*/blue_cup",
        "scale": 4.0,
        "orientation": (-0.7071068, 0.7071068, 0.0, 0.0),
    },
    # contiue calibaration here
    "red_bowl": {
        "num_faces": 5831,
        "usd_path": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/024_bowl.usd", 
        "prim_path": "/World/envs/env_.*/red_bowl",
        "scale": 10.0,
        "orientation": (0.7071068, 0.7071068, 0.0, 0.0),
    },
    # "forklift": {
    #     "num_faces": 38695,
    #     "usd_path": f"{ISAAC_NUCLEUS_DIR}/Props/Forklift/forklift.usd",
    #     "prim_path": "/World/envs/env_.*/forklift",
    #     "scale": 0.6,
    # },
    "ur10_mount": {
        "num_faces": 12494, # 11910
        "usd_path": f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ur10_mount.usd",
        "prim_path": "/World/envs/env_.*/ur10_mount",
        "scale": 3.0,
        "orientation": (0.7071068, 0.7071068, 0.0, 0.0),
    },
   "pallet": {
        "num_faces": 10054,
        "usd_path": f"{ISAAC_NUCLEUS_DIR}/Props/Pallet/o3dyn_pallet.usd",
        "prim_path": "/World/envs/env_.*/pallet",
        "scale": 1.0,
    },
#    "sortbot_housing": {
#         "num_faces": 1779,
#         "usd_path": f"{ISAAC_NUCLEUS_DIR}/Props/Sortbot_Housing/sortbot_housing.usd",
#         "prim_path": "/World/envs/env_.*/sortbot_housing",
#         "scale": 1.0,
#     },
    "potted_meat_can": {
        "num_faces": 10763,
        "usd_path": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/010_potted_meat_can.usd",
        "prim_path": "/World/envs/env_.*/potted_meat_can",
        "scale": 10.0,
         "orientation": (0.7071068, -0.7071068, 0.0, 0.0),
    },
    "wood_block": {
        "num_faces": 12319,
        "usd_path": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/036_wood_block.usd",
        "prim_path": "/World/envs/env_.*/wood_block",
        "scale": 6.0,
    }, 
    "small_corner_bracket_physics": {
        "num_faces": 2442,
        "usd_path": f"{ISAAC_NUCLEUS_DIR}/Props/Flip_Stack/small_corner_bracket_physics.usd",
        "prim_path": "/World/envs/env_.*/small_corner_bracket_physics",
        "scale": 50.0,
        "orientation": (0.7071068, 0.7071068, 0.0, 0.0),
    },
    "caster": {
        "num_faces": 11397,
        "usd_path": f"{ISAAC_NUCLEUS_DIR}/Props/Flip_Stack/caster.usd",
        "prim_path": "/World/envs/env_.*/caster",
        "scale": 15.0,
        "orientation": (0.7071068, 0.7071068, 0.0, 0.0),
    },
}
# During evaluation we test one obejt at at time for now we  want to see if we have annurate estiamtion of number of obejhcts so we
#overstimate the number of faces to increase potentially ceeling to improve dataset estimation
# We pretrain on single easy object fine tune on harder objects


def get_object_eval(object_ = ""):
    return {
        object_: usd_data_set_finetune[object_]
    }
