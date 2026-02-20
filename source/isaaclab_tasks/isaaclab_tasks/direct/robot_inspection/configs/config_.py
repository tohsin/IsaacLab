from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import os

# Resolve repository root
# File: source/isaaclab_tasks/isaaclab_tasks/direct/robot_inspection/configs/config_.py
# Root: ../../../../../../
CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ISAACLAB_REPO_ROOT = os.path.abspath(os.path.join(CURRENT_FILE_DIR, "../../../../../../"))

ROBOT_CONFIGS = {
    "jackal": {
        "usd_path": f"{ISAAC_NUCLEUS_DIR}/Robots/Clearpath/Jackal/jackal_basic.usd",
        "wheel_joint_expr": ".*wheel.*",
        "action_space": 4  # 4 wheels
    },
    "jackal_ptz": {
        "usd_path": os.path.join(ISAACLAB_REPO_ROOT, "assets/jackal_basic_ptz_o.usd"),
        "wheel_joint_expr": ".*wheel.*",
        "ptz_joint_expr":  ".*ptz.*",
        "action_space": 6  # 4 +2 wheels and ptz
    },
    "jetbot": {
        "usd_path": f"{ISAAC_NUCLEUS_DIR}/Robots/Jetbot/jetbot.usd", 
        "wheel_joint_expr": ".*wheel.*",
        "action_space": 2  # 2 wheels
    },
}
Env_params = {
    "Brick":{
        "num_faces": 12_000,
        "semantics_name": "Brick",
        "file_name": "/home/tosin/Desktop/IsaacLab/environments/ware_house_semantic.usd",
        "prim_path": "/World/ground/terrain/ware_house_brick/_61_foam_brick"
    },

    'Brick_default':{
        "num_faces": 12_000,
        "semantics_type": "class",
        "semantics_name": "brick",
        "file_name": "/home/tosin/IsaacLab_inspection/environments/ware_house_brick.usd",
        "prim_path": "/World/ground/terrain/_61_foam_brick",

    },
    'complex_forklift':{
        "num_faces": 21_000,
        "semantics_type": "class",
        "semantics_name": "forklift",
        "file_name": "/home/tosin/IsaacLab_inspection/environments/small_forklift.usd",
        "prim_path": "/World/ground/terrain/forklift"
    },
    'complex_forklift_2':{
        "num_faces": 21_000,
        "semantics_type": "class",
        "semantics_name": "forklift",
        "env_file_path": "/home/tosin/IsaacLab_inspection/environments/small_forklift.usd",
        "inspection_goal_prim_path": "/World/envs/env_.*/warehouse/forklift",
        "env_prim_path": "/World/envs/env_.*/warehouse"
    },

    # Currently using this
    'empty_room':{
        # Environment details
        "env_file_path": f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse_with_forklifts.usd",
        "env_prim_path": "/World/envs/env_.*/warehouse" , 
        # Object details
        "inspection_goal_prim_path": "/World/envs/env_.*/rubiks_cube",
        "inspection_goal_usd_path": f"{ISAAC_NUCLEUS_DIR}/Props/Rubiks_Cube/rubiks_cube.usd",
        "inspection_goal_scale": (10.0, 10.0, 10.0),
        "num_faces": 2000,
        "semantics_type": "class",
        "semantics_name": "inspection_goal",
    }
}

env_parameters = Env_params['empty_room']