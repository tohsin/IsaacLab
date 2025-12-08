from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
ROBOT_CONFIGS = {
    "jackal": {
        "usd_path": f"{ISAAC_NUCLEUS_DIR}/Robots/Clearpath/Jackal/jackal_basic.usd",
        "wheel_joint_expr": ".*wheel.*",
        "action_space": 4  # 4 wheels
    },
    "jetbot": {
        "usd_path": f"{ISAAC_NUCLEUS_DIR}/Robots/Jetbot/jetbot.usd", 
        "wheel_joint_expr": ".*wheel.*",
        "action_space": 2  # 2 wheels
    },
}
Env_params = {

    'empty_room':{
        "num_faces": 4000,
        "semantics_type": "class",
        "semantics_name": "inspection_goal",
        "env_file_path": f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse_with_forklifts.usd",
        "env_prim_path": "/World/envs/env_.*/warehouse" ,
        "inspection_goal_prim_path": "/World/envs/env_.*/rubiks_cube",
    }
}

env_parameters = Env_params['empty_room']