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



# Env_params = {
#     "Brick":{
#         "num_faces": 12_000,
#         "semantics_name": "Brick",
#         "file_name": "/home/tosin/Desktop/IsaacLab/environments/ware_house_semantic.usd",
#         "prim_path": "/World/ground/terrain/ware_house_brick/_61_foam_brick"
#     },

#     'Brick_default':{
#         "num_faces": 12_000,
#         "semantics_type": "class",
#         "semantics_name": "brick",
#         "file_name": "/home/tosin/IsaacLab_inspection/environments/ware_house_brick.usd",
#         "prim_path": "/World/ground/terrain/_61_foam_brick",

#     },
#     'complex_forklift':{
#         "num_faces": 21_000,
#         "semantics_type": "class",
#         "semantics_name": "forklift",
#         "file_name": "/home/tosin/IsaacLab_inspection/environments/small_forklift.usd",
#         "prim_path": "/World/ground/terrain/forklift"
#     },
#     'complex_forklift_2':{
#         "num_faces": 21_000,
#         "semantics_type": "class",
#         "semantics_name": "forklift",
#         "env_file_path": "/home/tosin/IsaacLab_inspection/environments/small_forklift.usd",
#         "inspection_goal_prim_path": "/World/envs/env_.*/warehouse/forklift",
#         "env_prim_path": "/World/envs/env_.*/warehouse"
#     },

#     # Currently using this
    
# }

class Inpsection_Target:
    def __init__(self, custom_name, num_faces, usd_path, prim_path, scale,
                semantics_type = "class", semantics_name = "inspection_goal", ):
        self.custom_name = custom_name
        self.num_faces = num_faces
        self.semantics_type = semantics_type
        self.semantics_name = semantics_name
        self.usd_path = usd_path
        self.prim_path = prim_path
        self.scale = scale

class Environment:
    def __init__(self, custom_name, usd_path, prim_path, inspection_targets=None, scale=None):
        self.custom_name = custom_name
        self.usd_path = usd_path
        self.prim_path = prim_path
        self.inspection_targets = inspection_targets
        self.scale = scale

inspection_datasets = [
    Inpsection_Target(
        custom_name="rubiks_cube",
        num_faces=2000,
        usd_path= f"{ISAAC_NUCLEUS_DIR}/Props/Rubiks_Cube/rubiks_cube.usd",
        prim_path="/World/envs/env_.*/rubiks_cube",
        scale=(10.0, 10.0, 10.0),
        semantics_type="class",
        semantics_name="inspection_goal",
    )
]

inspection_environment = Environment(
        custom_name="empty_room",
        # usd_path= f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse_with_forklifts.usd",
        usd_path= f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd",
        prim_path="/World/envs/env_.*/warehouse",
        inspection_targets=inspection_datasets[0],
    )