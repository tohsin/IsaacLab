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
class Inpsection_Target:
    def __init__(self, custom_name, num_faces, usd_path, prim_path, scale=10.0,
                semantics_type = "class", semantics_name = "inspection_goal", orientation=(1.0, 0.0, 0.0, 0.0)):
        self.custom_name = custom_name
        self.num_faces = num_faces
        self.semantics_type = semantics_type
        self.semantics_name = semantics_name
        self.usd_path = usd_path
        self.prim_path = prim_path
        self.scale = (float(scale), float(scale), float(scale)) if isinstance(scale, (int, float)) else scale
        self.orientation = orientation

class Environment:
    def __init__(self, custom_name, usd_path, prim_path, 
                 semantics_type="class", semantics_name="inspection_goal", 
                 inspection_targets=None, scale=None):
        self.custom_name = custom_name
        self.semantics_type = semantics_type
        self.semantics_name = semantics_name
        self.usd_path = usd_path
        self.prim_path = prim_path
        self.inspection_targets = inspection_targets
        self.scale = scale

inspection_datasets = {}

from .data_set import usd_data_set
for key, value in usd_data_set.items():
    inspection_datasets[key] = Inpsection_Target(
        custom_name = key,
        num_faces = value["num_faces"],
        usd_path = value["usd_path"],
        prim_path = value["prim_path"],
        scale = value.get("scale", 10.0),
        orientation = value.get("orientation", (1.0, 0.0, 0.0, 0.0))
    )

inspection_environment = Environment(
        custom_name="empty_room",
        # These would be global even if we randomize the inspection goal
        semantics_type = "class",
        semantics_name = list(usd_data_set.keys()),
        usd_path = f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd",
        prim_path = "/World/envs/env_.*/warehouse",
        inspection_targets = inspection_datasets,
    )

env_parameters = inspection_environment