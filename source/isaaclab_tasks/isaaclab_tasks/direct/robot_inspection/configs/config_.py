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
    def __init__(self, custom_name, num_faces, prim_path, usd_path=None, primitive=None,
                mesh_num_faces=None, scale=10.0, root_height=None,
                semantics_type = "class", semantics_name = "inspection_goal", orientation=(1.0, 0.0, 0.0, 0.0)):
        self.custom_name = custom_name
        # Expected reachable faces, used as the coverage/curriculum denominator.
        self.num_faces = num_faces
        # Actual mesh triangles, used for face-ID storage. Existing datasets
        # retain their previous behavior when no separate value is provided.
        self.mesh_num_faces = num_faces if mesh_num_faces is None else mesh_num_faces
        self.semantics_type = semantics_type
        self.semantics_name = semantics_name
        self.usd_path = usd_path
        self.primitive = primitive
        self.prim_path = prim_path
        self.scale = (float(scale), float(scale), float(scale)) if isinstance(scale, (int, float)) else scale
        self.root_height = root_height
        self.orientation = orientation

class Environment:
    def __init__(self, custom_name, usd_path, prim_path, 
                 semantics_type="class", semantics_name="inspection_goal", 
                 inspection_targets=None, scale=None, position=(0.0, 0.0, 0.0),
                 orientation=(1.0, 0.0, 0.0, 0.0), add_mesh_colliders=False):
        self.custom_name = custom_name
        self.semantics_type = semantics_type
        self.semantics_name = semantics_name
        self.usd_path = usd_path
        self.prim_path = prim_path
        self.inspection_targets = inspection_targets
        self.scale = scale
        self.position = position
        self.orientation = orientation
        self.add_mesh_colliders = add_mesh_colliders


def _environment_vector(name, default):
    """Read a three-component environment override with a clear error."""
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    values = tuple(float(value.strip()) for value in raw_value.split(","))
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three comma-separated numbers")
    return values

inspection_datasets = {}

from .data_set import usd_data_set
for key, value in usd_data_set.items():
    inspection_datasets[key] = Inpsection_Target(
        custom_name = key,
        num_faces = value["num_faces"],
        usd_path = value.get("usd_path"),
        prim_path = value["prim_path"],
        primitive = value.get("primitive"),
        mesh_num_faces = value.get("mesh_num_faces"),
        scale = value.get("scale", 10.0),
        root_height = value.get("root_height"),
        orientation = value.get("orientation", (1.0, 0.0, 0.0, 0.0))
    )

environment_usd_override = os.environ.get("ISAACLAB_INSPECTION_ENV_USD")
environment_name = os.environ.get("ISAACLAB_INSPECTION_ENV_NAME", "simple_warehouse")
environment_scale = _environment_vector("ISAACLAB_INSPECTION_ENV_SCALE", None)
environment_position = _environment_vector("ISAACLAB_INSPECTION_ENV_OFFSET", (0.0, 0.0, 0.0))
environment_add_mesh_colliders = os.environ.get(
    "ISAACLAB_INSPECTION_ENV_ADD_MESH_COLLIDERS", "0"
).lower() in {"1", "true", "yes", "on"}

inspection_environment = Environment(
        custom_name=environment_name,
        # These would be global even if we randomize the inspection goal
        semantics_type = "class",
        semantics_name = list(usd_data_set.keys()),
        usd_path = environment_usd_override or f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd",
        prim_path = "/World/envs/env_.*/warehouse",
        inspection_targets = inspection_datasets,
        scale = environment_scale,
        position = environment_position,
        add_mesh_colliders = environment_add_mesh_colliders,
    )

env_parameters = inspection_environment
