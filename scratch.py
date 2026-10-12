import numpy as np
import trimesh
import sys
sys.path.append("/home/tosin/Documents/GitHub/IsaacLab/source/isaaclab_tasks")
from isaaclab_tasks.direct.robot_inspection.utils.tessellated_primitives import _create_low_arm_mesh

try:
    mesh = _create_low_arm_mesh(
        body_radius=0.4,
        body_height=1.0,
        arm_length=0.65,
        arm_width=0.2,
        arm_thickness=0.16,
        arm_clearance=0.08,
        angular_segments=32,
    )
    print("Mesh generated successfully!")
    print(f"Vertices: {len(mesh.vertices)}")
    print(f"Faces: {len(mesh.faces)}")
    print(f"Watertight: {mesh.is_watertight}")
    print(f"Bounds: {mesh.bounds}")
except Exception as e:
    print(f"Error: {e}")
