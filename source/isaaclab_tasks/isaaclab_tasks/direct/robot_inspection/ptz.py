# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""
PTZ Camera Assembly in Isaac Sim using Isaac Lab spawners
"""

import argparse
from isaaclab.app import AppLauncher
import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg

# Parse arguments
parser = argparse.ArgumentParser(description="PTZ Camera Assembly")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch the app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import after app launch
import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
from pxr import UsdPhysics, Gf
from omni.physx.scripts import utils as physx_utils
import omni.usd

def create_ptz_camera():
    """Create PTZ camera assembly with two revolute joints"""
    
    # Get stage
    stage = omni.usd.get_context().get_stage()
    
    # Create parent Xform for PTZ assembly
    prim_utils.create_prim("/World/PTZ", "Xform")
    
    # 1. BASE CYLINDER (Pan mechanism) - rotates around Z axis
    cfg_base = sim_utils.CylinderCfg(
        radius=0.05,
        height=0.1,
        axis="Z",  # Cylinder oriented along Z
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2)),
    )
    cfg_base.func("/World/PTZ/base_cylinder", cfg_base, translation=(0.0, 0.0, 0.05))
    
    # 2. MOUNTING BRACKET (connects pan to tilt) - rectangle/cuboid
    cfg_bracket = sim_utils.CuboidCfg(
        size=(0.08, 0.06, 0.03),  # Rectangular bracket
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3)),
    )
    cfg_bracket.func("/World/PTZ/mounting_bracket", cfg_bracket, translation=(0.0, 0.0, 0.125))
    
    # 3. TILT CYLINDER - rotates around Y axis (perpendicular to pan)
    cfg_tilt = sim_utils.CylinderCfg(
        radius=0.03,
        height=0.08,
        axis="Y",  # Cylinder oriented along Y (horizontal)
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.3),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.4)),
    )
    cfg_tilt.func("/World/PTZ/tilt_cylinder", cfg_tilt, translation=(0.0, 0.0, 0.17))
    
    # 4. CAMERA MOUNT (where sensor attaches) - small box
    cfg_camera_mount = sim_utils.CuboidCfg(
        size=(0.04, 0.06, 0.04),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.5)),
    )
    cfg_camera_mount.func("/World/PTZ/camera_mount", cfg_camera_mount, translation=(0.0, 0.03, 0.17))
    
    return stage

def add_articulation_and_joints(stage):
    """Add articulation root and revolute joints to PTZ assembly"""
    
    # Apply ArticulationRootAPI to base cylinder
    base_prim = stage.GetPrimAtPath("/World/PTZ/base_cylinder")
    UsdPhysics.ArticulationRootAPI.Apply(base_prim)
    
    print("[INFO] Applied ArticulationRootAPI to base_cylinder")
    
    # Get prims for joint creation
    base_cyl = stage.GetPrimAtPath("/World/PTZ/base_cylinder")
    bracket = stage.GetPrimAtPath("/World/PTZ/mounting_bracket")
    tilt_cyl = stage.GetPrimAtPath("/World/PTZ/tilt_cylinder")
    camera_mount = stage.GetPrimAtPath("/World/PTZ/camera_mount")
    
    # Create Revolute Joint 1: Base (world) to base_cylinder (pan joint - Z axis)
    # This allows 360° rotation around Z axis
    pan_joint = physx_utils.createJoint(stage, "Revolute", base_cyl, bracket)
    pan_joint_prim = stage.GetPrimAtPath(pan_joint.GetPath())
    
    # Configure pan joint - rotate around Z axis
    revolute_joint_1 = UsdPhysics.RevoluteJoint(pan_joint_prim)
    revolute_joint_1.CreateAxisAttr("Z")  # Pan around Z axis
    revolute_joint_1.CreateLowerLimitAttr(-180)  # Degrees
    revolute_joint_1.CreateUpperLimitAttr(180)
    
    # Set joint frame positions (optional - adjust based on geometry)
    revolute_joint_1.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0.05))
    revolute_joint_1.CreateLocalPos1Attr(Gf.Vec3f(0, 0, -0.025))
    
    # Add drive for actuation (optional - for controlled movement)
    drive_api = UsdPhysics.DriveAPI.Apply(pan_joint_prim, "angular")
    drive_api.CreateTypeAttr("force")
    drive_api.CreateMaxForceAttr(1000.0)
    drive_api.CreateStiffnessAttr(0.0)  # For velocity/effort control
    drive_api.CreateDampingAttr(100.0)
    
    print(f"[INFO] Created pan joint: {pan_joint.GetPath()}")
    
    # Create Revolute Joint 2: Bracket to tilt_cylinder (tilt joint - Y axis)
    # This allows tilt motion (up/down), typically -90 to +90 degrees
    tilt_joint = physx_utils.createJoint(stage, "Revolute", bracket, tilt_cyl)
    tilt_joint_prim = stage.GetPrimAtPath(tilt_joint.GetPath())
    
    # Configure tilt joint - rotate around Y axis (perpendicular to pan)
    revolute_joint_2 = UsdPhysics.RevoluteJoint(tilt_joint_prim)
    revolute_joint_2.CreateAxisAttr("Y")  # Tilt around Y axis
    revolute_joint_2.CreateLowerLimitAttr(-90)
    revolute_joint_2.CreateUpperLimitAttr(90)
    
    revolute_joint_2.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0.025))
    revolute_joint_2.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    
    # Add drive for tilt actuation
    drive_api_2 = UsdPhysics.DriveAPI.Apply(tilt_joint_prim, "angular")
    drive_api_2.CreateTypeAttr("force")
    drive_api_2.CreateMaxForceAttr(1000.0)
    drive_api_2.CreateStiffnessAttr(0.0)
    drive_api_2.CreateDampingAttr(100.0)
    
    print(f"[INFO] Created tilt joint: {tilt_joint.GetPath()}")
    
    # Create Fixed Joint: Tilt cylinder to camera mount
    fixed_joint = physx_utils.createJoint(stage, "Fixed", tilt_cyl, camera_mount)
    print(f"[INFO] Created fixed joint: {fixed_joint.GetPath()}")

def add_camera_sensor(stage):
    """Add camera sensor to the camera mount"""
    from isaacsim.sensors.camera import Camera
    import numpy as np
    import isaacsim.core.utils.numpy.rotations as rot_utils
    


    # Create camera attached to camera_mount
    camera = Camera(
        prim_path="/World/PTZ/camera_mount/camera",
        update_period = 0.1,
        position=np.array([0.0, 0.05, 0.0]),  # Offset forward from mount
        height = 640,
        width = 640,
        data_types = ['rgb'],
        orientation=rot_utils.euler_angles_to_quats(np.array([0, 0, -90]), degrees=True),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.05, 0.),
            rot=(-0.5, 0.5, -0.5, 0.5),
            convention="ros"
        ),
    )
    camera.initialize()
    
    print("[INFO] Camera sensor added")
    return camera

def design_scene():
    """Design the complete scene with PTZ camera"""
    
    # Ground plane
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)
    
    # Light
    cfg_light = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg_light.func("/World/lightDistant", cfg_light, translation=(1, 0, 10))
    
    # Create PTZ camera assembly
    stage = create_ptz_camera()
    
    # Add articulation and joints
    add_articulation_and_joints(stage)
    
    # Add camera sensor
    camera = add_camera_sensor(stage)
    
    # Add some visual reference objects
    cfg_cube = sim_utils.CuboidCfg(
        size=(0.2, 0.2, 0.2),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
    )
    cfg_cube.func("/World/test_cube", cfg_cube, translation=(0.5, 0.0, 0.1))
    
    return camera

def main():
    """Main function"""
    
    # Initialize simulation
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    
    # Set camera view
    sim.set_camera_view([0.5, -0.5, 0.4], [0.0, 0.0, 0.15])
    
    # Design scene
    camera = design_scene()
    
    # Reset simulation
    sim.reset()
    
    print("[INFO]: PTZ Camera setup complete...")
    print("[INFO]: The PTZ has two revolute joints:")
    print("        - Pan joint (base): rotates around Z axis")
    print("        - Tilt joint (bracket->tilt): rotates around Y axis")
    
    # Simulate
    while simulation_app.is_running():
        sim.step()

if __name__ == "__main__":
    main()
    simulation_app.close()
