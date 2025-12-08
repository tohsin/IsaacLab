# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""
PTZ Camera Assembly in Isaac Sim using Isaac Lab spawners
"""

import argparse

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="PTZ Camera Assembly")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch the app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import after app launch
import isaacsim.core.utils.prims as prim_utils
from omni.physx.scripts import utils as physx_utils
import isaaclab.sim as sim_utils


def create_ptz_camera(stage):
    """Create PTZ camera assembly with two revolute joints"""


    parent_prim_path = "/World/PTZ"
    # Create parent Xform for PTZ assembly
    prim_utils.create_prim(parent_prim_path, "Xform")
    
    # 1. BASE CYLINDER (Pan mechanism) - rotates around Z axis
    cfg_base = sim_utils.CylinderCfg(
        radius=0.05,
        height=0.05,
        axis="Z",  # Cylinder oriented along Z
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2)),
    )
    base_prim_path = f"{parent_prim_path}/base_cylinder"
    cfg_base.func(base_prim_path, cfg_base, translation=(0.0, 0.0, 0.05))
    
    joint_path = f"{parent_prim_path}/pan_joint"
    physx_utils.create_joint(
        stage,
        joint_type="Revolute",
        from_prim=parent_prim_path, # body0 (the static world/parent)
        to_prim=base_prim_path,     # body1 (the moving part)
        prim_path=joint_path,
    )
    joint_prim = prim_utils.get_prim_at_path(joint_path)
    joint_prim.GetAttribute("axis").Set("Z")

    # Now, apply drive properties using the correct schema
    cfg_pan_drive = sim_utils.JointDrivePropertiesCfg(
        drive_type="force",
        stiffness=1000.0,
        damping=100.0
    )
    # 2. MOUNTING BRACKET (connects pan to tilt) - rectangle/cuboid
    cfg_bracket = sim_utils.CuboidCfg(
        size=(0.03, 0.04, 0.2),  # Rectangular bracket
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3)),
    )
    cfg_bracket.func("/World/PTZ/mounting_bracket", cfg_bracket, translation=(0.0, 0.0, 0.125))

    
    # # 3. TILT CYLINDER - rotates around Y axis (perpendicular to pan)
    # cfg_tilt = sim_utils.CylinderCfg(   # # Add some visual reference objects
    
    #     radius=0.03,
    #     height=0.08,
    #     axis="Y",  # Cylinder oriented along Y (horizontal)
    #     rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    #     mass_props=sim_utils.MassPropertiesCfg(mass=0.3),
    #     collision_props=sim_utils.CollisionPropertiesCfg(),
    #     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.4)),
    # )
    # cfg_tilt.func("/World/PTZ/tilt_cylinder", cfg_tilt, translation=(0.0, 0.0, 0.17))


    

def design_scene(stage):
    """Design the complete scene with PTZ camera"""
    
    # Ground plane
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)
    
    # Light
    cfg_light = sim_utils.DistantLightCfg(
        intensity=3000.0, 
        color=(0.75, 0.75, 0.75))
    cfg_light.func("/World/lightDistant", cfg_light, translation=(1, 0, 10))
    
    # Create PTZ camera assembly
    create_ptz_camera(stage)
    

    
    
    # # Add some visual reference objects
    # cfg_cube = sim_utils.CuboidCfg(
    #     size=(0.2, 0.2, 0.2),
    #     rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    #     mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #     collision_props=sim_utils.CollisionPropertiesCfg(),
    #     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
    # )
    # cfg_cube.func("/World/test_cube", cfg_cube, translation=(0.5, 0.0, 0.1))
    


def main():
    """Main function"""
    
    # Initialize simulation
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    
    stage = sim.stage
    # Set camera view
    sim.set_camera_view([0.5, -0.5, 0.4], [0.0, 0.0, 0.15])
    
    # Design scene
    camera = design_scene(stage)
    
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
