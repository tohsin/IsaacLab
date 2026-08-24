import torch
import isaaclab.sim as sim_utils
from isaaclab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
import random

class ObstacleDatasetHandler:
    def __init__(self, max_obstacles: int = 10):
        self.max_obstacles = max_obstacles
        self.horizontal_radii: list[float] = []
        # Use a fixed seed for obstacle generation so they are consistent across runs
        # The randomize aspect will handle positions in the curriculum
        self.rng = random.Random(42)

    def get_obstacle_configs(self) -> list[RigidObjectCfg]:
        configs = []
        self.horizontal_radii.clear()
        for i in range(self.max_obstacles):
            # random choice between cylinder, cuboid
            obstacle_type = self.rng.choice(["cylinder", "cuboid", "sphere", "cone"])
            
            # Generate random color for visual variety
            color = (self.rng.random(), self.rng.random(), self.rng.random())
            
            if obstacle_type == "cylinder":
                radius = self.rng.uniform(0.2, 0.5)
                horizontal_radius = radius
                height = self.rng.uniform(0.5, 2.5)
                spawn_cfg = sim_utils.CylinderCfg(
                    radius=radius,
                    height=height,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(density=500.0, mass=1000.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                )
            elif obstacle_type == "cuboid":
                size_x = self.rng.uniform(0.3, 1.0)
                size_y = self.rng.uniform(0.3, 1.0)
                size_z = self.rng.uniform(0.5, 2.5)
                # Obstacles currently keep identity yaw, so the circumscribed
                # XY circle is a conservative footprint for spawn clearance.
                horizontal_radius = 0.5 * (size_x**2 + size_y**2) ** 0.5
                spawn_cfg = sim_utils.CuboidCfg(
                    size=(size_x, size_y, size_z),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(density=500.0, mass=1000.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                )
            elif obstacle_type == "sphere":
                radius = self.rng.uniform(0.2, 0.5)
                horizontal_radius = radius
                spawn_cfg = sim_utils.SphereCfg(
                    radius=radius,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(density=500.0, mass=1000.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                )
            elif obstacle_type == "cone":
                radius = self.rng.uniform(0.2, 0.5)
                horizontal_radius = radius
                height = self.rng.uniform(0.5, 2.0)
                spawn_cfg = sim_utils.ConeCfg(
                    radius=radius,
                    height=height,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(density=500.0, mass=1000.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                )
                
            cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/obstacle_{i}",
                spawn=spawn_cfg,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -100.0)) # Start hidden far below
            )
            configs.append(cfg)
            self.horizontal_radii.append(horizontal_radius)
        return configs

    def get_horizontal_radii(self) -> list[float]:
        """Return conservative XY footprint radii in obstacle-config order."""
        return self.horizontal_radii.copy()
