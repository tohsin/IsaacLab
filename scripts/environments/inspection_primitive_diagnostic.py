# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded smoke test for the tessellated primitive inspection target."""

import argparse
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Validate primitive face IDs and semantic masking.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_steps", type=int, default=24)
parser.add_argument("--task", type=str, default="Isaac-Inspection-Camera-Direct-v0")
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from pxr import UsdGeom

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.math import quat_apply
from isaaclab_tasks.direct.robot_inspection import run_config
from isaaclab_tasks.utils import parse_env_cfg


def main():
    # Make the diagnostic deterministic and remove obstacle occlusion.
    run_config.cfg_mode.fixed_spawns = True
    run_config.cfg_mode.randomize_spawns = False
    run_config.cfg_mode.use_hardest_curriculum = False
    run_config.cfg_mode.is_simplified = True
    run_config.cfg_mode.use_wandb = False

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = 42
    env_cfg.max_obstacles = 0

    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        env.reset()
        unwrapped = env.unwrapped

        target_name = str(unwrapped.env_target_names[0])
        target_cfg = unwrapped.cfg.inspection_goal_cfg.inspection_targets[target_name]
        target_prim_path = target_cfg.prim_path.replace("env_.*", "env_0")
        mesh_prim = UsdGeom.Mesh(
            unwrapped.scene.stage.GetPrimAtPath(
                f"{target_prim_path}/geometry/mesh"
            )
        )
        authored_faces = len(mesh_prim.GetFaceVertexCountsAttr().Get())
        points = torch.tensor(mesh_prim.GetPointsAttr().Get(), dtype=torch.float64)
        face_indices = torch.tensor(mesh_prim.GetFaceVertexIndicesAttr().Get(), dtype=torch.long).view(-1, 3)
        triangle_z = points[face_indices, 2]
        local_min_z = points[:, 2].min()
        bottom_face_mask = torch.all(torch.isclose(triangle_z, local_min_z, atol=1.0e-9), dim=1)
        bottom_face_ids = set(torch.nonzero(bottom_face_mask, as_tuple=False).flatten().tolist())
        non_bottom_face_ids = set(torch.nonzero(~bottom_face_mask, as_tuple=False).flatten().tolist())

        print(f"[PRIMITIVE TEST] Target: {target_name}")
        print(f"[PRIMITIVE TEST] Authored triangle faces: {authored_faces}")
        configured_faces = int(unwrapped.total_mesh_faces[0].item())
        reachable_faces = int(unwrapped.coverage_num_faces[0].item())
        print(f"[PRIMITIVE TEST] Configured triangle faces: {configured_faces}")
        print(f"[PRIMITIVE TEST] Reachable-face denominator: {reachable_faces}")
        print(f"[PRIMITIVE TEST] Local mesh Z bounds: {float(points[:, 2].min())}..{float(points[:, 2].max())}")
        print(f"[PRIMITIVE TEST] Bottom/non-bottom triangles: {len(bottom_face_ids)}/{len(non_bottom_face_ids)}")
        print(f"[PRIMITIVE TEST] Environment sizes: {unwrapped.primitive_sizes[target_name].cpu().tolist()}")
        print(f"[PRIMITIVE TEST] Environment colors: {unwrapped.primitive_colors[target_name].tolist()}")

        discovered_faces: set[int] = set()
        discovered_bottom_faces: set[int] = set()
        discovered_non_bottom_faces: set[int] = set()
        max_semantic_pixels = 0
        max_masked_face_pixels = 0

        for step in range(args_cli.num_steps):
            # Rotate in place so the camera sweeps across the fixed target.
            actions = torch.zeros(
                (args_cli.num_envs, env.action_space.shape[-1]),
                device=unwrapped.device,
            )
            actions[:, 1] = -1.0
            with torch.inference_mode():
                env.step(actions)
                semantic_mask = unwrapped._get_semantic_mask(unwrapped._ptz_camera)
                face_ids = unwrapped._raycaster_camera.data.output.get("face_ids")
                goal = unwrapped.inspection_goals[target_name]
                root_pos = goal.data.root_pos_w[0]
                root_quat = goal.data.root_quat_w[0]
                local_up = torch.tensor([[0.0, 0.0, 1.0]], device=unwrapped.device)
                world_up = quat_apply(root_quat.unsqueeze(0), local_up)[0]

                if semantic_mask is not None and face_ids is not None:
                    semantic_mask = semantic_mask.squeeze(-1)
                    face_ids = face_ids.squeeze(-1)
                    masked_hits = semantic_mask & (face_ids >= 0)

                    current_faces = set(
                        int(face) for face in torch.unique(face_ids[0][masked_hits[0]]).cpu().tolist()
                    )
                    discovered_bottom_faces.update(current_faces & bottom_face_ids)
                    discovered_non_bottom_faces.update(current_faces & non_bottom_face_ids)

                    max_semantic_pixels = max(max_semantic_pixels, int(semantic_mask.sum().item()))
                    max_masked_face_pixels = max(max_masked_face_pixels, int(masked_hits.sum().item()))
                    discovered_faces.update(
                        int(face) for face in torch.unique(face_ids[masked_hits]).cpu().tolist()
                    )

                print(
                    f"[PRIMITIVE STEP {step + 1:02d}] root_z={root_pos[2].item():.4f} "
                    f"upright_z={world_up[2].item():.4f} "
                    f"bottom_faces={len(discovered_bottom_faces)} "
                    f"non_bottom_faces={len(discovered_non_bottom_faces)}"
                )

        semantic_info = unwrapped._ptz_camera.data.info.get("semantic_segmentation", {})
        print(f"[PRIMITIVE TEST] Semantic labels: {semantic_info.get('idToLabels', {})}")
        print(f"[PRIMITIVE TEST] Maximum target-mask pixels: {max_semantic_pixels}")
        print(f"[PRIMITIVE TEST] Maximum masked face pixels: {max_masked_face_pixels}")
        print(f"[PRIMITIVE TEST] Unique target faces observed: {len(discovered_faces)}")
        print(f"[PRIMITIVE TEST] Bottom faces observed: {len(discovered_bottom_faces)}")
        print(f"[PRIMITIVE TEST] Non-bottom faces observed: {len(discovered_non_bottom_faces)}")
        if discovered_faces:
            print(
                "[PRIMITIVE TEST] Observed face-ID range: "
                f"{min(discovered_faces)}..{max(discovered_faces)}"
            )

        if authored_faces != configured_faces:
            raise RuntimeError(
                f"Authored/configured triangle mismatch: {authored_faces} != {configured_faces}"
            )
        if max_semantic_pixels == 0:
            raise RuntimeError(f"The {target_name} target never appeared in the semantic mask")
        if max_masked_face_pixels == 0 or not discovered_faces:
            raise RuntimeError(f"No {target_name} face IDs survived semantic masking")
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
