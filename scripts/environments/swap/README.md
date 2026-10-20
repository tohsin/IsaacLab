# SWAP baseline for robot inspection

This directory contains a native IsaacLab implementation of the planning
structure from **SWAP: Semantics-aware Exploration and Inspection Path
Planning** (Dharmadhikari and Alexis, ICRA 2023). Run it through
`scripts/environments/inspection_swap.py`.

The baseline has three modes:

1. **Volumetric exploration.** A collision-free graph is sampled from the
   global occupancy state maintained by `SpatialStateManager`. Graph vertices
   receive unknown-volume gain, edges are checked through known free space,
   and the best gain-versus-cost branch is executed in receding horizon. A
   frontier path is used only if the sampled graph has no informative branch.
2. **Semantic hole coverage.** Once semantic depth produces a stable target
   cloud, the planner chooses reachable standoff viewpoints in unobserved
   azimuth sectors around that cloud.
3. **Semantic inspection.** Candidate viewpoints are scored against the
   accumulated points, observation-facing normals, distance band, and prior
   visits.

Every episode begins in volumetric exploration mode. If a sufficiently large
semantic component is already visible in the reset observation, the planner
immediately transitions to semantic hole coverage; it is not given the target
location in advance.

This is a **point-cloud UGV adaptation**, not a wrapper around the original
ROS/Voxblox GBPlanner and not a bit-for-bit reproduction of the aerial
system. The original method reconstructs a triangle mesh and finds boundary
edges; this version detects missing view sectors in the observed semantic
point cloud because that is the representation requested for this baseline.

## Evaluation boundary

Planner inputs are restricted to navigation-depth occupancy, PTZ depth,
semantic masks, and odometry. The planner module has no API for target poses,
ground-truth meshes, ray-caster face IDs, or face coverage. The runner reads
the environment's accumulated ray-caster coverage only to log the curve and
terminate a trial at the shared coverage goal. It also writes the observed
semantic cloud for the repository's offline point-cloud evaluation tools.

## Example

```bash
./isaaclab.sh -p scripts/environments/inspection_swap.py \
  --headless \
  --steps 1200 \
  --coverage_goal 0.95 \
  --inspection_dataset evaluation \
  --inspection_target extinguisher
```

Outputs are placed in `outputs/swap/<timestamp>/` unless `--output_dir` is
specified:

- `coverage.csv`: evaluator-only face coverage over simulation steps;
- `env_###_semantic_cloud.ply`: the planner's reconstructed target cloud;
- `results.json`: outcome and planning diagnostics;
- `run_config.json`: reproducibility configuration and data-boundary record.

## Live planning visualization

For a single visual debugging trial, omit `--headless` and add
`--visualize_planner`:

```bash
./isaaclab.sh -p scripts/environments/inspection_swap.py \
  --steps 1200 \
  --inspection_target ur10_mount \
  --visualize_planner
```

The viewport overlay uses cyan for an exploration route, orange for semantic
hole coverage, green for semantic inspection, magenta for the reconstructed
target cloud, and yellow for its estimated centroid. The terminal also prints
the current mode, remaining path length, semantic voxel count, observed
azimuth sectors, and evaluator-only coverage every ten steps. Change the rate
with `--visualization_interval`.

By default, SWAP receives the same full normalized chassis-action range
(`[-1, 1]`) as the learned policy. Both therefore pass through the same
environment-side velocity conversion, wheel saturation, and final
`action_scale`. This is the appropriate default for comparison with policies
evaluated using this repository's existing controller.

To reproduce the lower limits used in the paper comparison instead, add
`--max_linear_speed 1.0 --max_yaw_rate 1.0`. Those optional physical caps are
converted back to normalized actions while accounting for `action_scale`.
