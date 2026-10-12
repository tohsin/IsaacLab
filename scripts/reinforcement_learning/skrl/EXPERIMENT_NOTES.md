# SEEIR RL Experiment Notes

Last updated: 2026-09-10

This file is the durable context for the attention-fusion, curriculum-stability,
and collision-rate investigation. Update it after consequential training or
evaluation runs so that results do not depend on chat history.

## Main research question

The attention-fusion policy reaches the hardest curriculum faster than the pure
MLP-fusion policy, but the first CLS-token version oscillated after reaching the
hardest curriculum instead of improving monotonically. The current experiment
asks whether mean pooling retains the attention model's sample efficiency while
recovering the stable convergence and low collision rate seen in an older model.

The planned comparison keeps the attention width at `d_model=512` to match the
MLP fusion output width. This matches the representation width, although it does
not make the two networks' exact parameter counts identical.

## Important checkpoints

### Recovered August 31 policy

- Run: `SEEIR-Baseline/SEEIR-2026-08-31_20-03-08`
- Checkpoint: `checkpoints/agent_135000.pt`
- Full path relative to the repository:
  `scripts/reinforcement_learning/skrl/logs/skrl/SEEIR-Baseline/SEEIR-2026-08-31_20-03-08/checkpoints/agent_135000.pt`
- Relevant source snapshot: Git commit `94e285c785`
- Attention width: 256
- Fusion readout: mean pooling over the three attended modality tokens
- Encoder: two residual blocks per stage (the older, deeper approximately
  15-layer/wide ResNet setup)
- PPO epochs: 2
- PPO discount/lambda: 0.99 / 0.97
- Policy LR: `5e-5`
- Std LR: `6e-5`
- Entropy coefficient: `2e-5`
- Gradient clipping: 0.7
- Cosine minimum LR: 10% of the initial policy LR
- Training length: 35 million global steps
- gSDE was enabled in the recovered configuration.
- Historical reward values included coverage scale 400, occupancy scale 0.15,
  terminal collision penalty 1.0, and collision-force threshold 10 N.

The legacy architecture is represented by `LegacyAugust31EvaluationConfig` in
`scripts/reinforcement_learning/skrl/run_config.py`. It exists to load the old
checkpoint under the current environment and collision detector.

### Recent CLS-token attention policy

- Run: `Alblation-Baseline/Alblation_ATTN_FUS_2026-09-09_16-54-05`
- Checkpoint: `checkpoints/best_agent.pt`
- Attention width: 512
- Fusion readout: learned CLS token
- Encoder: one residual block per stage
- This is the main recent policy used in the UR10, forklift, and T-block
  comparisons below.

### MLP-fusion policy

- Run: `Alblation-Baseline/Alblation_MLP_FUSION_2026-09-06_11-02-41`
- It learned the hardest curriculum more slowly than attention, but its
  performance curve improved more monotonically after reaching it.

## Completed comparable evaluations

Unless stated otherwise, these were deterministic policy-mean evaluations with
the current base-link plus wheel contact detector and without the safety shield.
Raw JSON and NPZ files are stored beside each checkpoint under `eval_results/`.

| Policy | Target | Episodes | Mean coverage | Crash episodes | Main observation |
|---|---:|---:|---:|---:|---|
| August 31 mean/256 | UR10 mount | 512 | 94.89% | 6 (1.17%) | Genuine low-collision result under the current detector |
| August 31 mean/256 | Forklift | 512 | 81.60% | 210 (41.02%) | 202/210 impacts were the forklift; 198 were forward |
| August 31 mean/256 | Small corner bracket (physics) | 512 | 95.75% | 8 (1.56%) | All 8 impacts were obstacles; zero bracket impacts |
| August 31 mean/256 | Caster | 512 | 85.53% | 33 (6.45%) | 20 target, 12 obstacle, and 1 wall impact |
| Recent CLS/512 | UR10 mount | 1024 | 92.07% | 58 (5.66%) | 20 target and 38 obstacle impacts |
| Recent CLS/512 | Forklift | 1024 | 84.56% | 246 (24.02%) | 214/246 impacts were the forklift |
| Recent CLS/512 | Tessellated T-block | 1024 | 93.70% | 39 (3.81%) | 37/39 impacts were obstacles |
| Recent CLS/512 + shield | Tessellated T-block | 512 | 90.80% | 19 (3.71%) | Shield intervened in 64.45% of episodes and 12.47% of steps |

The corner-bracket result is important: high coverage, very low collision rate,
and no contact with the target indicate that the forklift failure is not a
general inability to inspect unfamiliar industrial geometry.

## Interpretation of the forklift failure

The forklift is an unusually adversarial target because its forks are low,
thin, and protrude far beyond the main body. Both policies obtain high coverage
in non-crashing forklift episodes, but frequently command forward motion into
the target.

The occupancy representation may make this worse:

- Global voxel resolution is 0.2 m.
- The local observation is `21 x 21 x 11`, covering about 4.2 m by 4.2 m.
- The collision proxy and safety shield skip the lowest `z` voxel to avoid
  interpreting the floor as an obstacle. Low forklift forks may occupy exactly
  that ignored band.
- The proxy and shield currently require occupancy log-odds greater than 1.1,
  while a newly observed occupied point adds 0.8. Thin or briefly observed
  geometry may therefore not count as occupied immediately.
- Rendered depth geometry and the USD collision mesh may not have identical
  boundaries.

Do not silently discard the forklift result. If the policy succeeds on a broad
set of other objects, report the forklift as a known low-protrusion/collision-
geometry limitation rather than including it in the main aggregate.

## Collision-related changes and findings

- Collision termination now observes the base link and four wheels, attributes
  contacts to the inspection target, obstacles, and warehouse wall, and detects
  sustained tip-over.
- Current training/evaluation settings use two consecutive contact detections
  and two consecutive excessive-tilt detections. The tip-over limit is 45
  degrees.
- Inspection targets are kinematic/static for the current experiments.
- A directional occupancy-map safety shield was implemented. It scales only
  unsafe linear motion while preserving turning and PTZ commands.
- On the T-block, the shield did not materially lower crashes and reduced mean
  coverage. Several crashes occurred in episodes where it never intervened.
  Therefore it is disabled for policy comparisons and the next pooling run.
- Increasing collision/proxy penalties did not produce a convincing collision
  reduction and degraded performance in one run. Reward magnitude alone is not
  considered the primary fix.
- A 0.6 m proxy radius can make tight inspection unnecessarily difficult; the
  current proxy radius is approximately 0.4 m.

Before globally increasing occupancy-map resolution, diagnose low geometry and
collision-mesh disagreement. Reducing resolution from 0.2 m to 0.1 m while
preserving physical coverage would require approximately `41 x 41 x 21`, make
old checkpoints incompatible, and increase global-map storage dramatically. A
separate small, high-resolution local safety map may be a better future design.

## Current planned training configuration

`TrainingConfig_PreTrain` is prepared for the controlled pooling experiment:

- Attention fusion enabled
- Transformer encoder enabled
- `d_model=512`
- Mean pooling
- One residual block per encoder stage
- PPO epochs: 3
- Mini-batches: 8
- Discount/lambda: 0.995 / 0.95
- Policy LR: `4e-5`
- Attention-fusion LR: `2e-5`
- Std LR: `4e-5`
- Entropy coefficient: `4e-5`
- Gradient clipping: 0.8
- Cosine LR schedule to 1% of the initial policy LR
- gSDE enabled
- Training length: 40 million global steps
- Directional safety shield disabled, preserving a policy-only comparison

Only pooling should differ from the recent CLS setup when interpreting this
experiment. Do not simultaneously restore the deeper CNN, change rewards, or
change map resolution.

## What to watch during the next training run

- Time/sample count at which the hardest curriculum is first reached
- Whether the hardest curriculum remains locked rather than repeatedly falling
  and recovering
- Approximate KL after reaching the hardest curriculum, including spikes and
  sustained trends rather than a single value
- Policy, attention-fusion, and std learning rates
- Per-action standard deviations, especially linear and angular velocity versus
  pan and tilt
- Success rate, crash rate, collision-proxy rate, and episode length together
- Whether improvement after reaching the hardest curriculum is monotonic or
  oscillatory

A falling KL accompanied by falling crash rate and rising success is healthy;
it is not necessary for every individual update or success measurement to be
monotonic. Repeated KL spikes followed by curriculum regression are the more
important warning sign.

## Evaluation queue

The August 31 checkpoint is being screened on multiple out-of-distribution
targets under identical current settings. At the time of this update:

- UR10 mount: complete
- Forklift: complete
- Small corner bracket physics: complete
- Caster: complete
- Recommended next targets: pallet, sortbot housing, and one convex control
  (`rubiks_cube` or `wood_block`)

For cross-object comparisons, prioritize coverage percentage, crash-episode
percentage, crash source, crash direction, and successful-only coverage. Raw
face counts are not comparable because each mesh has a different face count.

## Configuration selectors

During legacy checkpoint evaluation:

- `scripts/reinforcement_learning/skrl/run_config.py`: `CONFIG = configs_[2]`
- Environment run config: `cfg_mode = modes[2]`

For the planned 512-wide mean-pooling training run:

- `scripts/reinforcement_learning/skrl/run_config.py`: `CONFIG = configs_[0]`
- Environment run config: `cfg_mode = modes[1]`

Always verify the target name, selected checkpoint, `d_model`, pooling mode,
encoder depth, evaluation mode, and shield state in startup output before a
long run.
