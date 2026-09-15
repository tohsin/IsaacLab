# Geometric Diversity Plan for Inspection Training

Last updated: 2026-09-11

## Purpose

The current policy generalizes well to the UR10 mount and corner bracket, but
collides more often with the caster and sortbot housing. Most of those failures
are contacts with the inspection target rather than the free-standing
obstacles. This suggests a gap between the simple procedural training shapes
and irregular CAD geometry.

This document defines additional **generic procedural shapes** that can close
that gap without training directly on the evaluation objects. Caster and
sortbot should now be treated as validation objects because their results have
already influenced design decisions. Separate, untouched CAD objects should be
reserved for the final test set.

## Current training geometry

The current procedural dataset contains:

- Cuboid
- Upright T-block
- Flat T-block
- Sideways shell
- Sphere
- Upright cylinder
- Flat cylinder
- Upright cone
- Flat cone

These cover scale, yaw, orientation, curved surfaces, and a limited amount of
concavity. They do not provide much experience with thin appendages, offset
components, narrow gaps, or objects whose footprint changes substantially with
height.

## Shape vocabulary

The sketches below are conceptual. `#` represents solid target geometry and
`.` represents empty space.

### Low protrusion

A low protrusion is a thin part that extends horizontally from the main object
close to the floor. A forklift fork is the extreme example, but the training
shape should be generic rather than forklift-specific.

Side view:

```text
        main body
        #######
        #######
        #######========  thin low arm
________________________________________ floor
```

Why it matters:

- It may occupy only one 0.2 m voxel or be removed with floor filtering.
- The main body can look safely distant while the arm is already near the
  robot.
- A wheel can climb onto it without the chassis immediately registering a
  frontal impact.

Suggested randomization:

- Main body width/depth: 0.5-1.0 m
- Main body height: 0.8-1.8 m
- Arm length: 0.4-1.2 m
- Arm width: 0.08-0.35 m
- Arm thickness: 0.08-0.30 m
- Arm height above the floor: 0.03-0.25 m
- One or two arms, with full-yaw randomization

Sampling thicknesses below, near, and above the 0.2 m map resolution is
deliberate: the policy must learn how uncertain occupancy relates to collision
risk.

### Overhang

An overhang is an upper section that extends beyond the supporting lower
section. A table, mushroom, shelf, or wide machine head on a narrow pedestal
has this geometry.

Side view:

```text
           #############  overhanging plate
           #############
               #####      narrow support
               #####
               #####
________________________________________ floor
```

Why it matters:

- The footprint near the floor looks clear, but the chassis, camera mast, or
  upper robot can hit the wider section.
- A purely 2D ground-plane clearance estimate is insufficient.
- It teaches the 3D occupancy encoder to distinguish clearance at different
  heights.

Suggested randomization:

- Support width/depth: 0.35-0.8 m
- Support height: 0.4-1.1 m
- Upper plate width/depth: 0.9-1.8 m
- Upper plate thickness: 0.10-0.35 m
- Offset the upper plate by up to 0.35 m instead of always centering it

### Concave housing

A concave object has an inward-facing pocket or opening. A U-shaped machine
housing is a simple example. It is different from a solid cuboid because the
robot can approach or partially enter the opening and then have hazards on
multiple sides.

Top view:

```text
    opening
       v
    #.....#
    #.....#
    #.....#
    #######
```

Why it matters:

- A center-distance estimate is misleading: the target center may be far away
  even while one wall is very close.
- The policy must reverse or turn out rather than continue toward the target
  centroid.
- Inner faces encourage inspection trajectories around openings and corners.

Suggested randomization:

- Outer width/depth: 0.8-1.8 m
- Height: 0.6-1.8 m
- Wall thickness: 0.08-0.30 m
- Pocket depth: 30-85% of the total depth
- Opening width: 0.3-1.2 m
- Randomize which side contains the opening, then apply full yaw

The existing shell is related to this category, but one sideways shell with a
fixed topology is not enough to cover shallow pockets, deep channels, and
asymmetric U/C profiles.

### Offset component

An offset component is attached away from the object's center. A generic
example is a wheel or cylinder mounted on the end of an arm.

Top view:

```text
       main body
       #######
       #######-----O  offset wheel/cylinder
       #######
```

Why it matters:

- The geometric center and nearest collision surface are not closely related.
- The protruding component can move into the robot path while the bulk of the
  target remains visually distant.
- It approximates the geometric difficulty of a caster without copying the
  evaluation mesh.

Suggested randomization:

- Main body: cuboid, cylinder, or short pedestal
- Arm offset: 0.25-0.9 m
- Arm angle: random around the vertical axis
- Attached disc/cylinder radius: 0.12-0.4 m
- Attached component width: 0.08-0.3 m
- Attached component height: 0.05-0.6 m
- Randomly omit the wheel or replace it with a cuboid end piece

### Thin legs and narrow gaps

This is a body supported by several thin legs, or two parallel rails separated
by a gap.

Front view:

```text
       ###########
       ###########
        ##     ##
        ##     ##
________________________________________ floor
```

Why it matters:

- The gap may look traversable even when it is narrower than the robot.
- Thin legs may be poorly represented in a coarse occupancy map.
- The visible outline changes sharply with viewpoint.

Suggested randomization:

- Two to four legs
- Leg width: 0.08-0.35 m
- Gap width: 0.25-1.0 m
- Body height: 0.5-1.6 m
- Symmetric and asymmetric leg layouts

### Stepped or asymmetric block

A stepped object is a union of boxes at different heights and offsets. It is a
simple way to produce irregular silhouettes without curved-mesh complexity.

Side view:

```text
              ####
         #########
    ##############
________________________________________ floor
```

Why it matters:

- Different approach directions have different safe standoff distances.
- It provides corners, partial occlusions, and non-central nearest surfaces.
- It is easy to generate and should be a useful control before implementing
  more complicated composite meshes.

## Recommended first implementation batch

Do not implement every idea at once. Start with four generic families:

1. **Low-arm target**: a pedestal with one or two thin near-floor arms.
2. **U/C housing**: three walls surrounding a randomized pocket.
3. **Offset wheel target**: a body, randomized arm, and attached disc/cylinder.
4. **Overhang target**: a narrow support with a wider, sometimes offset top.

These four address the known failure modes more directly than adding more
spheres, cones, or transformer layers. Add the stepped block or thin-legged
body only if the first batch remains computationally manageable.

## Generator requirements

Each generated target should satisfy the following:

- Represent one static/kinematic inspection target with visual and collision
  geometry in agreement.
- Rest on the floor without starting in penetration or above the floor.
- Use full-yaw randomization unless the experiment intentionally tests a fixed
  pose.
- Vary component sizes, thicknesses, offsets, gaps, and total footprint.
- Produce enough unique geometry variants to avoid repeatedly training on the
  same sixteen meshes, subject to Warp-cache and GPU-memory constraints.
- Tessellate external surfaces reasonably uniformly so coverage reward is not
  dominated by one component merely because it has smaller triangles.
- Exclude permanently hidden, intersecting, and floor-contact faces from the
  reachable-face denominator.
- Validate face IDs, normals, semantic masks, and collision contacts before a
  long training run.

Avoid naively concatenating overlapping boxes while counting every triangle.
That leaves internal intersection faces that no camera can observe and makes
100% coverage impossible. Prefer a single external surface mesh, a Boolean
union, or an explicit reachable-face mask.

## Domain-randomization dimensions

Randomize more than overall scale. Useful independent axes are:

- Overall width, depth, and height
- Component thickness
- Component length and offset from the root
- Gap/opening width and depth
- Height above the floor
- Full target yaw
- Symmetric versus asymmetric layout
- Surface color/material

Keep a small fraction of easy, central, thick-component variants. If every new
target begins with extreme thin geometry, the curriculum may become difficult
before the policy learns the structural concept.

## Evaluation split

Suggested roles for the CAD objects:

- **Validation/tuning:** caster, sortbot housing, forklift, and possibly the
  corner bracket, because their results have already influenced the design.
- **Untouched final test:** select several objects not used for decisions, such
  as pallet, red bowl, tuna can, potted-meat can, wood block, blue cup, or
  Rubik's cube.

Do not add caster or sortbot directly to training and continue calling their
results unseen OOD generalization. Generic composite shapes may be motivated by
their failure categories, but final claims should still use untouched objects.

## Metrics for deciding whether geometry helped

Use identical seeds and evaluation settings for the existing and new policy.
Track:

- Raw face coverage
- Success rate at the declared coverage threshold
- Target-collision rate and obstacle-collision rate separately
- Median and p90/p95 episode length
- Coverage at the moment of collision
- Collision timing: early, middle, or after step 1000
- Inspection quality

The change is successful if it lowers validation-object target collisions
without materially reducing coverage or increasing timeouts. Do not select a
model solely because it happens to produce zero crashes in a small sample.

## Camera-cadence experiment before geometry changes

Run the camera-update experiment as an isolated change before adding new
targets. The current control interval is approximately:

```text
12 / 129 = 0.093 seconds per policy step
```

The current camera period is 0.24 seconds, so one image may be reused across
roughly two or three policy decisions. A value near 0.093 seconds aligns the
requested sensor period with the policy rate. Isaac Lab defines `0.0` as an
update every simulation step, which may render much more often than needed and
be substantially more expensive.

Record actual camera-frame changes or timestamps during a short diagnostic.
Do not assume that writing `0.1` guarantees exactly one fresh frame per policy
step; scheduling and render cadence should be verified once before committing
to another long training run.

Keep geometry, rewards, model architecture, map resolution, and PPO settings
unchanged for this camera-cadence run. Then any OOD improvement can reasonably
be attributed to fresher observations.
