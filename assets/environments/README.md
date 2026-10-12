# Inspection debug environments

inspection_null.py can load any local USD scene without changing the training
or evaluation defaults.

Install one of NVIDIA's native OpenUSD sample scenes:

~~~bash
python scripts/environments/install_inspection_environment.py usd_explorer_factory
python scripts/environments/install_inspection_environment.py defect_workshop
~~~

Then drive in it with:

~~~bash
python scripts/environments/inspection_null.py --environment usd_explorer_factory
python scripts/environments/inspection_null.py --environment defect_workshop
~~~

External presets suppress the task's procedural obstacles by default because
the scenes already contain industrial clutter. Add --keep_procedural_obstacles
if you intentionally want both.

The sample CAD scenes are treated as static geometry. The launcher adds exact
triangle-mesh collision to visible meshes that do not already have authored
collision. Disable that diagnostic conversion if it is too expensive:

~~~bash
python scripts/environments/inspection_null.py --environment usd_explorer_factory --no-add_environment_mesh_colliders
~~~

An arbitrary scene can be loaded in the same way:

~~~bash
python scripts/environments/inspection_null.py --environment custom --environment_usd /absolute/path/to/scene.usd
~~~

Use --environment_scale X Y Z and --environment_offset X Y Z when a scene's
authored units or origin need adjustment.

The large downloaded packs are ignored by Git. Preserve their original folder
layout because the USD stages use relative references and texture paths.
