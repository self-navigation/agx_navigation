# agx_baselines

Baseline navigation stacks that `tools/compare_run.py` runs against our
corrector. Not used by our own stack.

- `config/nav2_controller_<name>.yaml`: Nav2 local controller overlay, selected with `nav2_controller:=<name>` and discovered by `nav2.launch.py`. `mppi` is the base `nav2_params.yaml`.
- `config/nav2_profile_<name>.yaml`: profile loaded after `nav2_params.yaml` (`nav2_profile:=<name>`): `compare_static` (plain baseline), `compare_skid` (MPPI tuned to turn a skid-steer), `compare_hybrid` (Smac Hybrid-A* global planner).
