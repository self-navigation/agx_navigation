# agx_baselines

Baseline navigation stacks that `tools/compare_run.py` runs against our
corrector. Not used by our own stack.

- `config/nav2_controller_<name>.yaml`: Nav2 local controller overlay, selected with `nav2_controller:=<name>` and discovered by `nav2.launch.py`. `mppi` is the base `nav2_params.yaml`.
- `config/nav2_profile_<name>.yaml`: profile loaded after `nav2_params.yaml` (`nav2_profile:=<name>`): `compare_static` (plain baseline), `compare_skid` (MPPI tuned to turn a skid-steer), `compare_hybrid` (Smac Hybrid-A* global planner).
- `third_party/`: upstream controllers as submodules, built by `make build`: `vector_pursuit_controller` (Apache-2.0).

Controllers: `dwb`, `rpp`, `graceful` (from apt), `vpp` (Vector Pursuit) and
`mppi`, all limited to 0.5 m/s, 1.5 rad/s, forward only.
