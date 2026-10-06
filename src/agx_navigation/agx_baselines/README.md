# agx_baselines

Baseline navigation stacks that `tools/compare_run.py` runs against our
corrector. Not used by our own stack.

- `config/nav2_controller_<name>.yaml`: Nav2 local controller overlay, selected with `nav2_controller:=<name>` and discovered by `nav2.launch.py`. `mppi` is the base `nav2_params.yaml`.
- `config/nav2_profile_<name>.yaml`: profile loaded after `nav2_params.yaml` (`nav2_profile:=<name>`): `compare_static` (plain baseline), `compare_skid` (MPPI tuned to turn a skid-steer), `compare_hybrid` (Smac Hybrid-A* global planner).

## GMPC

Geometric SE(2) MPC from Tang et al., RA-L 2024 (arXiv:2403.07317), vendored
as the submodule `third_party/GMPC-Tracking-Control`. Upstream has no license,
so it can be used for comparison runs but not redistributed.

- `agx_baselines/gmpc/tracker.py`: ROS-free adapter that feeds our PMP plan to upstream `GeometricMPC`.
- `agx_baselines/gmpc/node.py`: `gmpc_controller`, a `FollowPath` action server.
- `launch/gmpc.launch.py`: `main.launch.py` with `nav_mode:=none` plus the node; run it with `make run LAUNCH_PKG=agx_baselines LAUNCH_FILE=gmpc.launch.py`.

Dependencies go in a venv that the launch file adds to the node's `PYTHONPATH`
(`gmpc_venv`, default `~/.venvs/gmpc`):

```bash
python3 -m venv --system-site-packages ~/.venvs/gmpc
~/.venvs/gmpc/bin/pip install casadi
~/.venvs/gmpc/bin/pip install git+https://github.com/artivis/manif.git   # needs cmake and eigen3
```

Uses the paper's settings (`N=10`, `Q=diag(20000, 20000, 2000)`, `R=0.3`) at
20 Hz, with the Nav2 arms' limits `|v| <= 0.5`, `|w| <= 1.5`.
