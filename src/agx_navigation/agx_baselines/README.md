# agx_baselines: comparison arms only

**Nothing here is part of our stack.** This package holds the configuration
and wrapper nodes for the baselines that `tools/compare_run.py` runs against
our corrector (#10, #33). If you are looking for how *our* robot navigates,
this is the wrong package; see `agx_bringup` and `agx_planning`.

| path | what |
| --- | --- |
| `config/nav2_controller_<name>.yaml` | controller_server overlay selecting a Nav2 local controller (`nav2_controller:=<name>`; `mppi` is the base `nav2_params.yaml` and has no overlay). `nav2.launch.py` discovers controllers from these files, so adding one needs no launch edit. |
| `config/nav2_profile_<name>.yaml` | comparison profile loaded after `nav2_params.yaml` for every Nav2 node (`nav2_profile:=<name>`) |

Profiles: `compare_static` (stage-1 baseline), `compare_skid` (compare_static
+ MPPI wz_std 0.8 / PathAngleCritic 5 so MPPI turns a skid-steer, job 210),
`compare_hybrid` (Smac Hybrid-A* global planner, job 200).

## GMPC arm (`pmp-gmpc`, #33 rank 1)

Geometric SE(2) MPC, Tang et al., *GMPC: Geometric Model Predictive Control for
Wheeled Mobile Robot Trajectory Tracking*, RA-L 9(5) 2024, arXiv:2403.07317.
Upstream code: <https://github.com/Garyandtang/GMPC-Tracking-Control>, vendored
as the git submodule `third_party/GMPC-Tracking-Control` (pinned). **Upstream
has no LICENSE file** -- it is "all rights reserved" by default, with a
citation request in the README. Fine for an academic comparison run; not
redistributable by us. Flagged, not resolved.

| path | what |
| --- | --- |
| `agx_baselines/gmpc/tracker.py` | ROS-free adapter: subclasses upstream `GeometricMPC`, overriding only `set_ref_traj` so the reference is our PMP plan (resampled onto the controller `dt`, reference `(v, w)` = SE(2) log of consecutive poses / `dt`). `solve` is upstream's, untouched. |
| `agx_baselines/gmpc/node.py` | `gmpc_controller`: a `nav2_msgs/FollowPath` server on `follow_path`, pose from TF `map->base_link`, zero twist on every terminal path, FollowPath error codes. |
| `launch/gmpc.launch.py` | includes `agx_bringup` `main.launch.py` with `nav_mode:=none` and adds the node. Selected through the generic Makefile hook `LAUNCH_PKG=agx_baselines LAUNCH_FILE=gmpc.launch.py` (what `tools/fixture_up.sh --nav-mode gmpc` does). |

Dependencies live OUTSIDE the system ROS python, in a venv the launch file
prepends to the node's `PYTHONPATH` (`gmpc_venv`, default `~/.venvs/gmpc`):

```bash
python3 -m venv --system-site-packages ~/.venvs/gmpc
~/.venvs/gmpc/bin/pip install casadi            # qpOASES plugin ships in the wheel
~/.venvs/gmpc/bin/pip install git+https://github.com/artivis/manif.git   # manifpy, not on PyPI; needs cmake + eigen3
```

Controller settings are upstream's paper defaults (`N=10`,
`Q=diag(20000, 20000, 2000)`, `R=0.3`), bounds `|v|<=0.5`, `|w|<=1.5` (the
velocity smoother's limits for the Nav2 arms), 20 Hz (paper: 50 Hz sim, 5 Hz on
their Scout Mini). The model is upstream's unicycle `(v, w)`; our `chi` has no
counterpart in it, exactly as for the Nav2 arms.
