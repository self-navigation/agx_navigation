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
