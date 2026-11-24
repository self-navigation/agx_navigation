# Shared numerical profiles

`agx_planning.profiles` loads version 1 experiment profiles without ROS. The
snapshot benchmark and model runner use this loader. It returns all
`PlannerConfig` and `VectorFieldConfig` fields, including defaults omitted from
the source YAML. Each config factory returns a fresh production config.

```python
from agx_planning.profiles import (
    field_config, load_profile, planner_config, profile_digest,
)

profile = load_profile("config/replanning/baseline_profile.yaml")
planner = planner_config(profile)
field = field_config(profile)
recipe_id = profile_digest(profile)
```

Save `profile` and `recipe_id` in the run manifest. The digest includes the
expanded settings, occupancy policy, declared field path, and pin tolerance.
Run manifests must also record the actual map geometry and crop bounds, map
source hashes, Git and submodule commits, and numerical binary identity. These
values depend on the run inputs and do not belong to a reusable profile.

`packed_float32` selects the existing packed transport. `raw_float64` selects a
direct numerical path. The core accepts either explicit choice. The snapshot
benchmark accepts only `packed_float32`, so its recorded transport and historical
hashes do not change. The two paths need precision-specific parity checks.
Bottleneck shaping is disabled in the initial profile campaign on both paths.

Loading a profile does not change ROS launch defaults. The existing
`planner.launch.py` and `vec_pmp.launch.py` still contain live overrides. Wiring
those nodes to this profile loader, and checking live parameter parity, remains
an adapter task. Phase A6 is therefore complete for the experiment tools and
model core, but not yet for the deployed launch path.
