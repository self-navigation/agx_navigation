"""Terrain-patch helpers for the GazeboBridge (Phase 3 domain randomization).

Thin adapter over the canonical builder in rudn_ordjo_building.surface_patches,
so the patches the bridge spawns during training are byte-identical to the ones
spawn_surface_patches.launch.py places. Imported lazily by the bridge (only when
an episode actually requests terrain), so flat-ground training never needs the
building package on the path.

A "patch" is the same dict schema the launch file accepts:
    {x, y, [z], width, length, profile, [yaw], [name]}
"""

from typing import Callable, List

import numpy as np

from rudn_ordjo_building.surface_patches import PROFILES, build_patch_sdf


def patch_sdf(patch: dict, name: str, idx: int = 0) -> str:
    """SDF string for one patch dict. Raises ValueError on unknown profile."""
    profile_name = patch.get("profile")
    profile = PROFILES.get(profile_name)
    if profile is None:
        raise ValueError(
            f"unknown terrain profile {profile_name!r}; "
            f"available: {list(PROFILES)}"
        )
    p = dict(patch)
    p["name"] = name
    return build_patch_sdf(p, profile, idx)


def along_path_terrain_sampler(
    nominal_poses,
    profiles: List[str] = None,
    n_range=(1, 3),
    size_range=(1.0, 3.0),
) -> Callable:
    """Return sampler(rng) -> list[patch] dropping 1..N slip patches on the path.

    Patches are centred on randomly chosen nominal vertices (so the robot is
    guaranteed to drive over them), with randomized profile/size/yaw. Bound to a
    specific nominal's poses; build a fresh sampler per nominal, or wrap to pull
    poses from the current episode.
    """
    profiles = profiles or ["slippery", "icy", "directional_x", "directional_y"]
    xy = np.asarray(nominal_poses)[:, :2]

    def sample(rng: np.random.Generator) -> List[dict]:
        n = int(rng.integers(n_range[0], n_range[1] + 1))
        patches = []
        for i in range(n):
            vtx = int(rng.integers(0, len(xy)))
            patches.append({
                "x": float(xy[vtx, 0]),
                "y": float(xy[vtx, 1]),
                "z": 0.001,
                "width": float(rng.uniform(*size_range)),
                "length": float(rng.uniform(*size_range)),
                "yaw": float(rng.uniform(-np.pi, np.pi)),
                "profile": str(rng.choice(profiles)),
                "name": f"rl_patch_{i}",
            })
        return patches

    return sample
