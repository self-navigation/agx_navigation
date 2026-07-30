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


def ground_friction_sampler(
    nominal_poses,
    inner=None,
    profiles: List[str] = None,
    margin: float = 4.0,
) -> Callable:
    """Wrap a patch sampler with ONE large patch under the whole trajectory, so
    the *global* ground friction varies per episode.

    WHY THIS AND NOT A RANDOMIZED slip_chi
    --------------------------------------
    `slip_chi` is a constant of the ASSUMED model (`track_effective` -> `c_w`),
    not of the physics: under GazeboBridge, changing it alters what the nominal
    and the observation believe about yaw loss but nothing about what the robot
    actually does. Worse, with recorded Tier-B nominals the feed-forward wheel
    commands were baked at a fixed chi, so perturbing chi at training time moves
    no command at all. Randomizing the ground the robot drives on is the lever
    that actually changes the plant, and therefore the only one that buys
    sim-to-real robustness here.

    The patch spans the trajectory's bounding box plus `margin`, so the robot
    cannot drive off it -- including the excursions we now deliberately train
    recovery from (see cfg.corridor_terminates). `inner` is the per-episode local
    patch sampler whose patches are laid ON TOP, keeping local slip variation.
    """
    profiles = profiles or ["slippery", "icy", "directional_x", "directional_y"]
    xy = np.asarray(nominal_poses)[:, :2]
    lo = xy.min(axis=0) - margin
    hi = xy.max(axis=0) + margin

    def sample(rng: np.random.Generator) -> List[dict]:
        base = {
            "x": float(0.5 * (lo[0] + hi[0])),
            "y": float(0.5 * (lo[1] + hi[1])),
            # Below the local patches, which sit at z=0.001, so where they
            # overlap the local (more extreme) profile is what the wheels touch.
            "z": 0.0005,
            "width": float(hi[0] - lo[0]),
            "length": float(hi[1] - lo[1]),
            "yaw": 0.0,
            "profile": str(rng.choice(profiles)),
            "name": "rl_ground",
        }
        patches = [base]
        if inner is not None:
            patches.extend(inner(rng))
        return patches

    return sample
