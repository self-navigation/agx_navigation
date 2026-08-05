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

from rudn_ordjo_building.surface_patches import (
    DEFAULT_PATCH_WEIGHTS, PROFILES, build_patch_sdf,
)


def _weighted_profiles(profiles, weights):
    """(names, probabilities) for rng.choice, normalized and validated.

    `profiles` may be a list of names (weights looked up in `weights`) or a
    {name: weight} dict. A name with no weight is an error rather than an
    implicit zero: silently never drawing a profile someone asked for is the
    kind of thing that goes unnoticed for weeks.
    """
    if isinstance(profiles, dict):
        items = list(profiles.items())
    else:
        missing = [p for p in profiles if p not in weights]
        if missing:
            raise ValueError(
                f"no sampling weight for profile(s) {missing}; add them to "
                f"DEFAULT_PATCH_WEIGHTS or pass an explicit {{name: weight}} dict"
            )
        items = [(p, weights[p]) for p in profiles]
    names = [n for n, _ in items]
    w = np.asarray([float(x) for _, x in items], dtype=float)
    unknown = [n for n in names if n not in PROFILES]
    if unknown:
        raise ValueError(f"unknown terrain profile(s) {unknown}; "
                         f"available: {list(PROFILES)}")
    if w.min() < 0 or w.sum() <= 0:
        raise ValueError(f"patch weights must be non-negative and sum > 0: {w}")
    return names, w / w.sum()


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
    profiles=None,
    n_range=(1, 3),
    size_range=(1.0, 3.0),
    weights=None,
) -> Callable:
    """Return sampler(rng) -> list[patch] dropping 1..N slip patches on the path.

    Patches are centred on randomly chosen nominal vertices (so the robot is
    guaranteed to drive over them), with randomized profile/size/yaw. Bound to a
    specific nominal's poses; build a fresh sampler per nominal, or wrap to pull
    poses from the current episode.

    Profiles are drawn with the WEIGHTS in DEFAULT_PATCH_WEIGHTS, not uniformly.
    Uniform draws over the old four-profile list put half of every patch set at
    or below tyre-on-ice friction, which tunes the controller for black ice
    rather than for the linoleum it deploys on. Pass `profiles` as a
    {name: weight} dict to override per call.
    """
    weights = weights or DEFAULT_PATCH_WEIGHTS
    profiles = profiles if profiles is not None else list(weights)
    names, probs = _weighted_profiles(profiles, weights)
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
                "profile": str(rng.choice(names, p=probs)),
                "name": f"rl_patch_{i}",
            })
        return patches

    return sample


def ground_friction_sampler(
    nominal_poses,
    inner=None,
    profiles=None,
    margin: float = 4.0,
    weights=None,
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
    # The GROUND is the surface the robot spends most of an episode on, so its
    # distribution is what the learned/tuned controller is really fitted to.
    # Drawing it uniformly over the old list meant a quarter of all episodes ran
    # on black ice END TO END. Weighted like the local patches, so the typical
    # episode is a realistic floor and the hard surfaces are the exception.
    # Directional profiles are excluded here: a whole-floor anisotropy has no
    # physical analogue, unlike a local patch of it.
    weights = weights or {k: v for k, v in DEFAULT_PATCH_WEIGHTS.items()
                          if not k.startswith("directional_")}
    profiles = profiles if profiles is not None else list(weights)
    names, probs = _weighted_profiles(profiles, weights)
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
            "profile": str(rng.choice(names, p=probs)),
            "name": "rl_ground",
        }
        patches = [base]
        if inner is not None:
            patches.extend(inner(rng))
        return patches

    return sample
