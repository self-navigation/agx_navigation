"""Surface-dependent skid-steer slip model for the KinematicBridge (S4, #44).

Pure module: numpy only, no rclpy, no torch.

WHAT IT MODELS. `slip_chi` is the skid-steer yaw loss,
``omega_actual = omega_ideal / chi`` (CLAUDE.md), and `chi` depends on the
ground friction `mu`. The table below is copied as a literal from
``sweep_data/ground_mu_chi_mu2_045.csv`` (gyro-referenced `slip_ident` sweep of
one big ground patch, wheel ``mu2 = 0.45``, plant ``2026-08-07-wheel-mu2-045``;
method in docs/measurement-rig.md). ``yaw_gain = 1 / chi``.

TWO BRANCHES.

* ``mu >= 0.5``: the robot steers; chi is 1.36-1.57, interpolated linearly in
  ``mu`` between the measured rows (6 usable arcs each, spread <= 0.34).
* ``mu <= 0.45`` (the KNEE, wheel ``mu2 = 0.45``): measured yaw gain 4-7 %, i.e.
  the robot essentially CANNOT STEER -- the second branch of the SVCM dichotomy
  theorem, not a corrector deficiency. Only 2 usable arcs per row and spreads of
  3-6 in chi, so these rows are an order-of-magnitude statement, not a
  calibration. Below ``mu = 0.3`` (unmeasured) the 0.3 row is held.
* ``0.45 < mu < 0.5`` is unmeasured; the yaw gain is interpolated linearly
  between the two branches (interpolating chi instead would put almost the
  whole cliff at 0.45-0.46, which is no better supported).

ASSUMPTION, NOT MEASURED: below the knee the LONGITUDINAL speed is scaled by
``LONG_FACTOR_BELOW_KNEE = 0.9``. No sweep measured forward slip; 0.9 is a
placeholder chosen in the supervisor plan (Q7). Above the knee the factor is 1.

Surface lookup: `patches_to_surface_fn` turns the patch dicts produced by
``rl_corrector.terrain.along_path_terrain_sampler`` (schema
``{x, y, [z], width, length, profile, [yaw], [name]}``; `width` along the
patch's local x, `length` along local y, rotated by `yaw`, as in
``surface_patches.build_patch_sdf``) into ``(x, y) -> mu``. Choices that are
ours, not Gazebo's: the robot is a POINT (its centre), so a patch under one
wheel only does not count; a directional profile uses ``min(mu, mu2)``; where
patches overlap the lowest mu wins; off-patch ground is ``GROUND_MU = 1.0``
(the world file's ground friction).
"""

from typing import Callable, Iterable, Optional

import numpy as np

# ground_mu -> (mean_chi, mean_yaw_gain), literal copy of
# sweep_data/ground_mu_chi_mu2_045.csv. Sorted ascending in mu.
CHI_TABLE = (
    # mu,   chi,     yaw_gain, usable_arcs, spread
    (0.30, 25.3620, 0.0400, 2, 6.0844),
    (0.40, 18.7288, 0.0547, 2, 5.7334),
    (0.45, 15.5832, 0.0650, 2, 3.4485),
    (0.50, 1.5713, 0.6413, 6, 0.3412),
    (0.60, 1.4037, 0.7126, 6, 0.0583),
    (0.80, 1.3651, 0.7326, 6, 0.0232),
    (1.00, 1.3575, 0.7366, 6, 0.0072),
)

KNEE_MU = 0.45          # wheel mu2; at or below it the robot cannot steer
STEER_MU_MIN = 0.50     # lowest measured mu of the steering branch
LONG_FACTOR_BELOW_KNEE = 0.9   # ASSUMPTION (not measured), see docstring
GROUND_MU = 1.0         # rl_corrector.world ground <mu>

_MU = np.array([r[0] for r in CHI_TABLE])
_GAIN = np.array([r[2] for r in CHI_TABLE])


def yaw_gain_of_mu(mu: float) -> float:
    """omega_actual / omega_ideal on ground of friction `mu` (== 1/chi).

    Linear in mu within each branch and across the unmeasured 0.45-0.5 gap;
    clamped to the end rows outside [0.3, 1.0]."""
    return float(np.interp(float(mu), _MU, _GAIN))


def chi_of_mu(mu: float) -> float:
    """Skid-steer yaw-loss factor chi(mu) = 1 / yaw_gain_of_mu(mu)."""
    return 1.0 / yaw_gain_of_mu(mu)


def long_factor_of_mu(mu: float) -> float:
    """Longitudinal speed factor. 1 above the knee; 0.9 at/below (ASSUMED)."""
    return LONG_FACTOR_BELOW_KNEE if float(mu) <= KNEE_MU else 1.0


def is_below_knee(mu: float) -> bool:
    return float(mu) <= KNEE_MU


def _profile_mu(profile_name: str) -> float:
    from rudn_ordjo_building.surface_patches import PROFILES
    prof = PROFILES.get(profile_name)
    if prof is None:
        raise ValueError(f"unknown terrain profile {profile_name!r}; "
                         f"available: {list(PROFILES)}")
    return float(min(prof.mu, prof.mu2))


def patches_to_surface_fn(patches: Optional[Iterable[dict]],
                          ground_mu: float = GROUND_MU,
                          profile_mu: Optional[Callable[[str], float]] = None,
                          ) -> Callable[[float, float], float]:
    """Return ``surface_fn(x, y) -> mu`` for a list of patch dicts.

    `profile_mu` maps a profile name to mu; by default it reads
    ``rudn_ordjo_building.surface_patches.PROFILES`` (imported lazily, so the
    module stays importable without the building package when unused).
    A patch may also carry an explicit ``"mu"`` key, which wins.
    """
    pm = profile_mu or _profile_mu
    rects = []
    for p in patches or ():
        mu = float(p["mu"]) if "mu" in p else pm(p["profile"])
        yaw = float(p.get("yaw", 0.0))
        rects.append((float(p["x"]), float(p["y"]), np.cos(yaw), np.sin(yaw),
                      0.5 * float(p["width"]), 0.5 * float(p["length"]), mu))

    def surface_fn(x: float, y: float) -> float:
        mu = float(ground_mu)
        for cx, cy, c, s, hw, hl, pmu in rects:
            dx, dy = x - cx, y - cy
            lx = c * dx + s * dy        # into the patch frame (rotate by -yaw)
            ly = -s * dx + c * dy
            if abs(lx) <= hw and abs(ly) <= hl and pmu < mu:
                mu = pmu
        return mu

    return surface_fn
