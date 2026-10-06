"""tools/compare_run.plan_to_path: the Type-A (pmp-mppi / pmp-rpp) path."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), *[".."] * 5, "tools"))
from compare_run import (ARM_TO_CONTROLLER, ARM_TO_NAV_MODE, PMP_PATH_ARMS,  # noqa: E402
                         plan_to_path)


def test_arms_map_to_controllers():
    # Every arm's controller is mppi (the base nav2_params.yaml) or has an
    # agx_baselines overlay; arms are added there, so no fixed set here. Arms
    # with their own nav mode (pmp-gmpc) run no Nav2 controller at all.
    cfg = os.path.join(os.path.dirname(__file__), *[".."] * 3,
                       "agx_baselines", "config")
    for a in PMP_PATH_ARMS:
        if ARM_TO_NAV_MODE.get(a, "nav2") != "nav2":
            continue
        c = ARM_TO_CONTROLLER[a]
        assert c == "mppi" or os.path.isfile(
            os.path.join(cfg, f"nav2_controller_{c}.yaml")), a


def test_spacing_endpoints_and_turn_in_place():
    poses = np.array([[0, 0, 0], [0, 0, 0.5], [1, 0, 0.5], [1, 1, 1.0], [1, 1, 3.0]], float)
    out = plan_to_path(poses, ds=0.05)
    assert np.allclose(out[0, :2], [0, 0]) and np.allclose(out[-1, :2], [1, 1])
    d = np.hypot(*np.diff(out[:, :2], axis=0).T)
    assert d.max() <= 0.05 + 1e-9 and d.min() > 0
    assert math.isclose(out[-1, 2], 3.0)  # final plan heading kept, not tangent
    assert np.all(np.abs(out[:, 2]) <= math.pi)


def test_reverse_keeps_plan_heading():
    poses = np.array([[1, 0, 0.0], [0.5, 0, 0.0], [0, 0, 0.0]])
    out = plan_to_path(poses)
    assert np.allclose(out[:, 2], 0.0)  # tangent would be pi


def test_yaw_wraps_across_pi():
    poses = np.array([[0, 0, 3.1], [1, 0, -3.1]])
    out = plan_to_path(poses, ds=0.5)
    assert np.all(np.abs(np.abs(out[:, 2]) - 3.1) < 0.1)
