import numpy as np
import pytest

from agx_planning.rl_corrector import slip_model as sm
from agx_planning.rl_corrector.config import RLCorrectorConfig
from agx_planning.rl_corrector.kinematic_bridge import KinematicBridge


def test_chi_table_endpoints():
    assert sm.chi_of_mu(1.0) == pytest.approx(1 / 0.7366)
    assert sm.chi_of_mu(0.5) == pytest.approx(1 / 0.6413)
    assert sm.yaw_gain_of_mu(0.3) == pytest.approx(0.04)
    # clamped outside the measured range
    assert sm.yaw_gain_of_mu(2.5) == pytest.approx(0.7366)
    assert sm.yaw_gain_of_mu(0.05) == pytest.approx(0.04)
    # csv means chi and gain separately (mean of ratios), so 1/chi ~ gain only
    # loosely; within a few % on the steering branch.
    for mu, chi, gain, *_ in sm.CHI_TABLE:
        if mu >= sm.STEER_MU_MIN:
            assert chi * gain == pytest.approx(1.0, rel=0.05)


def test_monotone_in_mu():
    mus = np.linspace(0.0, 3.0, 301)
    gains = [sm.yaw_gain_of_mu(m) for m in mus]
    assert np.all(np.diff(gains) >= 0)
    chis = [sm.chi_of_mu(m) for m in mus]
    assert np.all(np.diff(chis) <= 0)


def test_knee_branch():
    for mu in (0.05, 0.3, 0.4, 0.45):
        assert sm.is_below_knee(mu)
        assert sm.yaw_gain_of_mu(mu) < 0.07
        assert sm.long_factor_of_mu(mu) == sm.LONG_FACTOR_BELOW_KNEE
    for mu in (0.5, 0.6, 1.0, 2.5):
        assert not sm.is_below_knee(mu)
        assert sm.yaw_gain_of_mu(mu) > 0.6
        assert sm.long_factor_of_mu(mu) == 1.0


def _drive(bridge, wheels, n=50, dt=0.05):
    out = [bridge.reset((0.0, 0.0, 0.0))]
    for _ in range(n):
        out.append(bridge.step(wheels, dt))
    return out


def test_identity_when_surface_fn_none():
    cfg = RLCorrectorConfig()
    w = [1.0, 1.0, 2.0, 2.0]
    a = _drive(KinematicBridge(cfg), w)
    b = _drive(KinematicBridge(cfg, surface_fn=None), w)
    for sa, sb in zip(a, b):
        assert sa.pose == sb.pose and sa.imu == sb.imu and sa.omega == sb.omega


def test_slip_scales_yaw_and_gyro_reports_actual():
    cfg = RLCorrectorConfig()
    w = [1.0, 1.0, 2.0, 2.0]
    _, omega_nom = cfg.wheels_to_body(1.0, 2.0)
    br = KinematicBridge(cfg, surface_fn=lambda x, y: 0.3)
    st = _drive(br, w, n=1)[-1]
    expect = omega_nom * cfg.slip_chi * sm.yaw_gain_of_mu(0.3)
    assert st.omega == pytest.approx(expect)
    assert st.imu[0] == pytest.approx(expect)
    assert abs(st.omega) < 0.1 * abs(omega_nom)


def test_patch_lookup():
    patches = [{"x": 2.0, "y": 1.0, "z": 0.001, "width": 2.0, "length": 0.5,
                "yaw": np.pi / 2, "profile": "icy", "name": "rl_patch_0"},
               {"x": 10.0, "y": 0.0, "width": 1.0, "length": 1.0,
                "profile": "wet_tile"}]
    f = sm.patches_to_surface_fn(patches)
    assert f(2.0, 1.0) == pytest.approx(0.05)
    # yaw=90deg: width (2 m) now runs along world y
    assert f(2.0, 1.9) == pytest.approx(0.05)
    assert f(2.9, 1.0) == pytest.approx(sm.GROUND_MU)
    assert f(10.4, -0.4) == pytest.approx(0.30)
    assert f(-5.0, -5.0) == pytest.approx(sm.GROUND_MU)
    # directional profiles take the slippery axis
    g = sm.patches_to_surface_fn([{"x": 0, "y": 0, "width": 1, "length": 1,
                                   "profile": "directional_x"}])
    assert g(0, 0) == pytest.approx(0.15)
    with pytest.raises(ValueError):
        sm.patches_to_surface_fn([{"x": 0, "y": 0, "width": 1, "length": 1,
                                   "profile": "nope"}])


def test_bridge_builds_surface_from_terrain():
    cfg = RLCorrectorConfig()
    br = KinematicBridge(cfg, surface_from_terrain=True)
    br.reset((0.0, 0.0, 0.0), [{"x": 0, "y": 0, "width": 4, "length": 4,
                                "profile": "icy"}])
    br.step([1.0, 1.0, 2.0, 2.0], 0.05)
    assert br.last_mu == pytest.approx(0.05)
