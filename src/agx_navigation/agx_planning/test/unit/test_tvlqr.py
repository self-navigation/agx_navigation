"""Unit tests for the neighboring-optimal (TVLQR) corrector.

The load-bearing test is `test_closed_loop_converges_*`: it integrates the EXACT
nonlinear error kinematics (not the linearization the gains were derived from)
and asserts the error actually decays. Everything else checks a property in
isolation; that one checks the thing we care about.
"""

import math

import numpy as np
import pytest

from agx_planning.rl_corrector.config import RLCorrectorConfig
from agx_planning.runtime_corrector import tvlqr


DT = 0.1


def cfg() -> tvlqr.TVLQRConfig:
    return tvlqr.TVLQRConfig()


# ---------------------------------------------------------------- basics


def test_zero_error_gives_zero_correction():
    """The fail-safe contract: on-path means untouched reference."""
    K = tvlqr.steady_state_gain(0.3, 0.0, DT, cfg())
    v, w, diag = tvlqr.correct(K, np.zeros(3), 0.3, 0.0, cfg())
    assert v == pytest.approx(0.3)
    assert w == pytest.approx(0.0)
    assert diag.dv == pytest.approx(0.0)
    assert diag.domega == pytest.approx(0.0)
    assert diag.valid


def test_non_finite_error_fails_safe_to_reference():
    K = tvlqr.steady_state_gain(0.3, 0.0, DT, cfg())
    v, w, diag = tvlqr.correct(K, np.array([0.0, np.nan, 0.0]), 0.3, 0.1, cfg())
    assert (v, w) == (0.3, 0.1)
    assert not diag.valid


def test_cross_track_error_steers_back():
    """Robot displaced to the LEFT of the path (e_cross > 0) must be commanded to
    turn RIGHT (negative omega correction) to return."""
    K = tvlqr.steady_state_gain(0.3, 0.0, DT, cfg())
    _, _, diag = tvlqr.correct(K, np.array([0.0, 0.2, 0.0]), 0.3, 0.0, cfg())
    assert diag.domega < 0.0

    _, _, diag = tvlqr.correct(K, np.array([0.0, -0.2, 0.0]), 0.3, 0.0, cfg())
    assert diag.domega > 0.0


def test_along_track_lag_speeds_up():
    """Behind the reference (e_along < 0) -> positive speed correction."""
    K = tvlqr.steady_state_gain(0.3, 0.0, DT, cfg())
    _, _, diag = tvlqr.correct(K, np.array([-0.2, 0.0, 0.0]), 0.3, 0.0, cfg())
    assert diag.dv > 0.0


def test_corrections_are_saturated_and_flagged():
    c = cfg()
    c.max_dv = 0.01
    c.max_domega = 0.02
    K = tvlqr.steady_state_gain(0.3, 0.0, DT, c)
    _, _, diag = tvlqr.correct(K, np.array([-1.0, 1.0, 0.5]), 0.3, 0.0, c)
    assert abs(diag.dv) <= c.max_dv
    assert abs(diag.domega) <= c.max_domega
    assert diag.saturated_v and diag.saturated_omega
    # Raw values are retained so we can see HOW badly we ran out of authority.
    assert abs(diag.dv_raw) > c.max_dv


def test_steering_fades_when_too_slow_to_steer():
    """A stationary differential-drive robot cannot fix cross-track error; the
    corrector must not pretend otherwise."""
    c = cfg()
    K = tvlqr.steady_state_gain(0.0, 0.0, DT, c)
    _, _, diag = tvlqr.correct(K, np.array([0.0, 0.3, 0.0]), 0.0, 0.0, c)
    assert diag.steering_faded
    assert diag.domega == pytest.approx(0.0)


# ---------------------------------------------------------------- solver


def test_closed_loop_is_stable():
    """Discrete closed-loop A - BK must have all eigenvalues inside the unit
    circle -- the defining property of a stabilizing LQR solution."""
    for v_ref, omega_ref in [(0.3, 0.0), (0.45, 0.8), (0.15, -1.0), (0.3, 0.5)]:
        K = tvlqr.steady_state_gain(v_ref, omega_ref, DT, cfg())
        A, B = tvlqr.error_dynamics(v_ref, omega_ref)
        Ad, Bd = tvlqr.discretize(A, B, DT)
        eig = np.linalg.eigvals(Ad - Bd @ K)
        assert np.all(np.abs(eig) < 1.0), f"unstable at v={v_ref} w={omega_ref}: {eig}"


def test_gain_schedule_length_and_shape():
    n = 25
    gains = tvlqr.gain_schedule([0.3] * n, [0.1] * n, DT, cfg())
    assert len(gains) == n
    assert all(g.shape == (tvlqr.N_U, tvlqr.N_X) for g in gains)


def test_terminal_gains_are_stiffer_than_mid_trajectory():
    """Finite-horizon solution: near the end the terminal weight dominates, so
    the corrector should pull harder there than in the middle."""
    n = 60
    gains = tvlqr.gain_schedule([0.3] * n, [0.0] * n, DT, cfg())
    assert np.linalg.norm(gains[-1]) > np.linalg.norm(gains[n // 2])


def test_schedule_converges_to_steady_state_away_from_the_end():
    """Far from the terminal, the time-varying gain should approach the
    frozen-time solution -- a consistency check between the two solvers."""
    n = 400
    gains = tvlqr.gain_schedule([0.3] * n, [0.2] * n, DT, cfg())
    K_inf = tvlqr.steady_state_gain(0.3, 0.2, DT, cfg())
    assert np.allclose(gains[0], K_inf, atol=1e-4)


# ---------------------------------------------------------------- closed loop


def _simulate(e0, v_ref, omega_ref, steps, c, use_schedule=False):
    """Integrate the EXACT nonlinear error kinematics under the feedback law.

        e_along_dot   =  omega_ref * e_cross + v cos(e_heading) - v_ref
        e_cross_dot   = -omega_ref * e_along + v sin(e_heading)
        e_heading_dot =  omega - omega_ref

    Note this is the true dynamics, NOT the linearization the gains came from --
    so passing means the linear feedback genuinely stabilizes the real system
    over the error range we care about.
    """
    if use_schedule:
        gains = tvlqr.gain_schedule([v_ref] * steps, [omega_ref] * steps, DT, c)
    else:
        K = tvlqr.steady_state_gain(v_ref, omega_ref, DT, c)

    e = np.array(e0, dtype=float)
    hist = [e.copy()]
    for k in range(steps):
        Kk = gains[k] if use_schedule else K
        v, omega, _ = tvlqr.correct(Kk, e, v_ref, omega_ref, c, index=k)
        e_along, e_cross, e_head = e
        de = np.array([
            omega_ref * e_cross + v * math.cos(e_head) - v_ref,
            -omega_ref * e_along + v * math.sin(e_head),
            omega - omega_ref,
        ])
        e = e + de * DT
        e[2] = tvlqr.wrap_to_pi(e[2])
        hist.append(e.copy())
    return np.array(hist)


@pytest.mark.parametrize("e0", [
    (0.0, 0.20, 0.0),      # pure cross-track offset
    (0.15, 0.0, 0.0),      # pure along-track lead
    (0.0, 0.0, 0.30),      # pure heading error
    (-0.10, 0.15, -0.20),  # everything at once
])
def test_closed_loop_converges_straight(e0):
    hist = _simulate(e0, v_ref=0.3, omega_ref=0.0, steps=200, c=cfg())
    start = np.linalg.norm(hist[0])
    end = np.linalg.norm(hist[-1])
    assert end < 0.1 * start, f"error did not decay: {start:.3f} -> {end:.3f}"


@pytest.mark.parametrize("omega_ref", [0.4, -0.4, 0.8])
def test_closed_loop_converges_on_arcs(omega_ref):
    hist = _simulate((0.05, 0.15, 0.1), v_ref=0.3, omega_ref=omega_ref,
                     steps=200, c=cfg())
    assert np.linalg.norm(hist[-1]) < 0.1 * np.linalg.norm(hist[0])


def test_closed_loop_converges_with_time_varying_schedule():
    hist = _simulate((0.0, 0.2, 0.0), v_ref=0.3, omega_ref=0.2, steps=200,
                     c=cfg(), use_schedule=True)
    assert np.linalg.norm(hist[-1]) < 0.1 * np.linalg.norm(hist[0])


def test_closed_loop_monotone_ish_decay():
    """No sustained oscillation: the error norm late in the run must be well
    below its early peak, and must not be growing at the end."""
    hist = _simulate((0.0, 0.2, 0.0), v_ref=0.3, omega_ref=0.0, steps=200, c=cfg())
    norms = np.linalg.norm(hist, axis=1)
    assert norms[-1] < norms[-20]  # still shrinking at the end
    assert norms[-1] < 0.05 * norms.max()


# ---------------------------------------------------------------- kinematics


def test_twist_wheel_roundtrip():
    """The reduction the real robot depends on must be lossless."""
    kin = RLCorrectorConfig()
    for v, omega in [(0.3, 0.0), (0.0, 0.5), (-0.2, -0.7), (0.45, 1.0)]:
        wl, wr = tvlqr.twist_to_wheels(v, omega, kin)
        v2, omega2 = tvlqr.wheels_to_twist(wl, wr, kin)
        assert v2 == pytest.approx(v)
        assert omega2 == pytest.approx(omega)


def test_gain_cache_reuses_buckets_and_matches_direct_solve():
    c = cfg()
    cache = tvlqr.GainCache(c, DT)
    # Two twists inside one bucket must share a gain and cost one solve.
    K1 = cache.get(0.300, 0.10)
    K2 = cache.get(0.3005, 0.101)
    assert len(cache) == 1
    assert np.allclose(K1, K2)
    # A distant twist gets its own bucket.
    cache.get(0.45, -0.8)
    assert len(cache) == 2
    # The cached gain is the direct solve at the BUCKET CENTRE.
    k = cache.key(0.300, 0.10)
    expected = tvlqr.steady_state_gain(k[0] * c.cache_dv, k[1] * c.cache_domega, DT, c)
    assert np.allclose(K1, expected)


def test_gain_cache_is_stable_under_reversed_visit_order():
    """Bucket centres, not first-caller values, define the gain -- so the cache
    cannot depend on the order twists arrive in."""
    c = cfg()
    a, b = tvlqr.GainCache(c, DT), tvlqr.GainCache(c, DT)
    assert np.allclose(a.get(0.301, 0.099), b.get(0.299, 0.101))


def test_disabled_by_default():
    """Loading the config must not turn the corrector on."""
    assert tvlqr.TVLQRConfig().enabled is False


def test_diagnostics_array_matches_declared_fields():
    """as_array() order is API for logged runs; keep it in step with FIELDS."""
    d = tvlqr.CorrectionDiagnostics()
    assert len(d.as_array()) == len(tvlqr.CorrectionDiagnostics.FIELDS)
