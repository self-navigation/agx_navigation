"""Unit tests for the TVLQG surface filter.

Runs the filter against a synthetic plant with known (k, rho, b_g) and checks
that the estimates converge and that their sigmas are consistent with the
actual errors.
"""

import math

import numpy as np
import pytest

from agx_planning.runtime_corrector import tvlqg

DT = 0.1           # filter / corrector rate
SUB = 10           # plant sub-steps per filter step


def slalom(t: float, v: float = 0.5, w_amp: float = 0.8, period: float = 6.0):
    return v, w_amp * math.sin(2.0 * math.pi * t / period)


def straight(t: float):
    return 0.5, 0.0


def run(profile, T: float, k=-0.33, rho=0.05, bg=0.004, seed=0, cfg=None,
        k0=0.0, tau=0.0, tau_filter=None):
    """Simulate truth + filter; return the filter and the true state.
    ``tau`` is the plant's yaw lag, ``tau_filter`` the filter's (default: same)."""
    rng = np.random.default_rng(seed)
    tf = tau if tau_filter is None else tau_filter
    cfg = cfg or tvlqg.TVLQGConfig(k0=k0, yaw_tau=tf)
    truth = np.array([0.0, 0.0, 0.0, 0.0, k, rho, bg])
    f = tvlqg.SurfaceFilter(cfg, truth[:3] + rng.normal(0, [0.02, 0.02, 0.01]))
    dts = DT / SUB
    for i in range(int(round(T / DT))):
        v, w = profile(i * DT)
        # Input noise held over the filter interval: the discretization the
        # filter assumes (variance density^2 / DT).
        vn = v + rng.normal(0, cfg.q_v / math.sqrt(DT))
        wn = w + rng.normal(0, cfg.q_omega / math.sqrt(DT))
        for _ in range(SUB):
            truth = tvlqg.propagate(truth, vn, wn, dts, tau)
        f.predict(v, w, DT)
        z = truth[:3] + rng.normal(0, [cfg.r_pos, cfg.r_pos, cfg.r_theta])
        f.update_pose(z)
        f.update_gyro(truth[tvlqg.IW] + bg + rng.normal(0, cfg.r_gyro))
    return f, truth


# ---------------------------------------------------------------- model


def test_jacobian_predicts_a_small_perturbation():
    x = np.array([0.3, -0.2, 0.7, 0.8, -0.33, 0.05, 0.01])
    v, w, dt, tau = 0.6, 0.9, DT, 0.27
    F, _ = tvlqg.jacobians(x, v, w, dt, tau)
    d = 1e-4 * np.arange(1, tvlqg.N_S + 1)
    lin = F @ d
    act = tvlqg.propagate(x + d, v, w, dt, tau) - tvlqg.propagate(x, v, w, dt, tau)
    assert np.allclose(lin, act, atol=1e-7)


def test_yaw_lag_reduces_to_instant_at_tau_zero():
    x = np.array([0.0, 0.0, 0.0, 0.0, -0.33, 0.05, 0.0])
    a = tvlqg.propagate(x, 0.5, 0.8, DT, 0.0)
    assert a[tvlqg.IW] == pytest.approx(1.05 * 0.8)
    assert a[tvlqg.ITH] == pytest.approx(1.05 * 0.8 * DT)


def test_yaw_lag_rises_with_the_time_constant():
    x = np.zeros(tvlqg.N_S)
    for _ in range(3):                        # 0.3 s ~ one tau
        x = tvlqg.propagate(x, 0.5, 1.0, DT, 0.3)
    assert x[tvlqg.IW] == pytest.approx(1.0 - math.exp(-1.0), abs=1e-9)


def test_slide_is_outward_for_negative_k():
    """k < 0 with a left turn pushes the body to the RIGHT (outward)."""
    x = np.zeros(tvlqg.N_S)
    x[tvlqg.IK] = -0.33
    x1 = tvlqg.propagate(x, 0.5, 0.8, DT)
    x0 = tvlqg.propagate(np.zeros(tvlqg.N_S), 0.5, 0.8, DT)
    assert x1[1] < x0[1]


# ---------------------------------------------------------------- estimation


@pytest.mark.parametrize("k_true", [-0.33, -0.03])
def test_k_converges_on_a_slalom(k_true):
    f, truth = run(slalom, 60.0, k=k_true)
    k, s = f.k
    assert abs(k - k_true) < 0.05
    assert abs(k - k_true) < 3.0 * s
    rho, sr = f.rho
    assert abs(rho - truth[tvlqg.IRHO]) < 0.02


@pytest.mark.parametrize("tau", [0.0, 0.27])
def test_lagged_plant_with_matching_filter(tau):
    """The real chassis' yaw lag (0.27 s), modelled: k and rho still recovered,
    and the gate stays quiet."""
    f, truth = run(slalom, 60.0, k=-0.03, rho=0.07, tau=tau)
    k, s = f.k
    assert abs(k + 0.03) < 3.0 * s and s < 0.05
    assert abs(f.rho[0] - 0.07) < 0.02
    assert f.n_gated < 20                     # of 1200 updates


def test_unmodelled_lag_is_what_broke_the_real_replay():
    """Plant lags 0.27 s, filter assumes instant: rho_hat goes wrong and the
    gate fires a lot -- the 2026-10-09 failure, reproduced."""
    f, _ = run(slalom, 60.0, k=-0.03, rho=0.07, tau=0.27, tau_filter=0.0)
    assert f.n_gated > 100 or abs(f.rho[0] - 0.07) > 0.05


def test_k_unobservable_on_a_straight():
    """No moving turn, no information about k: it stays at the prior and its
    sigma grows by the random walk. This is a property of the vehicle, and it
    bounds how early a surface switch can fire."""
    cfg = tvlqg.TVLQGConfig()
    f, _ = run(straight, 30.0, k=-0.33, cfg=cfg)
    k, s = f.k
    assert abs(k - cfg.k0) < 0.02
    assert s > cfg.sigma_k0


def test_sigma_is_honest_over_seeds():
    """The 2-sigma band holds the true k in ~95% of runs."""
    inside = 0
    n = 40
    for seed in range(n):
        f, truth = run(slalom, 30.0, seed=seed)
        k, s = f.k
        inside += abs(k - truth[tvlqg.IK]) < 2.0 * s
    assert inside / n > 0.85


def test_nis_is_chi2_consistent():
    """Pose-update NIS averages ~3 (its dof) when the model is right."""
    rng_runs = []
    cfg = tvlqg.TVLQGConfig()
    rng = np.random.default_rng(1)
    truth = np.array([0.0, 0.0, 0.0, 0.0, -0.33, 0.05, 0.004])
    f = tvlqg.SurfaceFilter(cfg, truth[:3])
    dts = DT / SUB
    for i in range(600):
        v, w = slalom(i * DT)
        vn = v + rng.normal(0, cfg.q_v / math.sqrt(DT))
        wn = w + rng.normal(0, cfg.q_omega / math.sqrt(DT))
        for _ in range(SUB):
            truth = tvlqg.propagate(truth, vn, wn, dts)
        f.predict(v, w, DT)
        f.update_pose(truth[:3] + rng.normal(0, [cfg.r_pos, cfg.r_pos, cfg.r_theta]))
        if i > 50:
            rng_runs.append(f.nis)
    assert 2.4 < np.mean(rng_runs) < 3.6


def test_surface_departure_flags_the_other_surface():
    f, _ = run(slalom, 60.0, k=-0.33, k0=0.0)
    assert f.surface_departure(-0.33) < 3.0
    assert f.surface_departure(-0.03) > 3.0


# ---------------------------------------------------------------- pre-drive covariance


def test_covariance_along_plan_shrinks_in_turns_grows_on_straights():
    cfg = tvlqg.TVLQGConfig()
    # 10 s straight, 10 s slalom, 10 s straight.
    n = 300
    v = np.full(n, 0.5)
    w = np.zeros(n)
    w[100:200] = [0.8 * math.sin(2 * math.pi * i * DT / 6.0) for i in range(100)]
    x = np.zeros(tvlqg.N_S)
    poses = []
    for i in range(n):
        poses.append(x[:3].copy())
        x = tvlqg.propagate(x, v[i], w[i], DT)
    P = tvlqg.covariance_along_plan(np.array(poses), v, w, DT, cfg)
    sk = np.sqrt(P[:, tvlqg.IK, tvlqg.IK])
    assert sk[99] == pytest.approx(cfg.sigma_k0, rel=0.1)   # nothing learned yet
    assert sk[199] < 0.5 * sk[99]                          # the turns taught it
    assert sk[299] > sk[199]                               # straight: random walk


# ---------------------------------------------------------------- control (CONTROL)


def test_surface_switch_activates_on_rejection_then_follows_k_hat():
    sw = tvlqg.SurfaceSwitch(0.0, 3.0)
    assert not sw.update(-0.2, 0.1)           # 2 sigma: plan's surface kept
    assert sw.k_bar == 0.0 and not sw.active
    assert sw.update(-0.31, 0.1)              # 3.1 sigma: rejected, fires once
    assert sw.k_bar == pytest.approx(-0.31)
    assert not sw.update(-0.05, 0.03)         # latched: follows k_hat
    assert sw.active and sw.k_bar == pytest.approx(-0.05)
    assert not sw.update(float("nan"), 0.1)


def _closed_loop(mode: str, k_true=-0.33, seed=0, T=40.0):
    """Track a plan made for k = 0 (moving turns both ways) on a plant that
    skids with k_true. mode: 'plain' (sideslip_k 0), 'typed' (sideslip_k =
    k_true), 'tvlqg' (k_bar from the filter). Returns (max |e_cross|, final
    k_bar or nan)."""
    from agx_planning.runtime_corrector import tvlqr
    rng = np.random.default_rng(seed)

    def ref_cmd(t):
        return 0.5, 0.8 * math.sin(2.0 * math.pi * t / 8.0)

    n = int(round(T / DT))
    plan = [np.zeros(tvlqg.N_S)]
    for i in range(n):
        plan.append(tvlqg.propagate(plan[-1], *ref_cmd(i * DT), DT))
    cfg = tvlqr.TVLQRConfig(enabled=True,
                            sideslip_k=k_true if mode == "typed" else 0.0)
    gains = tvlqr.GainCache(cfg, DT)
    fcfg = tvlqg.TVLQGConfig()
    truth = np.array([0.0, 0.0, 0.0, 0.0, k_true, 0.0, 0.0])
    f = tvlqg.SurfaceFilter(fcfg, truth[:3]) if mode == "tvlqg" else None
    sw = tvlqg.SurfaceSwitch(0.0, 3.0)
    caches = {}
    worst, cmd = 0.0, (0.0, 0.0)
    for i in range(n):
        if f is not None:
            if i:
                f.predict(*cmd, DT)
            f.update_pose(truth[:3] + rng.normal(0, [fcfg.r_pos, fcfg.r_pos, fcfg.r_theta]))
            f.update_gyro(truth[tvlqg.IW] + rng.normal(0, fcfg.r_gyro))
            sw.update(*f.k)
            kq = round(sw.k_bar / fcfg.k_bucket) * fcfg.k_bucket
            gains = caches.setdefault(kq, tvlqr.GainCache(
                tvlqr.TVLQRConfig(enabled=True, sideslip_k=kq), DT))
            cfg = tvlqr.TVLQRConfig(enabled=True, sideslip_k=sw.k_bar)
        v_ref, w_ref = ref_cmd(i * DT)
        err = tvlqr.slip_error(tvlqr.tracking_error(plan[i][:3], truth[:3]), v_ref, w_ref, cfg)
        v, w, _ = tvlqr.correct(gains.get(v_ref, w_ref), err, v_ref, w_ref, cfg)
        cmd = (v, w)
        for _ in range(SUB):
            truth = tvlqg.propagate(truth, v, w, DT / SUB)
        e = tvlqr.tracking_error(plan[i + 1][:3], truth[:3])
        if i * DT > 10.0:                     # after the filter's first turns
            worst = max(worst, abs(e[1]))
    return worst, (sw.k_bar if f is not None else float("nan"))


def test_tvlqg_closes_most_of_the_gap_to_the_typed_in_slide():
    """The control step's acceptance test: with k ESTIMATED, the steady
    cross-track error is near what typing in the true k gives, and well below
    plain TVLQR's."""
    plain, _ = _closed_loop("plain")
    typed, _ = _closed_loop("typed")
    assert typed < 0.7 * plain
    for seed in range(3):
        est, k_bar = _closed_loop("tvlqg", seed=seed)
        assert abs(k_bar + 0.33) < 0.1
        assert est < typed + 0.3 * (plain - typed)


def test_tvlqg_leaves_a_no_slide_plant_alone():
    """Real wood (k ~ -0.03): the level k = 0 is never rejected, so TVLQG is
    plain TVLQR exactly."""
    est, k_bar = _closed_loop("tvlqg", k_true=-0.03)
    plain, _ = _closed_loop("plain", k_true=-0.03)
    assert k_bar == 0.0
    assert est == pytest.approx(plain, abs=0.01)


def test_deviation_bound_widens_while_k_is_unknown_and_sits_near_half_a_metre():
    # Straight 3 s, a moving turn 4 s, straight 6 s at 0.5 m/s, 0.1 s ticks.
    import dataclasses

    from agx_planning.runtime_corrector import tvlqr
    dt = 0.1
    w = np.r_[np.zeros(30), np.full(40, 0.6), np.zeros(60)]
    v = np.full_like(w, 0.5)
    th = np.cumsum(w) * dt
    poses = np.c_[np.cumsum(v * np.cos(th)) * dt, np.cumsum(v * np.sin(th)) * dt, th]

    def bound(**kw):
        cfg = dataclasses.replace(tvlqg.TVLQGConfig(), **kw)
        return tvlqg.deviation_bound_along_plan(poses, v, w, dt, cfg, tvlqr.TVLQRConfig(), 3.29)

    b = bound()
    assert b.shape == (len(w),) and np.all(b > 0)
    # The slide's prior (k unknown, sigma 0.3) widens the bound in the first
    # moving turn against a surface known in advance; the effect is large
    # without the fitted loop noise and small (~5 cm) with it.
    known = bound(sigma_k0=0.001)
    assert b[40:60].max() - known[40:60].max() > 0.03
    assert (bound(q_loop_cross=0.0)[40:60].max()
            > 1.5 * bound(q_loop_cross=0.0, sigma_k0=0.001)[40:60].max())
    # Same size as the bench-chosen offplan_dist (0.5 m) -- the fit's finding.
    assert 0.4 < np.median(b) < 0.6
