"""rejoin_env and rejoin_policy tests: pure numpy, synthetic plans."""

import numpy as np
import pytest

from agx_planning.rl_corrector import rejoin_env as renv
from agx_planning.rl_corrector.config import RLCorrectorConfig
from agx_planning.rl_corrector.rejoin_policy import RejoinPolicy
from agx_planning.runtime_corrector import tvlqr
from agx_planning.supervisor.walls import SignedWallDistance

KIN = RLCorrectorConfig()
DT = 0.1


def _arc_plan(n=200, v=0.4, om=0.0):
    """Constant-twist plan, the EXACT arc sampled on the tick (real plans come
    from a continuous solver), wheels from the twist."""
    wl, wr = KIN.body_to_wheels(v, om)
    t = np.arange(n) * DT
    p = np.zeros((n, 5))
    p[:, 2] = om * t
    if om == 0:
        p[:, 0] = v * t
    else:
        p[:, 0], p[:, 1] = v / om * np.sin(om * t), v / om * (1 - np.cos(om * t))
    p[:, 3], p[:, 4] = wl, wr
    return p


def _walls():
    wall = np.zeros((400, 400), bool)
    wall[0], wall[-1], wall[:, 0], wall[:, -1] = True, True, True, True
    return SignedWallDistance.from_wall_mask(wall, 0.05, -10.0, -10.0)


def _labels(plan, ks, m, dev):
    n = len(ks)
    traj = np.zeros((n, renv.TRAJ_POINTS, 5))
    for j, k in enumerate(ks):  # on-plan "teacher": the plan itself
        f = k + np.linspace(0, m, renv.TRAJ_POINTS)
        for c in range(5):
            traj[j, :, c] = np.interp(f, np.arange(len(plan)), plan[:, c])
    s = np.zeros((n, 10))
    s[:, 9] = m * DT
    return dict(s=s, traj=traj, m=np.ones(n, int), j_chi0=np.zeros(n), plan=np.array(["p"] * n),
                k=np.array(ks), tgt=np.array(ks) + m, dev=np.asarray(dev, float))


def test_batched_gains_match_gain_schedule():
    rng = np.random.default_rng(0)
    v, om = rng.uniform(-0.1, 0.5, (3, 40)), rng.uniform(-1, 1, (3, 40))
    valid = np.ones((3, 40), bool)
    valid[1, 25:] = False
    cfg = tvlqr.TVLQRConfig()
    K = renv.batched_gains(v, om, valid, DT, cfg)
    for b, n in ((0, 40), (1, 25), (2, 40)):
        G = np.array(tvlqr.gain_schedule(v[b, :n], om[b, :n], DT, cfg))
        assert np.abs(G - K[b, :n]).max() < 1e-12
    assert np.all(K[1, 25:] == 0)


@pytest.mark.parametrize("om", [0.0, 0.3])
def test_on_plan_teacher_costs_nothing_at_nominal_chi(om):
    plan = _arc_plan(om=om)
    lab = _labels(plan, [10, 50], 40, np.zeros((2, 5)))
    env = renv.RejoinEnv({"p": plan}, lab, _walls())
    ii = np.arange(2)
    r = env.rollout(ii, env.a_teacher[ii], np.full(2, renv.CHI_NOMINAL))
    assert np.all(r["pin"] < 2e-3)
    assert np.all(r["cost"] < 1e-4)
    # the teacher's knots ARE the plan's knots on-plan: zero residual
    assert np.abs(env.teacher_residual(ii)).max() < 1e-9


def test_yaw_deficit_costs_on_a_turn_not_on_a_straight():
    for om, expect_cost in ((0.0, False), (0.3, True)):
        plan = _arc_plan(om=om)
        lab = _labels(plan, [10], 60, np.zeros((1, 5)))
        env = renv.RejoinEnv({"p": plan}, lab, _walls())
        r = env.rollout([0], env.a_teacher[[0]], np.array([2.0]))
        assert (r["cost"][0] > 1e-3) == expect_cost


def test_tvlqr_recovers_a_lateral_offset_the_plan_knots_do_not():
    plan = _arc_plan()
    lab = _labels(plan, [10], 80, [[0.0, 0.3, 0.0, 0.0, 0.0]])
    env = renv.RejoinEnv({"p": plan}, lab, _walls())
    # following the plan's twist from x0 keeps the 0.3 m offset to the end
    r = env.rollout([0], env.plan_knots[[0]], np.array([renv.CHI_NOMINAL]))
    assert r["pin"][0] == pytest.approx(0.3, abs=0.02)


def test_state_layout():
    plan = _arc_plan(om=0.2)
    lab = _labels(plan, [5, 30], 40, np.zeros((2, 5)))
    env = renv.RejoinEnv({"p": plan}, lab, _walls())
    s = env.state(np.arange(2), np.array([1.5, 1.8]))
    assert s.shape == (2, 10 + 2 * renv.N_KNOTS)
    assert np.allclose(s[:, 8], [1.5, 1.8])
    v, om = KIN.wheels_to_body(plan[0, 3], plan[0, 4])
    assert np.allclose(s[:, 10::2], v) and np.allclose(s[:, 11::2], om)


def test_policy_export_zero_init_is_the_plan(tmp_path):
    sd, ad, h = 10 + 2 * renv.N_KNOTS, 2 * renv.N_KNOTS, 8
    rng = np.random.default_rng(0)
    w = dict(s_mean=np.zeros(sd), s_std=np.ones(sd), n_layers=np.array(3),
             W0=rng.normal(size=(h, sd)), b0=np.zeros(h), W1=rng.normal(size=(h, h)), b1=np.zeros(h),
             W2=np.zeros((ad, h)), b2=np.zeros(ad), scale=np.ones(ad), n_knots=np.array(renv.N_KNOTS))
    np.savez(tmp_path / "p.npz", **w)
    pol = RejoinPolicy(str(tmp_path / "p.npz"))
    s = rng.normal(size=(3, sd))
    assert np.allclose(pol.act(s), s[:, 10:])
    assert np.allclose(pol.act(s[0]), s[0, 10:])
