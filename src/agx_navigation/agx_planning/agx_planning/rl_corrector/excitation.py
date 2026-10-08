"""Open-loop wheel-command sequences for plant identification.

Pure numpy. A sequence is per-side wheel speeds ``(w_l, w_r)`` [rad/s] on a
fixed grid ``dt``, expanded to ``[fl, rl, fr, rr] = [w_l, w_l, w_r, w_r]``.
``random_segments`` chains ramp-and-hold segments toward random targets
(forward arcs, pivots, stops, short reversals); ``plan_commands`` resamples a
library plan's nominal ``wheel_cmds`` to ``dt`` with a zero-order hold. Feeding
the same sequence to Gazebo and to the surrogate (``dynamic_bridge``) exposes
their open-loop mismatch.
"""

from dataclasses import dataclass

import numpy as np

W_MAX = 13.0          # [rad/s] plans reach 12.7-13.2
ACC_MIN, ACC_MAX = 4.0, 35.0   # [rad/s^2] plans: p99 16.5, max 34.7
SEG_MIN_S, SEG_MAX_S = 0.6, 4.0

# kind -> probability. Forward arcs dominate because that is what plans do.
KINDS = (("arc", 0.55), ("pivot", 0.15), ("straight", 0.12), ("stop", 0.10),
         ("reverse", 0.08))


@dataclass
class Sequence:
    name: str
    dt: float
    wheels: np.ndarray        # (N, 4) [fl, rl, fr, rr] rad/s

    @property
    def duration(self) -> float:
        return self.wheels.shape[0] * self.dt


def sides_to_wheels(wl, wr) -> np.ndarray:
    wl, wr = np.asarray(wl, float), np.asarray(wr, float)
    return np.stack([wl, wl, wr, wr], axis=-1)


def _target(kind: str, rng: np.random.Generator):
    if kind == "arc":
        mean = rng.uniform(1.0, W_MAX)
        # |diff| <= 2*mean: from a gentle arc down to one side stopped
        # (the turn-on-the-spot-ish end of a plan's corners); clipped below.
        diff = rng.uniform(-1, 1) * 2.0 * mean * rng.uniform(0.1, 1.0)
        return mean - diff / 2, mean + diff / 2
    if kind == "pivot":
        w = rng.uniform(2.0, 6.0) * rng.choice([-1, 1])   # plans: |w_r - w_l| <= 11
        return -w, w
    if kind == "straight":
        w = rng.uniform(2.0, W_MAX)
        return w, w
    if kind == "stop":
        return 0.0, 0.0
    if kind == "reverse":
        mean = -rng.uniform(1.0, 6.0)
        diff = rng.uniform(-3, 3)
        return mean - diff / 2, mean + diff / 2
    raise ValueError(kind)


def random_segments(seed: int, duration: float = 60.0, dt: float = 0.02) -> Sequence:
    """Seeded segment-chain excitation; identical for a given (seed, duration, dt)."""
    rng = np.random.default_rng(seed)
    names, probs = zip(*KINDS)
    n = int(round(duration / dt))
    out = np.zeros((n, 2))
    cur = np.zeros(2)
    i = 0
    while i < n:
        kind = names[rng.choice(len(names), p=np.asarray(probs) / sum(probs))]
        tgt = np.clip(np.array(_target(kind, rng)), -W_MAX, W_MAX)
        acc = rng.uniform(ACC_MIN, ACC_MAX)
        hold = rng.uniform(SEG_MIN_S, SEG_MAX_S)
        steps = int(round(hold / dt))
        for _ in range(steps):
            if i >= n:
                break
            delta = np.clip(tgt - cur, -acc * dt, acc * dt)
            cur = cur + delta
            out[i] = cur
            i += 1
    # Every sequence ends at rest so the final pose is a settled one.
    tail = int(round(1.0 / dt))
    out[-tail:] = 0.0
    return Sequence(f"rand_s{seed}", dt, sides_to_wheels(out[:, 0], out[:, 1]))


def plan_commands(path: str, dt: float = 0.02, tail_s: float = 1.0) -> Sequence:
    """A library plan's nominal per-side commands, zero-order held onto `dt`."""
    import os
    with np.load(path) as f:
        cmds = np.asarray(f["wheel_cmds"], float)
        dts = float(f["dt_sample"]) if "dt_sample" in f else 0.1
    t = np.arange(0.0, cmds.shape[0] * dts, dt)
    idx = np.minimum((t / dts).astype(int), cmds.shape[0] - 1)
    sides = np.vstack([cmds[idx], np.zeros((int(round(tail_s / dt)), 2))])
    name = "plan_" + os.path.splitext(os.path.basename(path))[0]
    return Sequence(name, dt, sides_to_wheels(sides[:, 0], sides[:, 1]))


def run_sequence(bridge, seq: Sequence, start_pose=(0.0, 0.0, 0.0), terrain=None) -> dict:
    """Drive `bridge` open loop through `seq`; returns arrays, row k = after step k.

    ``t`` is control time (k+1)*dt; row 0 of ``pose`` etc. is the reset state
    (t = 0), so ``pose`` has N+1 rows and ``cmd`` N.
    """
    st = bridge.reset(start_pose, terrain)
    poses, vs, ws, wheels, imus, contact = [st.pose], [st.v], [st.omega], [st.wheel_speeds], [st.imu], [st.contact]
    for cmd in seq.wheels:
        st = bridge.step(list(map(float, cmd)), seq.dt)
        poses.append(st.pose); vs.append(st.v); ws.append(st.omega)
        wheels.append(st.wheel_speeds); imus.append(st.imu); contact.append(st.contact)

    def arr(xs, width):
        return np.array([x if x is not None else [np.nan] * width for x in xs], float)
    return dict(
        name=seq.name, dt=seq.dt, cmd=seq.wheels,
        t=np.arange(len(poses)) * seq.dt,
        pose=np.array(poses, float), v_odom=np.array(vs, float), omega_odom=np.array(ws, float),
        wheel_speeds=arr(wheels, 4), imu=arr(imus, 3), contact=np.array(contact, bool),
    )
