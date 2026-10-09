"""Real Scout Mini standing still for 20 min (2026-10-09): IMU noise, bias drift (Allan deviation) and odometry at rest.

Source: run_data/2026-10-09_static_noise/{imu6.csv, odom.csv} (from the bag via run_data/2026-10-09_tools/bag2imu6.py).
Run: .venv/bin/python figures/2026-10-09/real_static_noise/render.py   (writes stats.txt and the PNGs here)
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
RUN = HERE.parents[2] / "run_data" / "2026-10-09_static_noise"
imu = np.loadtxt(RUN / "imu6.csv", delimiter=",")
odo = np.loadtxt(RUN / "odom.csv", delimiter=",")
# The first ~22 s the user was driving the robot out of the way by RC (odom v up to 1 m/s, 173 deg of yaw);
# keep from T_CUT on. Two later odom blips (51.5 s, 319.3 s) are single encoder ticks, not motion.
T_CUT = 60.0
t_all0 = imu[0, 0]
imu = imu[imu[:, 0] - t_all0 >= T_CUT]
odo = odo[odo[:, 0] - t_all0 >= T_CUT]
t = imu[:, 0] - imu[0, 0]
dt = np.median(np.diff(t))
names = ["wx", "wy", "wz", "ax", "ay", "az"]
out = []
p = lambda *a: out.append(" ".join(str(x) for x in a))

p(f"window: from {T_CUT:.0f} s after the first IMU sample")
p(f"IMU: n={len(t)}, span {t[-1]:.1f} s, median dt {dt*1e3:.2f} ms, max gap {np.diff(t).max()*1e3:.1f} ms")
p(f"odom: n={len(odo)}, median dt {np.median(np.diff(odo[:,0]))*1e3:.2f} ms")


def adev(x, dt):
    """Overlapping Allan deviation on log-spaced cluster sizes."""
    n = len(x)
    ms = np.unique(np.logspace(0, np.log10(n // 9), 60).astype(int))
    th = np.cumsum(np.r_[0, x]) * dt
    taus, ad = [], []
    for m in ms:
        tau = m * dt
        d = th[2 * m:] - 2 * th[m:-m] + th[:-2 * m]
        taus.append(tau)
        ad.append(np.sqrt(np.sum(d ** 2) / (2 * tau ** 2 * (n - 2 * m))))
    return np.array(taus), np.array(ad)


fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
res = {}
for i, nm in enumerate(names):
    x = imu[:, 1 + i]
    tau, ad = adev(x, dt)
    # white-noise density: ADEV at tau=1 s on the -1/2 slope fit over 0.1..3 s
    sel = (tau > 0.1) & (tau < 3)
    N = np.exp(np.mean(np.log(ad[sel]) + 0.5 * np.log(tau[sel])))
    B = ad.min() / 0.664
    tmin = tau[ad.argmin()]
    drift = np.polyfit(t, x, 1)[0] * 60
    res[nm] = (tau, ad)
    unit = "rad/s" if nm[0] == "w" else "m/s^2"
    p(f"{nm}: mean {x.mean():+.5f} {unit}, sd {x.std():.5f}, density N {N:.2e} {unit}/sqrt(Hz), "
      f"bias instability {B:.2e} at tau {tmin:.0f} s, linear drift {drift:+.2e} {unit}/min")
    ax = axs[0] if nm[0] == "w" else axs[1]
    ax.loglog(tau, ad, label=nm)
for ax, u in zip(axs, ["rad/s", "m/s²"]):
    ax.set_xlabel("τ, s"); ax.set_ylabel(f"Allan deviation, {u}"); ax.grid(True, which="both", alpha=.3); ax.legend()
axs[0].set_title("gyro"); axs[1].set_title("accelerometer")
fig.suptitle("Real Scout Mini at rest, 20 min (2026-10-09)")
fig.tight_layout(); fig.savefig(HERE / "allan.png", dpi=130)

wz = imu[:, 3]
p(f"wz mean in deg/h: {np.degrees(wz.mean())*3600:+.1f}; integrated yaw over the run {np.degrees(np.sum(wz)*dt):+.2f} deg; "
  f"after removing the mean bias {np.degrees(np.sum(wz - wz.mean())*dt):+.3f} deg")
# first vs last 5 min bias
q = len(wz) // 4
p(f"wz bias first 5 min {wz[:q].mean():+.2e}, last 5 min {wz[-q:].mean():+.2e} rad/s")

to = odo[:, 0] - imu[0, 0]
p(f"odom at rest: v nonzero samples {np.count_nonzero(odo[:,4])}, w nonzero {np.count_nonzero(odo[:,5])}; "
  f"v range [{odo[:,4].min():.4f}, {odo[:,4].max():.4f}], w range [{odo[:,5].min():.4f}, {odo[:,5].max():.4f}]")
p(f"odom pose drift: dx {odo[-1,1]-odo[0,1]:+.5f} m, dy {odo[-1,2]-odo[0,2]:+.5f} m, dyaw {np.degrees(odo[-1,3]-odo[0,3]):+.4f} deg")

# footstep check: accel z variance in 10 s windows
w = int(10 / dt)
az = imu[: len(imu) // w * w, 6].reshape(-1, w)
p(f"az sd per 10 s window: min {az.std(1).min():.4f}, median {np.median(az.std(1)):.4f}, max {az.std(1).max():.4f} m/s^2")

fig, axs = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
k = 100
tt = t[: len(t) // k * k].reshape(-1, k).mean(1)
axs[0].plot(t, wz * 1e3, lw=.3, color="0.6", label="raw 100 Hz")
axs[0].plot(tt, wz[: len(tt) * k].reshape(-1, k).mean(1) * 1e3, color="C0", label="1 s mean")
axs[0].set_ylabel("ω_z, mrad/s"); axs[0].legend(loc="upper right")
axs[1].plot(t, np.degrees(np.cumsum(wz - wz.mean()) * dt), color="C1")
axs[1].set_ylabel("∫(ω_z − mean) dt, deg")
axs[2].plot(tt, imu[: len(tt) * k, 6].reshape(-1, k).std(1), color="C2")
axs[2].set_ylabel("a_z sd per 1 s, m/s²"); axs[2].set_xlabel("t, s")
axs[2].set_xlabel(f"t − {T_CUT:.0f} s, s")
fig.suptitle("Yaw gyro and floor vibration at rest")
fig.tight_layout(); fig.savefig(HERE / "timeseries.png", dpi=130)

(HERE / "stats.txt").write_text("\n".join(out) + "\n")
print("\n".join(out))
