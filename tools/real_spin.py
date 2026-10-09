#!/usr/bin/env python3
"""Commanded vs achieved twist per constant segment of a real-robot drive.py run.

Reads can.log (`candump -ta can0`), imu.csv (bag2csv.py) and drive.log
(segment start times) from the run directory. Per segment with a non-zero
command, over its hold window (ramp-in and the last 0.3 s dropped):

  gyro        wz from /imu/data (the ground truth for yaw rate)
  chassis     (v, w) the firmware believes it achieved (CAN 0x221)
  odometer    per-side surface speed from CAN 0x311 (mm counters, firmware's
              own wheel-radius constant -- calibrate against a tape)
  gain_w      gyro wz / commanded w   (what a twist-only corrector must invert)
  chi         odometer-implied w (geometric track) / gyro wz
  current     mean |motor current| (0x251-0x254, 0.1 A units)

--clip drops samples whose 0.5 s moving mean departs from the segment median
by more than that fraction, and reports how many.

    python3 tools/real_spin.py RUN_DIR [--clip 0.15]
"""
import argparse
import os
import struct

import numpy as np

TRACK = 0.416503  # scout_mini.xacro, as pmp_planner/config.py


def load_can(path):
    can = {}
    for line in open(path):
        p = line.split()
        if len(p) < 5:
            continue
        n = int(p[3].strip("[]"))
        can.setdefault(int(p[2], 16), []).append(
            (float(p[0].strip("()")), bytes(int(x, 16) for x in p[4:4 + n])))
    return can


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--clip", type=float, default=0.15)
    a = ap.parse_args()
    d = a.run_dir
    can = load_can(os.path.join(d, "can.log"))
    ms = np.array([(t, *struct.unpack(">hh", b[:4])) for t, b in can[0x221]], float)
    ms[:, 1:] /= 1000.0
    od = np.array([(t, *struct.unpack(">ii", b[:8])) for t, b in can[0x311]], float)
    od[:, 1:] /= 1000.0
    cur = {i: np.array([(t, abs(struct.unpack(">hh", b[:4])[1]) * 0.1) for t, b in can[i]])
           for i in (0x251, 0x252, 0x253, 0x254)}
    imu = np.loadtxt(os.path.join(d, "imu.csv"), delimiter=",")
    segs = []
    for line in open(os.path.join(d, "drive.log")):
        if " seg " in line:
            s = line.split()
            segs.append((float(s[0]), float(s[2][2:]), float(s[3][2:]), float(s[4][5:])))
    print(f"{'v_cmd':>6} {'w_cmd':>6} {'win':>5} {'gyro_w':>7} {'gain_w':>6} {'fw_w':>6} {'fw_v':>6} "
          f"{'odoL':>6} {'odoR':>6} {'chi':>5} {'I[A]':>5} {'kept':>5}")
    for t, v, w, hold in segs:
        if abs(w) < 1e-9 and abs(v) < 1e-9:
            continue
        lo, hi = t + 1.5, t + 1.0 + hold - 0.3
        m = (imu[:, 0] >= lo) & (imu[:, 0] <= hi)
        ti, gz = imu[m, 0], imu[m, 1]
        k = max(1, int(0.5 / np.median(np.diff(ti))))
        mov = np.convolve(gz, np.ones(k) / k, mode="same")
        med = np.median(gz)
        ok = np.abs(mov - med) <= a.clip * max(abs(med), 0.05)
        keep = ti[ok]
        # per-signal means restricted to the kept gyro time span (nearest-sample mask)
        def kept_mask(tt):
            j = np.clip(np.searchsorted(ti, tt), 0, len(ti) - 1)
            return (tt >= lo) & (tt <= hi) & ok[j]
        gw = gz[ok].mean()
        mm = kept_mask(ms[:, 0])
        # odometer speed by finite difference, then masked mean
        vod = np.gradient(od[:, 1:], od[:, 0], axis=0)
        mo = kept_mask(od[:, 0])
        vl, vr = vod[mo].mean(axis=0)
        w_ideal = (vr - vl) / TRACK
        i_mean = np.mean([c[kept_mask(c[:, 0]), 1].mean() for c in cur.values()])
        gain = gw / w if abs(w) > 1e-9 else float("nan")
        chi = w_ideal / gw if abs(gw) > 0.05 else float("nan")
        print(f"{v:+6.2f} {w:+6.2f} {hi - lo:5.1f} {gw:+7.3f} {gain:6.3f} {ms[mm, 2].mean():+6.3f} "
              f"{ms[mm, 1].mean():+6.3f} {vl:+6.3f} {vr:+6.3f} {chi:5.2f} {i_mean:5.2f} {ok.mean():5.0%}")


if __name__ == "__main__":
    main()
