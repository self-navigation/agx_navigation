#!/usr/bin/env python3
"""Extract every Nth lidar scan of a real-robot bag into <run>/scans_sub.npz.

Stores header stamps ``t`` and per-scan ``pts_i`` (N,3): x, y in the lidar
frame and each point's time offset from the stamp, cropped to z -0.25..0.5 m
and decimated to a 3 cm grid. The time offsets let figures deskew scans taken
while turning.

    .venv-icp/bin/python tools/real_scans_extract.py RUN_DIR [--every 5]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_icp import cloud_xyz  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run_dir")
ap.add_argument("--every", type=int, default=5)
a = ap.parse_args()
ts, out = [], {}
with AnyReader([Path(a.run_dir) / "bag"]) as r:
    conns = [c for c in r.connections if c.topic == "/lidar/points"]
    for i, (c, _, raw) in enumerate(r.messages(connections=conns)):
        if i % a.every:
            continue
        m = r.deserialize(raw, c.msgtype)
        xyz, tp = cloud_xyz(m)
        tp = np.zeros(len(xyz)) if tp is None else (tp - m.header.stamp.sec) - 1e-9 * m.header.stamp.nanosec
        keep = (xyz[:, 2] > -0.25) & (xyz[:, 2] < 0.5)
        xy, tp = xyz[keep, :2], tp[keep]
        _, idx = np.unique(np.round(xy / 0.03).astype(np.int32), axis=0, return_index=True)
        out[f"pts_{len(ts)}"] = np.c_[xy[idx], tp[idx]].astype(np.float32)
        ts.append(m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec)
np.savez_compressed(os.path.join(a.run_dir, "scans_sub.npz"), t=np.array(ts), **out)
print(len(ts), "scans ->", os.path.join(a.run_dir, "scans_sub.npz"))
