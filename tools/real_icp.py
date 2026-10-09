#!/usr/bin/env python3
"""Lidar odometry for a real-robot run using KISS-ICP.

Runs KISS-ICP over /lidar/points of the run's bag and writes <run>/icp.csv (t,
x, y, yaw) in the lidar frame, with the first scan as origin. It stands in for
ground truth when measuring the sideways slide k, and compares the ICP yaw rate
with the gyro (imu.csv) as a sanity check.

Needs Python <= 3.13 for the kiss-icp wheels:
    uv venv -p 3.12 .venv-icp && uv pip install -p .venv-icp/bin/python kiss-icp rosbags
    .venv-icp/bin/python tools/real_icp.py RUN_DIR
"""
import argparse
import os

import numpy as np
from kiss_icp.config import KISSConfig
from kiss_icp.kiss_icp import KissICP
from rosbags.highlevel import AnyReader
from pathlib import Path

DT = {1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16, 5: np.int32, 6: np.uint32, 7: np.float32, 8: np.float64}


def cloud_xyz(msg):
    """PointCloud2 -> (N,3) float64 finite points and per-point time if present."""
    fields = {f.name: (f.offset, DT[f.datatype]) for f in msg.fields}
    dtype = np.dtype({"names": list(fields), "formats": [v[1] for v in fields.values()],
                      "offsets": [v[0] for v in fields.values()], "itemsize": msg.point_step})
    a = np.frombuffer(bytes(msg.data), dtype=dtype, count=msg.width * msg.height)
    xyz = np.stack([a["x"], a["y"], a["z"]], axis=1).astype(np.float64)
    ok = np.isfinite(xyz).all(axis=1) & (np.linalg.norm(xyz, axis=1) > 0.3)
    ts = a["timestamp"].astype(np.float64)[ok] if "timestamp" in fields else None
    return xyz[ok], ts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--max-range", type=float, default=30.0)
    a = ap.parse_args()
    cfg = KISSConfig()
    cfg.data.max_range = a.max_range
    cfg.data.min_range = 0.3
    cfg.data.deskew = True
    cfg.mapping.voxel_size = 0.1  # indoor; default derives 1% of max_range
    odo = KissICP(cfg)
    rows = []
    with AnyReader([Path(a.run_dir) / "bag"]) as r:
        conns = [c for c in r.connections if c.topic == "/lidar/points"]
        for c, t_rec, raw in r.messages(connections=conns):
            m = r.deserialize(raw, c.msgtype)
            t = m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec
            xyz, ts = cloud_xyz(m)
            if ts is not None and ts.size:
                ts = (ts - ts.min()) / max(ts.max() - ts.min(), 1e-9)  # kiss-icp wants [0,1]
            else:
                ts = np.zeros(len(xyz))
            odo.register_frame(xyz, ts)
            P = odo.last_pose
            rows.append((t, P[0, 3], P[1, 3], np.arctan2(P[1, 0], P[0, 0])))
    rows = np.array(rows)
    np.savetxt(os.path.join(a.run_dir, "icp.csv"), rows, delimiter=",", fmt="%.6f")
    t, x, y, yaw = rows.T
    yaw = np.unwrap(yaw)
    print(f"{len(rows)} scans, {t[-1] - t[0]:.1f} s; end pose x {x[-1]:+.3f} y {y[-1]:+.3f} "
          f"yaw {np.degrees(yaw[-1]):+.1f} deg; path {np.sum(np.hypot(np.diff(x), np.diff(y))):.3f} m")
    imu_f = os.path.join(a.run_dir, "imu.csv")
    if os.path.exists(imu_f):
        imu = np.loadtxt(imu_f, delimiter=",")
        g = np.concatenate([[0], np.cumsum(0.5 * (imu[1:, 1] + imu[:-1, 1]) * np.diff(imu[:, 0]))])
        gy = np.interp(t, imu[:, 0], g)
        gy -= gy[0]
        err = np.degrees(yaw - yaw[0] - gy)
        print(f"ICP yaw - gyro yaw: end {err[-1]:+.2f} deg, max |.| {np.abs(err).max():.2f} deg "
              f"(gyro total {np.degrees(gy[-1]):+.1f} deg)")


if __name__ == "__main__":
    main()
