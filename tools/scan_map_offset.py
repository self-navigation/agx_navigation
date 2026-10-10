#!/usr/bin/env python3
"""Check whether the lidar sees the walls where the baked map draws them.

Run against a fixture started with localization:=amcl (which brings up /scan),
standing still at a known spawn pose with no goal. Collects --n scans and
/amcl_pose, projects the scan endpoints into the map from the true pose, and
grid-searches the rigid (dx, dy, dyaw) shift that best aligns them with the
baked wall pixels on the wall distance transform. A consistent shift across
poses indicates a map/scan offset; a shift near zero points at amcl instead.

    python3 tools/scan_map_offset.py --truth X Y YAW [--n 20] [--out f.json]
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseWithCovarianceStamped
from PIL import Image
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from scipy.ndimage import distance_transform_edt
from sensor_msgs.msg import LaserScan

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MAP = os.path.join(R, "src/rudn-ordjo-building/maps/floor_6")


def load_map():
    meta = yaml.safe_load(open(MAP + ".yaml"))
    img = np.asarray(Image.open(MAP + ".png").convert("L"))[::-1]  # row 0 = origin y
    wall = img == 0
    d = distance_transform_edt(~wall) * meta["resolution"]
    return d, meta["resolution"], meta["origin"][0], meta["origin"][1]


def wall_dist(d, res, ox, oy, x, y):
    c = np.clip(((x - ox) / res).astype(int), 0, d.shape[1] - 1)
    r = np.clip(((y - oy) / res).astype(int), 0, d.shape[0] - 1)
    return d[r, c]


class Probe(Node):
    def __init__(self, n):
        super().__init__("scan_map_offset")
        self.n, self.scans, self.amcl = n, [], []
        self.create_subscription(LaserScan, "/scan", self._scan, qos_profile_sensor_data)
        self.create_subscription(PoseWithCovarianceStamped, "/amcl_pose", self._amcl, 10)

    def _scan(self, m):
        if len(self.scans) < self.n:
            a = m.angle_min + m.angle_increment * np.arange(len(m.ranges))
            r = np.asarray(m.ranges, dtype=float)
            ok = np.isfinite(r) & (r > m.range_min) & (r < min(m.range_max, 15.0))
            self.scans.append(np.c_[r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok])])

    def _amcl(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        self.amcl.append((p.x, p.y, 2 * math.atan2(q.z, q.w)))


def best_shift(pts_body, truth, d, res, ox, oy, span=0.30, step=0.01, yaw_span=0.03):
    x0, y0, th0 = truth
    best = None
    for dth in np.arange(-yaw_span, yaw_span + 1e-9, 0.005):
        c, s = math.cos(th0 + dth), math.sin(th0 + dth)
        wx = x0 + c * pts_body[:, 0] - s * pts_body[:, 1]
        wy = y0 + s * pts_body[:, 0] + c * pts_body[:, 1]
        for dx in np.arange(-span, span + 1e-9, step):
            for dy in np.arange(-span, span + 1e-9, step):
                e = wall_dist(d, res, ox, oy, wx + dx, wy + dy)
                cost = float(np.mean(np.minimum(e, 0.3) ** 2))  # truncated: robust to clutter
                if best is None or cost < best[0]:
                    best = (cost, dx, dy, dth)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", nargs=3, type=float, required=True, metavar=("X", "Y", "YAW"))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--timeout", type=float, default=120.0, help="wall-clock seconds")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rclpy.init()
    node = Probe(a.n)
    import time
    t_end = time.time() + a.timeout
    while rclpy.ok() and len(node.scans) < a.n and time.time() < t_end:
        rclpy.spin_once(node, timeout_sec=0.2)
    for _ in range(25):  # a few more amcl updates
        rclpy.spin_once(node, timeout_sec=0.2)
    if not node.scans:
        print("no /scan received", file=sys.stderr)
        return 2
    d, res, ox, oy = load_map()
    pts = np.vstack(node.scans)
    truth = tuple(a.truth)
    c, s = math.cos(truth[2]), math.sin(truth[2])
    wx = truth[0] + c * pts[:, 0] - s * pts[:, 1]
    wy = truth[1] + s * pts[:, 0] + c * pts[:, 1]
    e0 = wall_dist(d, res, ox, oy, wx, wy)
    cost, dx, dy, dth = best_shift(pts[:: max(1, len(pts) // 4000)], truth, d, res, ox, oy)
    out = dict(truth=truth, n_scans=len(node.scans), n_pts=int(len(pts)),
               scan_wall_dist_median=float(np.median(e0)), scan_wall_dist_p90=float(np.percentile(e0, 90)),
               frac_within_5cm=float(np.mean(e0 <= 0.05)),
               best_shift=dict(dx=round(dx, 3), dy=round(dy, 3), dyaw=round(dth, 4), cost=cost),
               amcl_last=node.amcl[-1] if node.amcl else None,
               amcl_minus_truth=([round(node.amcl[-1][i] - truth[i], 3) for i in range(3)]
                                 if node.amcl else None))
    print(json.dumps(out, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
