#!/usr/bin/env python3
"""Total mass, CoM and yaw inertia of a URDF, for the planar surrogate.

Composes joint origins at zero angles, moves each link's inertial into the
root frame and sums mass, centre of mass and the inertia about the vertical
axis through the CoM. These are DynamicParams.mass and .izz in
rl_corrector.dynamic_bridge; re-run when the robot description changes.

    xacro src/scout_ros2/scout_description/urdf/scout_mini.xacro sim:=true \\
        sim_sensors:=false > /tmp/scout.urdf
    python3 tools/urdf_planar_inertia.py /tmp/scout.urdf
"""

import sys
import xml.etree.ElementTree as ET

import numpy as np


def rpy_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def origin(el):
    o = el.find("origin") if el is not None else None
    xyz = np.fromstring(o.get("xyz", "0 0 0"), sep=" ") if o is not None else np.zeros(3)
    rpy = np.fromstring(o.get("rpy", "0 0 0"), sep=" ") if o is not None else np.zeros(3)
    return xyz, rpy_matrix(*rpy)


def main(path):
    root = ET.parse(path).getroot()
    links = {l.get("name"): l for l in root.findall("link")}
    joints = root.findall("joint")
    children = {j.find("child").get("link") for j in joints}
    base = next(n for n in links if n not in children)
    pose = {base: (np.zeros(3), np.eye(3))}
    pending = list(joints)
    while pending:
        rest = []
        for j in pending:
            par, ch = j.find("parent").get("link"), j.find("child").get("link")
            if par not in pose:
                rest.append(j); continue
            pp, pr = pose[par]
            xyz, rot = origin(j)
            pose[ch] = (pp + pr @ xyz, pr @ rot)
        if len(rest) == len(pending):
            raise SystemExit(f"unreachable joints: {[j.get('name') for j in rest]}")
        pending = rest

    items = []
    for name, l in links.items():
        inn = l.find("inertial")
        if inn is None:
            continue
        m = float(inn.find("mass").get("value"))
        a = {k: float(v) for k, v in inn.find("inertia").attrib.items()}
        I = np.array([[a["ixx"], a["ixy"], a["ixz"]], [a["ixy"], a["iyy"], a["iyz"]],
                      [a["ixz"], a["iyz"], a["izz"]]])
        lp, lr = pose[name]
        xyz, rot = origin(inn)
        R = lr @ rot
        items.append((name, m, lp + lr @ xyz, R @ I @ R.T))

    M = sum(m for _, m, _, _ in items)
    com = sum(m * c for _, m, c, _ in items) / M
    print(f"root link: {base}")
    print(f"{'link':28s} {'mass':>7s} {'x':>8s} {'y':>8s} {'z':>8s} {'Izz_own':>9s} {'m*d^2':>9s}")
    izz = 0.0
    for name, m, c, I in items:
        d2 = (c[0] - com[0]) ** 2 + (c[1] - com[1]) ** 2
        izz += I[2, 2] + m * d2
        print(f"{name:28s} {m:7.3f} {c[0]:8.4f} {c[1]:8.4f} {c[2]:8.4f} {I[2, 2]:9.5f} {m * d2:9.5f}")
    print(f"total mass {M:.4f} kg")
    print(f"CoM (root frame) {com[0]:.4f} {com[1]:.4f} {com[2]:.4f} m")
    print(f"Izz about vertical axis through CoM {izz:.4f} kg m^2")


if __name__ == "__main__":
    main(sys.argv[1])
