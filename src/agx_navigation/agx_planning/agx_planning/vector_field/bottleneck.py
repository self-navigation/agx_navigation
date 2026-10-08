"""Bottleneck alignment: shape the field direction to stop and aim at doorways.

No ROS 2 imports. Narrow passages are found from the obstacle map as saddles of
the free-space EDT, tested in four orientations and kept if the opening is
between min_width and max_width with open space (EDT >= open_clear) `approach`
metres away on both sides. Near each one the unit direction F is replaced by a
line-following field on the passage axis,

    psi(s, n) = axis - converge_angle * (2/pi) * atan(n / converge_dist)

so the planner reaches the axis, turns in place and drives straight through.
The zone spans `approach` before and `exit` past the opening and `lateral`
either side, fading over `fade`. T is unchanged; `VectorFieldGrid` carries the
shaped direction as a separate channel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage

_DIRS = (
    (1.0, 0.0),
    (0.0, 1.0),
    (np.sqrt(0.5), np.sqrt(0.5)),
    (np.sqrt(0.5), -np.sqrt(0.5)),
)


@dataclass
class Bottleneck:
    x: float          # opening centre, world [m]
    y: float
    ax: float         # unit passage axis (sign arbitrary until oriented)
    ay: float
    width: float      # opening width, 2 * EDT at the centre [m]


def _line_footprint(d: Tuple[float, float], half_len: int) -> np.ndarray:
    """Footprint of a grid line through the centre along (dx, dy) in (x=col, y=row)."""
    n = 2 * half_len + 1
    fp = np.zeros((n, n), dtype=bool)
    dx, dy = d
    step = max(abs(dx), abs(dy))
    for k in range(-half_len, half_len + 1):
        c = half_len + int(round(k * dx / step))
        r = half_len + int(round(k * dy / step))
        fp[r, c] = True
    return fp


def find_bottlenecks(
    obstacle_mask: np.ndarray,
    resolution: float,
    origin_x: float,
    origin_y: float,
    max_width: float = 1.2,
    min_width: float = 0.6,
    approach: float = 1.0,
    open_clear: float = 0.45,
    merge_dist: float = 0.3,
) -> List[Bottleneck]:
    """Narrow passages of the free space (EDT saddles); see the module docstring."""
    free = ~obstacle_mask
    edt = ndimage.distance_transform_edt(free) * resolution
    h, w = edt.shape
    narrow = free & (edt < max_width / 2.0) & (edt >= min_width / 2.0)
    found: List[Bottleneck] = []
    for d in _DIRS:
        across = (-d[1], d[0])
        # Across: a max over +-1 step. Along: a min over +-4 steps (~0.2-0.28 m).
        mx = ndimage.maximum_filter(edt, footprint=_line_footprint(across, 1), mode="nearest")
        mn = ndimage.minimum_filter(edt, footprint=_line_footprint(d, 4), mode="nearest")
        mask = narrow & (edt >= mx - 1e-9) & (edt <= mn + 1e-9)
        lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
        for i in range(1, n + 1):
            rows, cols = np.nonzero(lab == i)
            k = int(np.argmin(edt[rows, cols]))
            r0, c0 = rows[k], cols[k]
            cx = origin_x + (c0 + 0.5) * resolution
            cy = origin_y + (r0 + 0.5) * resolution

            def clear_at(sign: float) -> bool:
                px, py = cx + sign * approach * d[0], cy + sign * approach * d[1]
                c = int((px - origin_x) / resolution)
                r = int((py - origin_y) / resolution)
                return 0 <= r < h and 0 <= c < w and edt[r, c] >= open_clear

            if not (clear_at(1.0) and clear_at(-1.0)):
                continue
            found.append(Bottleneck(cx, cy, d[0], d[1], 2.0 * float(edt[r0, c0])))
    # One opening can pass in two orientations (e.g. 0 and 45 deg); keep the
    # narrower reading, which is the one taken square to the opening.
    found.sort(key=lambda b: b.width)
    kept: List[Bottleneck] = []
    for b in found:
        if all(np.hypot(b.x - k.x, b.y - k.y) > merge_dist for k in kept):
            kept.append(b)
    return kept


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def shape_direction(
    gx: np.ndarray,
    gy: np.ndarray,
    obstacle_mask: np.ndarray,
    bottlenecks: List[Bottleneck],
    resolution: float,
    origin_x: float,
    origin_y: float,
    approach: float = 1.0,
    exit: float = 0.6,
    lateral: float = 0.8,
    fade: float = 0.3,
    converge_angle: float = np.deg2rad(80.0),
    converge_dist: float = 0.15,
    goal_xy: Optional[Tuple[float, float]] = None,
    goal_clear: float = 0.6,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """Blend the unit field (gx, gy) into the line-following field near each bottleneck.

    Returns (fx, fy, oriented): the shaped unit direction and, per bottleneck
    the field actually crosses, (Bottleneck, sign), where sign orients the axis
    along the field's travel direction at the opening. A bottleneck where the
    field has no clear through-direction (|F . axis| < 0.5: the goal is in the
    doorway, or a cut locus) is left alone.

    Each zone covers only cells with a straight wall-free line to the opening
    centre, so it never reaches through a wall into the next room. It also
    fades out within goal_clear (+ fade) of goal_xy: a goal inside a zone
    would otherwise sit off the axis line the field holds the robot on, and
    the plan could never converge on it.
    """
    h, w = gx.shape
    fx, fy = gx.copy(), gy.copy()
    weight = np.zeros_like(gx)
    dx_acc = np.zeros_like(gx)
    dy_acc = np.zeros_like(gx)
    oriented = []
    rr, cc = np.mgrid[0:h, 0:w]
    X = origin_x + (cc + 0.5) * resolution
    Y = origin_y + (rr + 0.5) * resolution
    reach = max(approach, exit) + fade
    for b in bottlenecks:
        c0 = int((b.x - origin_x) / resolution)
        r0 = int((b.y - origin_y) / resolution)
        if not (0 <= r0 < h and 0 <= c0 < w):
            continue
        through = gx[r0, c0] * b.ax + gy[r0, c0] * b.ay
        if abs(through) < 0.5:
            continue
        sgn = 1.0 if through > 0 else -1.0
        ax, ay = sgn * b.ax, sgn * b.ay
        oriented.append((b, sgn))
        # Window around the opening.
        rad = int(np.ceil((np.hypot(reach, lateral + fade)) / resolution)) + 1
        ra, rb = max(0, r0 - rad), min(h, r0 + rad + 1)
        ca, cb = max(0, c0 - rad), min(w, c0 + rad + 1)
        px, py = X[ra:rb, ca:cb] - b.x, Y[ra:rb, ca:cb] - b.y
        s = px * ax + py * ay            # along the axis, + = towards the goal
        n = -px * ay + py * ax           # left of the axis
        # Longitudinal weight: 1 on [-approach, exit], fading over `fade` outside.
        ws = np.where(s < 0, _smoothstep((s + approach + fade) / fade), _smoothstep((exit + fade - s) / fade))
        wn = _smoothstep((lateral + fade - np.abs(n)) / fade)
        wb = ws * wn
        if goal_xy is not None:
            dg = np.hypot(X[ra:rb, ca:cb] - goal_xy[0], Y[ra:rb, ca:cb] - goal_xy[1])
            wb = wb * _smoothstep((dg - goal_clear) / fade)
        # Line of sight to the opening centre (no wall in between).
        win_obs = obstacle_mask[ra:rb, ca:cb]
        los = _line_of_sight(win_obs, r0 - ra, c0 - ca, wb > 0)
        wb = np.where(los & ~win_obs, wb, 0.0)
        psi = np.arctan2(ay, ax) - converge_angle * (2.0 / np.pi) * np.arctan(n / converge_dist)
        # Several zones can overlap (two doors facing each other); a weighted
        # average of their directions, then the blend with the base field.
        dx_acc[ra:rb, ca:cb] += wb * np.cos(psi)
        dy_acc[ra:rb, ca:cb] += wb * np.sin(psi)
        weight[ra:rb, ca:cb] += wb
    m = weight > 0
    wt = np.clip(weight, 0.0, 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ddx = np.where(m, dx_acc / np.where(m, weight, 1.0), 0.0)
        ddy = np.where(m, dy_acc / np.where(m, weight, 1.0), 0.0)
    bx = (1.0 - wt) * gx + wt * ddx
    by = (1.0 - wt) * gy + wt * ddy
    norm = np.hypot(bx, by)
    ok = m & (norm > 1e-6)
    fx[ok] = bx[ok] / norm[ok]
    fy[ok] = by[ok] / norm[ok]
    return fx, fy, oriented


def _line_of_sight(obs: np.ndarray, r0: int, c0: int, cand: np.ndarray) -> np.ndarray:
    """Cells in `cand` with a straight wall-free segment to (r0, c0)."""
    out = np.zeros_like(cand)
    rows, cols = np.nonzero(cand)
    if rows.size == 0:
        return out
    n_s = int(np.max(np.hypot(rows - r0, cols - c0))) + 2
    t = np.linspace(0.0, 1.0, n_s)[None, :]
    rs = np.rint(r0 + (rows[:, None] - r0) * t).astype(int)
    cs = np.rint(c0 + (cols[:, None] - c0) * t).astype(int)
    blocked = obs[rs, cs].any(axis=1)
    out[rows, cols] = ~blocked
    return out
