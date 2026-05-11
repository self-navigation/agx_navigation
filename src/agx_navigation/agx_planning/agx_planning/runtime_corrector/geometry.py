"""2-D path geometry helpers shared by the corrector and recovery strategies."""

import math


def project_onto_segment(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> tuple[float, float, float]:
    """Project (px, py) onto segment AB. Returns (proj_x, proj_y, t) with t in [0, 1]."""
    dx, dy = bx - ax, by - ay
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-12:
        return ax, ay, 0.0
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_len_sq))
    return ax + t * dx, ay + t * dy, t


def nearest_projection_on_path(
    rx: float,
    ry: float,
    path: list[tuple[float, float, float]],
) -> tuple[float, float, float, int, float]:
    """Find the nearest projection of (rx, ry) onto a path polyline.

    path -- ordered list of (x, y, theta) poses.

    Returns (proj_x, proj_y, proj_theta, seg_idx, t):
      proj_x/y   -- coordinates of the nearest point on the path
      proj_theta -- linearly interpolated heading at the projection
      seg_idx    -- index of the segment's start point in path
      t          -- fractional position within that segment [0, 1]
    """
    if not path:
        raise ValueError("path must not be empty")
    if len(path) == 1:
        return path[0][0], path[0][1], path[0][2], 0, 0.0

    best_dist = math.inf
    best: tuple[float, float, float, int, float] = (
        path[0][0],
        path[0][1],
        path[0][2],
        0,
        0.0,
    )

    for i in range(len(path) - 1):
        ax, ay, atheta = path[i]
        bx, by, btheta = path[i + 1]
        px, py, t = project_onto_segment(rx, ry, ax, ay, bx, by)
        dist = math.hypot(rx - px, ry - py)
        if dist < best_dist:
            best_dist = dist
            proj_theta = atheta + t * math.remainder(btheta - atheta, 2 * math.pi)
            best = (px, py, proj_theta, i, t)

    return best


def walk_ahead_on_path(
    path: list[tuple[float, float, float]],
    seg_idx: int,
    t: float,
    distance: float,
) -> tuple[float, float, float]:
    """Walk `distance` metres ahead along the path from position (seg_idx, t).

    Returns (x, y, theta) at the resulting point. Clamps to the path end when
    `distance` exceeds the remaining arclength.
    """
    if not path:
        raise ValueError("path must not be empty")
    if len(path) == 1:
        return path[0]

    remaining = distance

    for i in range(seg_idx, len(path) - 1):
        ax, ay, atheta = path[i]
        bx, by, btheta = path[i + 1]
        seg_len = math.hypot(bx - ax, by - ay)
        start_t = t if i == seg_idx else 0.0
        available = seg_len * (1.0 - start_t)
        if remaining <= available + 1e-9:
            frac = start_t + (remaining / seg_len if seg_len > 1e-12 else 0.0)
            frac = min(frac, 1.0)
            dtheta = math.remainder(btheta - atheta, 2 * math.pi)
            return (
                ax + frac * (bx - ax),
                ay + frac * (by - ay),
                atheta + frac * dtheta,
            )
        remaining -= available

    return path[-1]
