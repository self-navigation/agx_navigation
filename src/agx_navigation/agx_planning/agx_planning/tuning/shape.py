"""Shape descriptors and an interest score for candidate trajectories.

WHY THIS EXISTS
---------------
The 100-plan library came from uniform random start/goal pairs, and it shows:
the gallery's first page has every interesting shape in it and pages 3-5 are
straight lines. So the evaluation set is capped by what the library happens to
contain, and every per-shape claim this project has made rests on exactly ONE
plan of that shape -- the U-turn notch is 5906 rollouts of `floor_6_00031` and
nothing else. We cannot tell a property of U-turns from a property of that
U-turn.

The fix is to generate candidates deliberately and screen them BEFORE paying for
a PMP solve. This module is the screen: pure geometry over a polyline, no ROS,
no Gazebo, no scipy. Same rule as the other `tuning/` modules.

THE PIVOT IS THE WHOLE DIFFICULTY
---------------------------------
A PMP plan usually opens by spinning in place to face the path: a large heading
change over ~no distance. Every descriptor here is built from heading deltas, so
an untrimmed pivot dominates all of them -- which is exactly why
`tools/classify_plans.py` labels 58 of 100 plans CORNER when the gallery shows
most are visually straight. `trim_pivot` is therefore not a refinement, it is
the difference between a descriptor that measures the path and one that measures
the pirouette. Everything public here trims by default.

Note the deliberate asymmetry with the pivot as a SELECTION CRITERION: we want
plans that *require* a pivot (that is the realistic case -- a real robot rarely
starts already facing its route), while measuring the shape of what happens
after it. `pivot_demand` scores the former; `descriptors` ignores it.

WHAT "INTERESTING" MEANS HERE
-----------------------------
Not "hard to drive" -- we cannot know that without driving it. It means "the
planner had a non-trivial problem to solve", which is a property of the
start/goal pair and is checkable from the map:

  * `blocked`      the straight line start->goal hits an obstacle, so the plan
                   must route around something rather than drive at the goal;
  * `detour`       route length / straight-line distance -- how far around;
  * turning, sign changes, net-vs-gross turn -- the shape descriptors, which
                   separate a corner from an S from a U-turn;
  * `pivot_demand` how much in-place rotation the start and goal headings force.

These are combined by `interest_score` only as a RANKING, never as a label. The
labels remain a coarse bucketing to be confirmed against the gallery by eye --
the automatic ones have misled once already and the picture is the authority.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as _field
from typing import Iterable, Sequence

Point = tuple[float, float]

# Below this, a heading delta is noise rather than steering. Shared by the
# sign-change counter and the turn totals so they cannot disagree about what
# counts as a turn.
TURN_EPS = 0.03


def resample(pts: Sequence[Point], step: float = 0.15) -> list[Point]:
    """Arc-length resample, so curvature is not dominated by sample density.

    A PMP rollout and an FM2 streamline sample at completely different rates; we
    compare their shapes, so both must be put on the same footing first.
    """
    pts = [(float(x), float(y)) for x, y in pts]
    if len(pts) < 2:
        return pts
    out = [pts[0]]
    acc = 0.0
    for i in range(1, len(pts)):
        acc += math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
        if acc >= step:
            out.append(pts[i])
            acc = 0.0
    if out[-1] != pts[-1]:
        out.append(pts[-1])
    return out


# Where the PMP opening pivot has finished, MEASURED rather than guessed
# (2026-08-14). Real plans turn ~2.8 rad within their first ~0.7 m of travel --
# a very tight arc, not a pure spin, so arc-length resampling does not remove it
# on its own. Sweeping this threshold over the 100-plan library moves the label
# counts steadily until 0.7 and then not at all (STRAIGHT/CORNER: 41/25 at 0.30,
# 64/16 at 0.70, 65/16 at 2.00) -- a plateau, so 0.7 is the pivot's real extent
# and not a knob tuned to a wanted answer. The 64 STRAIGHT also matches what the
# gallery actually shows, which 25 CORNER did not.
#
# `tools/plot_trajectory_gallery.py` keeps its own 0.30: it is display-only and
# its figures are committed, so changing it would silently redraw published
# pictures. This module is what selection should use.
PIVOT_TRAVEL_M = 0.70


def trim_pivot(pts: Sequence[Point], min_travel: float = PIVOT_TRAVEL_M) -> list[Point]:
    """Drop the leading in-place reorientation. See the module docstring.

    Returns the tail beginning at the last sample before `min_travel` of travel
    has accumulated, so the path still starts where the robot did.
    """
    pts = [(float(x), float(y)) for x, y in pts]
    acc = 0.0
    for i in range(1, len(pts)):
        acc += math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
        if acc >= min_travel:
            return pts[i - 1:]
    return pts


def _headings(pts: Sequence[Point]) -> list[float]:
    return [math.atan2(pts[i + 1][1] - pts[i][1], pts[i + 1][0] - pts[i][0])
            for i in range(len(pts) - 1)]


def _wrap(d: float) -> float:
    return (d + math.pi) % (2 * math.pi) - math.pi


def _turns(headings: Sequence[float]) -> list[float]:
    return [_wrap(headings[i] - headings[i - 1]) for i in range(1, len(headings))]


@dataclass(frozen=True)
class Descriptors:
    """Geometry of one path, after pivot trimming and arc-length resampling."""

    length: float          # path length [m]
    net: float             # straight-line start->end displacement [m]
    straightness: float    # net / length; 1.0 is a straight line
    total_abs_turn: float  # sum |dtheta| -- how much steering happens [rad]
    net_turn: float        # signed sum -- a corner has net ~ total, an S ~ 0
    max_turn: float        # largest single-step delta; flags degenerate plans
    sign_changes: int      # direction reversals in the smoothed turn signal

    @property
    def turn_per_metre(self) -> float:
        return self.total_abs_turn / self.length if self.length > 0 else math.nan


def descriptors(pts: Sequence[Point], *, trim: bool = True,
                step: float = 0.15) -> Descriptors | None:
    """Shape descriptors for a polyline. None if it is too short to describe.

    Numerically compatible with `tools/classify_plans.py` when `trim=False`;
    the default trims, which is what makes the labels match the gallery.
    """
    work = trim_pivot(pts) if trim else [(float(x), float(y)) for x, y in pts]
    work = resample(work, step)
    if len(work) < 4:
        return None
    length = sum(math.hypot(work[i][0] - work[i - 1][0], work[i][1] - work[i - 1][1])
                 for i in range(1, len(work)))
    net = math.hypot(work[-1][0] - work[0][0], work[-1][1] - work[0][1])
    turns = _turns(_headings(work))
    if not turns:
        return None

    # Smooth before counting sign changes: a raw per-sample turn signal flickers
    # around zero on a straight run and would report dozens of "reversals".
    w = 3
    smoothed = []
    for i in range(len(turns)):
        window = turns[max(0, i - w): i + w + 1]
        smoothed.append(sum(window) / len(window))
    sign_changes, prev = 0, 0
    for t in smoothed:
        s = 1 if t > TURN_EPS else (-1 if t < -TURN_EPS else 0)
        if s:
            if prev and s != prev:
                sign_changes += 1
            prev = s

    return Descriptors(
        length=length,
        net=net,
        straightness=net / length if length > 0 else math.nan,
        total_abs_turn=sum(abs(t) for t in turns),
        net_turn=sum(turns),
        max_turn=max(abs(t) for t in turns),
        sign_changes=sign_changes,
    )


def label(d: Descriptors) -> str:
    """Coarse shape bucket. A RANKING AID, NOT A VERDICT -- confirm by eye.

    Ordered most-specific first: a U-turn is also a corner by any looser test,
    so it must be caught before one.
    """
    if d.straightness > 0.985 and d.total_abs_turn < 0.6:
        return "STRAIGHT"
    if d.sign_changes >= 3:
        return "ZIGZAG"
    if d.sign_changes >= 1 and abs(d.net_turn) < 0.5 * d.total_abs_turn:
        return "S"
    if abs(d.net_turn) > 2.4:          # ~140 deg of net rotation
        return "UTURN"
    if d.total_abs_turn > 5.0 and d.straightness < 0.5:
        return "LOOP"
    if d.total_abs_turn > 0.6:
        return "CORNER"
    return "STRAIGHT"


def pivot_demand(start_theta: float, goal_theta: float,
                 pts: Sequence[Point]) -> tuple[float, float]:
    """In-place rotation the start and goal headings force, in radians.

    The realistic case for a real deployment: the robot is parked facing
    whatever it was facing, and must end facing whatever the dock/goal requires.
    Neither is likely to be the path's own direction. Returns
    `(start_pivot, goal_pivot)`, each in [0, pi].

    Computed against the path's initial and final travel directions, taken over
    a short baseline so a single noisy sample cannot set them.
    """
    pts = [(float(x), float(y)) for x, y in pts]
    if len(pts) < 2:
        return 0.0, 0.0
    entry = trim_pivot(pts)
    if len(entry) < 2:
        entry = pts
    path_in = math.atan2(entry[1][1] - entry[0][1], entry[1][0] - entry[0][0])
    path_out = math.atan2(pts[-1][1] - pts[-2][1], pts[-1][0] - pts[-2][0])
    return abs(_wrap(path_in - start_theta)), abs(_wrap(goal_theta - path_out))


def line_blocked(start: Point, goal: Point, occupied, origin: Point,
                 resolution: float) -> bool:
    """Does the straight line start->goal cross an occupied cell?

    `occupied` is a (height, width) boolean array in the OccupancyGrid's
    row-major layout (row 0 is y_min), matching `random_goals.reachable_mask`.

    This is the user's core screening constraint: if the direct line is clear,
    the planner's job is "drive at the goal" and the plan carries no shape
    regardless of how far apart the endpoints are. Distance was the old filter
    and it is why the library is mostly straight lines -- a long straight
    corridor passes a distance test perfectly.

    Supercover traversal (every cell the segment touches, not the thin
    Bresenham line), because a segment clipping the CORNER of a wall must count
    as blocked -- that is exactly the case a plan has to route around.
    """
    h = len(occupied)
    w = len(occupied[0]) if h else 0
    if h == 0 or w == 0:
        return False

    def to_cell(p: Point) -> tuple[int, int]:
        return (int((p[0] - origin[0]) / resolution),
                int((p[1] - origin[1]) / resolution))

    x0, y0 = to_cell(start)
    x1, y1 = to_cell(goal)
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    x, y = x0, y0
    err = dx - dy
    n = dx + dy
    for _ in range(n + 1):
        if 0 <= y < h and 0 <= x < w:
            if occupied[y][x]:
                return True
        else:
            # Off-map counts as blocked: the baked map marks everything outside
            # the building envelope unknown, and a line leaving it is not a
            # drivable shortcut.
            return True
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        # Step ONE axis per iteration (not both, as plain Bresenham does on a
        # diagonal) so no wall corner is squeezed through diagonally.
        if e2 > -dy:
            err -= dy
            x += sx
        elif e2 < dx:
            err += dx
            y += sy
    return False


@dataclass(frozen=True)
class Candidate:
    """A screened start/goal pair, before any PMP solve has been paid for."""

    start: Point
    goal: Point
    start_theta: float
    goal_theta: float
    blocked: bool
    detour: float
    desc: Descriptors
    start_pivot: float
    goal_pivot: float
    shape: str = _field(default="")

    @property
    def score(self) -> float:
        return interest_score(self)


def interest_score(c: Candidate) -> float:
    """Rank candidates by how much of a problem the planner was given.

    Deliberately a sum of bounded terms rather than a product: a product lets
    any single zero term veto a candidate, and we want e.g. a genuinely tortuous
    route with a clear line of sight to still rank well. Each term is clipped so
    no single one can run away and dominate the ranking -- this orders
    candidates, it does not measure anything physical.
    """
    d = c.desc
    return (
        (1.0 if c.blocked else 0.0)                     # the primary constraint
        + min(c.detour - 1.0, 1.5)                      # how far around
        + min(d.total_abs_turn / 3.0, 2.0)              # how much steering
        + min(d.sign_changes * 0.4, 1.6)                # reversals: S / zigzag
        + min((c.start_pivot + c.goal_pivot) / math.pi, 1.0)
    )


def stratify(candidates: Iterable[Candidate], per_shape: int) -> list[Candidate]:
    """Take the top `per_shape` of each shape bucket, best-scoring first.

    The point of the whole exercise: a set with several examples per category,
    so a per-shape claim stops resting on one trajectory. Buckets that cannot
    fill are returned short rather than padded from another bucket -- a set that
    silently substitutes three corners for a missing U-turn is exactly the
    failure this replaces.
    """
    by_shape: dict[str, list[Candidate]] = {}
    for c in candidates:
        by_shape.setdefault(c.shape or label(c.desc), []).append(c)
    out: list[Candidate] = []
    for shape in sorted(by_shape):
        picks = sorted(by_shape[shape], key=lambda c: -c.score)[:per_shape]
        out.extend(picks)
    return out
