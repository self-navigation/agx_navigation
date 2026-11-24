from typing import Optional, Tuple
import copy
import hashlib
import json
import numpy as np

from .preparation import prepare_travel_time


class VectorFieldGrid:
    """T(x, y), its gradient, and a derived unit-vector direction field.

    The direction field F_unit = -normalize(grad T, eps) is computed
    on the fly from the stored gradient during query, so only three
    arrays are stored: T, dT/dx, dT/dy.

    field_eps controls how fast the alignment cost fades near the goal
    and at cut loci: |F_unit| -> 1 where |grad T| >> eps, -> 0 where
    |grad T| << eps (goal sink, saddles, flat regions).

    Sign convention: F points in the direction of descending T (toward goal).

    Optionally also holds a signed wall distance d(x, y) (>0 free, <0 in
    walls), Gaussian-smoothed by wall_sigma so the footprint cost built on
    it is close to C^1 for solve_bvp; query_dist samples it.

    Optionally holds a separate unit DIRECTION field (fx, fy) that replaces
    -normalize(grad T) as F_unit, e.g. the bottleneck-aligned field
    (bottleneck.py). T and its gradient (the position pull) are unchanged,
    and F_unit keeps the eps fade of |grad T|, so it still vanishes at the goal
    sink and at cut loci.
    """

    def __init__(self):
        self._tt: Optional[np.ndarray] = None
        self._dT_dx: Optional[np.ndarray] = None
        self._dT_dy: Optional[np.ndarray] = None
        self._wd: Optional[np.ndarray] = None
        self._dwd_dx: Optional[np.ndarray] = None
        self._dwd_dy: Optional[np.ndarray] = None
        self._fx: Optional[np.ndarray] = None
        self._fy: Optional[np.ndarray] = None
        self._origin_x = 0.0
        self._origin_y = 0.0
        self._res = 1.0
        self._tt_max = 1.0
        self._field_eps = 1e-2
        self._ready = False
        self._version = 0
        self._identity = None
        self._field_key = None
        self._frozen = False

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def version(self) -> int:
        return self._version

    @property
    def identity(self):
        """Identity of the actual query data, settings, and known field key."""
        return self._identity

    @property
    def field_key(self):
        return self._field_key

    @property
    def identified(self) -> bool:
        return self._field_key is not None

    def require_goal(self, goal):
        """Reject a goal whose source cell differs from an identified field."""
        goal = np.asarray(goal, dtype=float)
        if goal.shape != (3,) or not np.isfinite(goal).all():
            raise ValueError("goal must contain three finite values")
        if self._field_key is not None:
            cell = self._field_key.geometry.world_to_cell(goal[0], goal[1])
            if cell != self._field_key.goal_cell:
                raise ValueError("goal does not match the field source cell")

    def frozen_copy(self):
        """Own immutable buffers for one rollout, independent of later updates."""
        if self._frozen:
            return self
        result = copy.copy(self)
        for name, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                packed = np.ascontiguousarray(value)
                result.__dict__[name] = np.frombuffer(packed.tobytes(), dtype=packed.dtype).reshape(packed.shape)
        result._frozen = True
        return result

    def update(
        self,
        T_field: np.ndarray,
        origin_x: float,
        origin_y: float,
        resolution: float,
        field_eps: float = 1e-2,
        wall_dist: Optional[np.ndarray] = None,
        wall_sigma: float = 0.05,
        direction: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        field_key=None,
    ):
        if self._frozen:
            raise ValueError("a frozen field cannot be updated")
        T = np.asarray(T_field, dtype=np.float64)
        if T.ndim != 2 or min(T.shape) < 2:
            raise ValueError("travel time requires a two-dimensional grid of at least 2x2")
        if (not np.isfinite([origin_x, origin_y, resolution, field_eps, wall_sigma]).all()
                or resolution <= 0 or field_eps <= 0 or wall_sigma < 0):
            raise ValueError("invalid field geometry or query settings")
        if field_key is not None:
            geometry = field_key.geometry
            if (geometry.shape != T.shape or geometry.resolution != resolution
                    or geometry.origin != (origin_x, origin_y, 0.0)):
                raise ValueError("field data does not match its key geometry")
        T_filled = prepare_travel_time(T)
        d_drow, d_dcol = np.gradient(T_filled, resolution, resolution)
        wd = w_dcol = w_drow = None
        if wall_dist is not None:
            from scipy.ndimage import gaussian_filter
            wd = np.array(wall_dist, dtype=np.float64, copy=True)
            if wd.shape != T.shape or not np.isfinite(wd).all():
                raise ValueError("wall distance must be finite and match travel time")
            if wall_sigma > 0.0:
                wd = gaussian_filter(wd, wall_sigma / resolution, mode="nearest")
            w_drow, w_dcol = np.gradient(wd, resolution, resolution)
        fx = fy = None
        if direction is not None:
            fx, fy = (np.array(value, dtype=np.float64, copy=True) for value in direction)
            if any(value.shape != T.shape or not np.isfinite(value).all() for value in (fx, fy)):
                raise ValueError("directions must be finite and match travel time")

        # A content identity also protects legacy fields, whose packed transport
        # does not contain source or map-revision metadata. It cannot invent them.
        metadata = dict(origin=[float(origin_x), float(origin_y)], resolution=float(resolution),
                        field_eps=float(field_eps), wall_sigma=float(wall_sigma),
                        convention="cell-centres-v1", key=None if field_key is None else field_key.to_dict())
        digest = hashlib.sha256(json.dumps(metadata, sort_keys=True, allow_nan=False).encode())
        for value in (T_filled, wd, fx, fy):
            if value is None:
                digest.update(b"none")
            else:
                digest.update(str((value.shape, value.dtype.str)).encode())
                digest.update(np.ascontiguousarray(value).tobytes())

        self._tt, self._dT_dx, self._dT_dy = T_filled, d_dcol, d_drow
        self._wd, self._dwd_dx, self._dwd_dy = wd, w_dcol, w_drow
        self._fx, self._fy = fx, fy
        self._origin_x, self._origin_y, self._res = origin_x, origin_y, resolution
        self._tt_max = float(T_filled.max())
        self._field_eps = field_eps
        self._field_key = field_key
        self._identity = digest.hexdigest()
        self._ready = True
        self._version += 1

    def _interpolation_coordinates(self, px, py, shape):
        # Occupancy geometry uses cell corners; sampled arrays use cell centres.
        # The outer half-cell band holds the nearest centre value. Physical map
        # bounds remain half-open, including the full first and last cells.
        rows, cols = shape
        wx, wy = np.asarray(px), np.asarray(py)
        x = (wx - self._origin_x) / self._res
        y = (wy - self._origin_y) / self._res
        inside = ((wx >= self._origin_x) & (wx < self._origin_x + cols * self._res)
                  & (wy >= self._origin_y) & (wy < self._origin_y + rows * self._res)
                  & (x >= 0) & (x < cols) & (y >= 0) & (y < rows))
        # Invalid positions use the same outside result without producing an
        # invalid array index before the result mask is applied.
        u = np.clip(np.where(np.isfinite(x), x - 0.5, 0.0), 0.0, cols - 1)
        v = np.clip(np.where(np.isfinite(y), y - 0.5, 0.0), 0.0, rows - 1)
        x0 = np.minimum(u.astype(int), cols - 2)
        y0 = np.minimum(v.astype(int), rows - 2)
        return inside, x0, y0, u - x0, v - y0

    def query_vec(
        self,
        px: np.ndarray,
        py: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized bilinear sample at world (px, py).

        Returns (T, dT/dx, dT/dy, F_unit_x, F_unit_y).
        F_unit = -normalize(grad T, eps): fades to zero where the field
        collapses (goal sink, cut loci) instead of producing noise.
        Out-of-bounds: zero gradient/direction, max-T sentinel.
        """
        if not self._ready:
            zero = np.zeros_like(px)
            return zero.copy(), zero.copy(), zero.copy(), zero.copy(), zero.copy()

        in_bounds, x0, y0, fx, fy = self._interpolation_coordinates(px, py, self._tt.shape)

        def bilerp(arr: np.ndarray) -> np.ndarray:
            return (
                arr[y0, x0] * (1.0 - fx) * (1.0 - fy)
                + arr[y0, x0 + 1] * fx * (1.0 - fy)
                + arr[y0 + 1, x0] * (1.0 - fx) * fy
                + arr[y0 + 1, x0 + 1] * fx * fy
            )

        T = bilerp(self._tt)
        dx = bilerp(self._dT_dx)
        dy = bilerp(self._dT_dy)

        # Unit direction: descend T. Eps regularisation fades alignment
        # cost near the goal and at cut loci rather than fighting noise.
        eps = self._field_eps
        norm = np.sqrt(dx * dx + dy * dy + eps * eps)
        if self._fx is None:
            fux = -dx / norm
            fuy = -dy / norm
        else:
            # Shaped direction, renormalised after interpolation (it turns
            # sharply on a bottleneck's axis line), times the same eps fade.
            fade = np.sqrt(dx * dx + dy * dy) / norm
            sx, sy = bilerp(self._fx), bilerp(self._fy)
            sn = np.hypot(sx, sy)
            sn = np.where(sn > 1e-9, sn, 1.0)
            fux = sx / sn * fade
            fuy = sy / sn * fade

        T = np.where(in_bounds, T, self._tt_max)
        dx = np.where(in_bounds, dx, 0.0)
        dy = np.where(in_bounds, dy, 0.0)
        fux = np.where(in_bounds, fux, 0.0)
        fuy = np.where(in_bounds, fuy, 0.0)
        return T, dx, dy, fux, fuy

    @property
    def has_wall_dist(self) -> bool:
        return self._wd is not None

    def query_dist(
        self,
        px: np.ndarray,
        py: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bilinear sample of the signed wall distance and its gradient.

        Returns (d, dd/dx, dd/dy). Out of bounds counts as deep inside a
        wall with zero gradient, so it costs but does not steer.
        """
        if self._wd is None:
            raise ValueError("field has no wall-distance channel")
        in_bounds, x0, y0, fx, fy = self._interpolation_coordinates(px, py, self._wd.shape)

        def bilerp(arr: np.ndarray) -> np.ndarray:
            return (
                arr[y0, x0] * (1.0 - fx) * (1.0 - fy)
                + arr[y0, x0 + 1] * fx * (1.0 - fy)
                + arr[y0 + 1, x0] * (1.0 - fx) * fy
                + arr[y0 + 1, x0 + 1] * fx * fy
            )

        d = np.where(in_bounds, bilerp(self._wd), -1.0)
        gx = np.where(in_bounds, bilerp(self._dwd_dx), 0.0)
        gy = np.where(in_bounds, bilerp(self._dwd_dy), 0.0)
        return d, gx, gy

    def query_scalar(
        self,
        px: float,
        py: float,
    ) -> Tuple[float, float, float, float, float]:
        T, dx, dy, fux, fuy = self.query_vec(np.array([px]), np.array([py]))
        return float(T[0]), float(dx[0]), float(dy[0]), float(fux[0]), float(fuy[0])

    def in_bounds(self, px: np.ndarray, py: np.ndarray) -> np.ndarray:
        if not self._ready:
            return np.zeros_like(px, dtype=bool)
        inside, _, _, _, _ = self._interpolation_coordinates(px, py, self._tt.shape)
        return inside

    def trace_streamline(
        self,
        x0: float,
        y0: float,
        length_m: float,
        ds: Optional[float] = None,
        goal_xy: Optional[Tuple[float, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Trace dp/ds = F_unit(p) starting at (x0, y0) for up to length_m."""
        if not self._ready:
            return np.zeros((0, 2)), np.zeros((0, 2))
        if ds is None:
            ds = self._res

        if goal_xy is not None:
            if (x0 - goal_xy[0]) ** 2 + (y0 - goal_xy[1]) ** 2 < 1e-2:
                return np.zeros((0, 2)), np.zeros((0, 2))

        n_max = max(8, int(np.ceil(length_m / ds)) + 1)
        ref_x = np.empty(n_max, dtype=np.float64)
        ref_y = np.empty(n_max, dtype=np.float64)
        nx = np.empty(n_max, dtype=np.float64)
        ny = np.empty(n_max, dtype=np.float64)

        px, py = float(x0), float(y0)
        n = 0
        for _ in range(n_max):
            _, _, _, fux, fuy = self.query_scalar(px, py)
            mag = float(np.hypot(fux, fuy))
            if mag < 1e-3:
                break
            tx, ty = fux / mag, fuy / mag
            ref_x[n] = px
            ref_y[n] = py
            nx[n] = -ty
            ny[n] = tx
            n += 1
            if goal_xy is not None:
                if (px - goal_xy[0]) ** 2 + (py - goal_xy[1]) ** 2 < 1e-2:
                    break
            px += ds * tx
            py += ds * ty

        if n == 0:
            return np.zeros((0, 2)), np.zeros((0, 2))
        return np.column_stack([ref_x[:n], ref_y[:n]]), np.column_stack(
            [nx[:n], ny[:n]]
        )
