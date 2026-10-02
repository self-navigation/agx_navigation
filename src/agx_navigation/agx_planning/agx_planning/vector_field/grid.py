from typing import Optional, Tuple
import numpy as np


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
    """

    def __init__(self):
        self._tt: Optional[np.ndarray] = None
        self._dT_dx: Optional[np.ndarray] = None
        self._dT_dy: Optional[np.ndarray] = None
        self._wd: Optional[np.ndarray] = None
        self._dwd_dx: Optional[np.ndarray] = None
        self._dwd_dy: Optional[np.ndarray] = None
        self._origin_x = 0.0
        self._origin_y = 0.0
        self._res = 1.0
        self._tt_max = 1.0
        self._field_eps = 1e-2
        self._ready = False
        self._version = 0

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def version(self) -> int:
        return self._version

    def update(
        self,
        T_field: np.ndarray,
        origin_x: float,
        origin_y: float,
        resolution: float,
        field_eps: float = 1e-2,
        wall_dist: Optional[np.ndarray] = None,
        wall_sigma: float = 0.05,
    ):
        T = T_field.astype(np.float64)
        finite_mask = np.isfinite(T)
        T_max_finite = float(T[finite_mask].max()) if finite_mask.any() else 1.0
        T_filled = np.where(finite_mask, T, T_max_finite)

        # np.gradient returns (dT/drow, dT/dcol); rows index y, cols index x.
        d_drow, d_dcol = np.gradient(T_filled, resolution, resolution)

        self._tt = T_filled
        self._dT_dx = d_dcol
        self._dT_dy = d_drow
        if wall_dist is not None:
            from scipy.ndimage import gaussian_filter
            wd = wall_dist.astype(np.float64)
            if wall_sigma > 0.0:
                wd = gaussian_filter(wd, wall_sigma / resolution, mode="nearest")
            w_drow, w_dcol = np.gradient(wd, resolution, resolution)
            self._wd, self._dwd_dx, self._dwd_dy = wd, w_dcol, w_drow
        else:
            self._wd = self._dwd_dx = self._dwd_dy = None
        self._origin_x = origin_x
        self._origin_y = origin_y
        self._res = resolution
        self._tt_max = T_max_finite
        self._field_eps = field_eps
        self._ready = True
        self._version += 1

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

        u = (px - self._origin_x) / self._res
        w = (py - self._origin_y) / self._res

        rows, cols = self._tt.shape
        in_bounds = (u >= 0) & (u <= cols - 1) & (w >= 0) & (w <= rows - 1)
        u_c = np.clip(u, 0.0, cols - 1.0001)
        w_c = np.clip(w, 0.0, rows - 1.0001)

        x0 = u_c.astype(int)
        y0 = w_c.astype(int)
        fx = u_c - x0
        fy = w_c - y0

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
        fux = -dx / norm
        fuy = -dy / norm

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
        u = (px - self._origin_x) / self._res
        w = (py - self._origin_y) / self._res
        rows, cols = self._wd.shape
        in_bounds = (u >= 0) & (u <= cols - 1) & (w >= 0) & (w <= rows - 1)
        u_c = np.clip(u, 0.0, cols - 1.0001)
        w_c = np.clip(w, 0.0, rows - 1.0001)
        x0 = u_c.astype(int)
        y0 = w_c.astype(int)
        fx = u_c - x0
        fy = w_c - y0

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
        u = (px - self._origin_x) / self._res
        w = (py - self._origin_y) / self._res
        rows, cols = self._tt.shape
        return (u >= 0) & (u <= cols - 1) & (w >= 0) & (w <= rows - 1)

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
