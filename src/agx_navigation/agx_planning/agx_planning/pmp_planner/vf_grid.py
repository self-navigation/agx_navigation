from typing import Optional, Tuple
import numpy as np
from scipy.ndimage import gaussian_filter


class VectorFieldGrid:
    """T(x, y), its gradient, and a unit-vector direction field F.

    Two field sources are kept because they serve different purposes:

    - dT/dx, dT/dy: recomputed from np.gradient so the position-costate
      ODEs (beta * grad T) stay consistent with the T grid being penalised,
      regardless of upstream smoothing or sign convention.
    - F_unit: derived from the upstream (Fx, Fy) channels and re-normalised
      with eps regularisation so |F_unit| -> 0 where the underlying field
      magnitude collapses (goal sink, saddles, flat regions), fading the
      alignment cost instead of fighting the terminal target.

    Sign convention: (Fx, Fy) is the "follow this direction" field. If the
    upstream publishes raw +grad T (away from goal), flip the sign upstream
    or set align_smooth_sigma > 0 to derive F from -grad(smooth(T)) here.

    Concurrency: instances are immutable after update(). The node uses
    atomic-reference-swap of self._field on field arrival (atomic under
    CPython's GIL on bare attribute assignment), so threaded readers
    never observe a torn update. Replan-trigger code keeps a reference
    to the previous instance for path-masked diffing against the new one.
    """

    def __init__(self):
        self._tt: Optional[np.ndarray] = None
        self._dT_dx: Optional[np.ndarray] = None
        self._dT_dy: Optional[np.ndarray] = None
        self._Fu_x: Optional[np.ndarray] = None
        self._Fu_y: Optional[np.ndarray] = None
        self._origin_x = 0.0
        self._origin_y = 0.0
        self._res = 1.0
        self._tt_max = 1.0
        self._ready = False
        # Monotonic counter; bumped on every update so the solver can
        # detect a replaced field and drop a now-stale warm start. With
        # atomic-swap of grid instances (offline mode), this counter
        # resets per-instance, so the node also calls reset_warm_start()
        # on every swap regardless of version.
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
        Fx_field: Optional[np.ndarray],
        Fy_field: Optional[np.ndarray],
        origin_x: float,
        origin_y: float,
        resolution: float,
        field_eps: float = 1e-2,
        align_smooth_sigma: float = 0.0,
    ):
        T = T_field.astype(np.float64)
        # FMM may produce inf for unreachable cells; replace with the largest
        # finite value before differentiation so the gradient stays finite.
        finite_mask = np.isfinite(T)
        if finite_mask.any():
            T_max_finite = float(T[finite_mask].max())
        else:
            T_max_finite = 1.0
        T_filled = np.where(finite_mask, T, T_max_finite)

        # np.gradient returns (dT/drow, dT/dcol). ROS map convention:
        # rows index y, cols index x. Position costates use the raw
        # (un-smoothed) gradient so the position penalty stays sharp.
        d_drow, d_dcol = np.gradient(T_filled, resolution, resolution)

        # Alignment direction field source priority:
        #   1. align_smooth_sigma > 0: derive from grad(gaussian_filter(T)).
        #   2. Upstream-provided (Fx, Fy): use as-is.
        #   3. Fallback: derive from -grad T (legacy 1-channel message).
        if align_smooth_sigma > 0.0:
            T_align = gaussian_filter(T_filled, sigma=align_smooth_sigma)
            d_drow_align, d_dcol_align = np.gradient(
                T_align,
                resolution,
                resolution,
            )
            Fx = -d_dcol_align
            Fy = -d_drow_align
        elif Fx_field is not None and Fy_field is not None:
            Fx = Fx_field.astype(np.float64)
            Fy = Fy_field.astype(np.float64)
        else:
            Fx = -d_dcol
            Fy = -d_drow
        # Smooth re-normalize: |F_unit| -> 1 for |F| >> eps and -> 0 for
        # |F| << eps. The latter fades the alignment cost in flat regions.
        norm = np.sqrt(Fx * Fx + Fy * Fy + field_eps * field_eps)
        Fu_x = Fx / norm
        Fu_y = Fy / norm

        self._tt = T_filled
        self._dT_dx = d_dcol
        self._dT_dy = d_drow
        self._Fu_x = Fu_x
        self._Fu_y = Fu_y
        self._origin_x = origin_x
        self._origin_y = origin_y
        self._res = resolution
        self._tt_max = T_max_finite
        self._ready = True
        self._version += 1

    def query_vec(
        self,
        px: np.ndarray,
        py: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized bilinear sample at world (px, py).

        Returns (T, dT/dx, dT/dy, F_unit_x, F_unit_y). World (origin_x,
        origin_y) is the corner of cell (0, 0); grid index = (world -
        origin) / resolution. Out-of-bounds: returns zero gradient and
        zero direction (no force) and the max-T sentinel (still penalised).
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
        fux = bilerp(self._Fu_x)
        fuy = bilerp(self._Fu_y)

        T = np.where(in_bounds, T, self._tt_max)
        dx = np.where(in_bounds, dx, 0.0)
        dy = np.where(in_bounds, dy, 0.0)
        fux = np.where(in_bounds, fux, 0.0)
        fuy = np.where(in_bounds, fuy, 0.0)
        return T, dx, dy, fux, fuy

    def query_scalar(
        self,
        px: float,
        py: float,
    ) -> Tuple[float, float, float, float, float]:
        T, dx, dy, fux, fuy = self.query_vec(np.array([px]), np.array([py]))
        return float(T[0]), float(dx[0]), float(dy[0]), float(fux[0]), float(fuy[0])

    def in_bounds(self, px: np.ndarray, py: np.ndarray) -> np.ndarray:
        """Boolean mask: which (px, py) lie within the grid extent.

        Used by the offline-mode path-masked field-diff replan trigger:
        cells out-of-bounds in either old-or-new grid are tagged as
        infinite diff so newly-discovered terrain on the planned path
        always triggers a replan.
        """
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
        """Trace dp/ds = F_unit(p) starting at (x0, y0) for up to length_m.

        Stops on (a) |F| < 1e-3 (goal sink or saddle), (b) length_m of arc
        length consumed, (c) leaving the grid bounds, or (d) within
        sqrt(0.01) m of goal_xy. Early goal-stop means p_pursuit naturally
        collapses to p_goal when the streamline reaches it within the
        lookahead distance.

        Returns (ref_pts (N, 2), n_perp (N, 2)) where n_perp is the unit
        normal rotated 90 deg CCW from F at each sample (used for the
        cross-track residual). Both arrays are empty if the trace cannot
        start (e.g. already at goal, F collapses immediately).
        """
        if not self._ready:
            return np.zeros((0, 2)), np.zeros((0, 2))
        if ds is None:
            ds = self._res

        # If we're already inside the goal stop ball, the loop below would
        # record exactly one ref point (chassis) before tripping the stop
        # check and breaking. The caller (solve) then sets p_pursuit =
        # chassis position, which makes the BVP terminal cost pull toward
        # "stay where you are" -- chassis never moves. Return empty
        # instead so the caller's fallback sets p_pursuit = goal directly.
        # (This fixes a long-standing online-mode bug where the chassis
        # would idle in the [goal_tolerance_xy, 0.1m] ring around the goal
        # until TF noise pushed it inside; offline mode exposed it because
        # there's no observation noise to bail us out.)
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
            px = px + ds * tx
            py = py + ds * ty
        if n == 0:
            return np.zeros((0, 2)), np.zeros((0, 2))
        ref_pts = np.column_stack([ref_x[:n], ref_y[:n]])
        n_perp = np.column_stack([nx[:n], ny[:n]])
        return ref_pts, n_perp
