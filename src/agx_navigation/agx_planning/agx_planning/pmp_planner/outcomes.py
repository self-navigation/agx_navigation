"""Per-invocation solve results, separate from the numerical warm-start cache."""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class SolveDiagnostics:
    wall_s: float = 0.0
    warm_start: bool = False
    nodes: int = 0
    iterations: int = 0
    scipy_status: Optional[int] = None
    max_rms_residual: Optional[float] = None
    position_pin_error: Optional[float] = None
    heading_pin_error: Optional[float] = None
    wheel_pin_error: Optional[float] = None
    max_accel: Optional[float] = None


@dataclass(frozen=True)
class SolveOutcome:
    """Only a successful invocation carries its new full-horizon solution.

    The solution is the SciPy piecewise polynomial from this call. Persistent
    plan records must copy its knots and coefficients, not pickle this object.
    """

    success: bool
    status: str
    message: str
    solution: Any = None
    diagnostics: SolveDiagnostics = SolveDiagnostics()
    command: Optional[tuple[float, float]] = None
