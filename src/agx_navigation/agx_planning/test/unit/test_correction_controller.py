"""Unit tests for CorrectionController and recovery strategies.

All tests are pure Python — no ROS2 required.  Run with:
    PYTHONPATH=<repo>/src/agx_navigation/agx_planning pytest test/
"""

import math
import pytest

from agx_planning.runtime_corrector.strategies import (
    ExitKind,
    RecoveryConfig,
    TwistOutcome,
    ExitOutcome,
    ResumeOutcome,
    WaitOutcome,
    default_strategies,
)
from agx_planning.runtime_corrector.config import CorrectorConfig
from agx_planning.runtime_corrector.correction_controller import (
    CorrectionController,
    CorrectionResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg() -> CorrectorConfig:
    return CorrectorConfig(
        corridor_epsilon=0.3,
        recovery_corridor_epsilon=0.1,
        recovery_angle_tolerance=0.1,
        recovery_look_ahead=0.5,
        recovery_v_max=0.3,
        recovery_omega_max=1.0,
        recovery_K_v=1.0,
        recovery_K_bearing=2.0,
        recovery_K_theta=2.0,
    )


@pytest.fixture
def rcfg(cfg) -> RecoveryConfig:
    return RecoveryConfig(
        recovery_corridor_epsilon=cfg.recovery_corridor_epsilon,
        recovery_angle_tolerance=cfg.recovery_angle_tolerance,
        look_ahead_distance=cfg.recovery_look_ahead,
        v_max=cfg.recovery_v_max,
        omega_max=cfg.recovery_omega_max,
        K_v=cfg.recovery_K_v,
        K_bearing=cfg.recovery_K_bearing,
        K_theta=cfg.recovery_K_theta,
        max_pursuit_bearing_err=cfg.recovery_max_pursuit_bearing_err,
        near_endpoint_distance=cfg.recovery_near_endpoint_distance,
    )


@pytest.fixture
def controller(cfg, rcfg) -> CorrectionController:
    return CorrectionController(cfg, default_strategies(rcfg))


def straight_path(n: int = 5, length: float = 2.0) -> list:
    """Path going east (+x), evenly spaced."""
    step = length / (n - 1)
    return [(i * step, 0.0, 0.0) for i in range(n)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_endpoint_proximity(controller):
    """Robot within recovery_corridor_epsilon of path end → ENDPOINT exit."""
    path = straight_path()
    fx, fy, _ = path[-1]
    pose = (fx + 0.05, fy, 0.0)  # 0.05 m from end, within epsilon=0.1
    result = controller.step(pose, path, result_received=True)
    assert result.should_exit
    assert result.exit_kind == ExitKind.ENDPOINT


def test_endpoint_proximity_blocked_without_result(controller):
    """Endpoint proximity does NOT fire when result has not arrived."""
    path = straight_path()
    fx, fy, _ = path[-1]
    pose = (fx + 0.05, fy, 0.0)
    result = controller.step(pose, path, result_received=False)
    assert not result.should_exit


def test_overshot_wait(controller):
    """Robot past path end, no result yet → hold position."""
    path = straight_path()
    fx, fy, ftheta = path[-1]
    # Place robot clearly past the end along final heading
    pose = (fx + 0.5, fy, ftheta)
    result = controller.step(pose, path, result_received=False)
    assert result.waiting_for_chunks
    assert not result.should_exit


def test_overshot_align(controller):
    """Robot past end, result in, heading off → rotate in place."""
    path = straight_path()
    fx, fy, ftheta = path[-1]
    pose = (fx + 0.5, fy, ftheta + 0.5)  # 0.5 rad off, > tolerance=0.1
    result = controller.step(pose, path, result_received=True)
    assert not result.should_exit
    assert not result.waiting_for_chunks
    assert result.twist is not None
    v, omega = result.twist
    assert v == pytest.approx(0.0)
    assert omega != pytest.approx(0.0)


def test_overshot_exit(controller):
    """Robot past end, result in, heading aligned → OVERSHOT exit."""
    path = straight_path()
    fx, fy, ftheta = path[-1]
    pose = (fx + 0.5, fy, ftheta)  # heading matches exactly
    result = controller.step(pose, path, result_received=True)
    assert result.should_exit
    assert result.exit_kind == ExitKind.OVERSHOT


def test_recovered(controller):
    """Robot back in corridor (spatial + heading) → ResumeOutcome / snap_index set."""
    path = straight_path()
    # Place robot on the path midway, heading aligned
    mx, my, mtheta = path[2]
    pose = (mx, my + 0.02, mtheta)  # 0.02 m off — within epsilon=0.1
    result = controller.step(pose, path, result_received=False)
    assert result.snap_index is not None
    assert result.exit_kind == ExitKind.RECOVERED
    assert not result.should_exit


def test_rotate_in_place(controller):
    """Robot spatially close but heading off → RotateInPlaceStrategy."""
    path = straight_path()
    mx, my, _ = path[2]
    # Within corridor but 90° heading error
    pose = (mx, my + 0.05, math.pi / 2)
    result = controller.step(pose, path, result_received=False)
    assert result.twist is not None
    v, omega = result.twist
    assert v == pytest.approx(0.0)
    assert omega != pytest.approx(0.0)
    assert "RotateInPlace" in result.correction_strat


def test_lookahead_pursuit(controller):
    """Robot offset laterally with carrot ahead → LookAheadPursuitStrategy (v > 0)."""
    path = straight_path(n=10, length=4.0)
    # Robot starts well to the side of the midpoint, heading east
    pose = (1.0, 0.8, 0.0)  # 0.8 m lateral offset, well outside epsilon=0.1
    result = controller.step(pose, path, result_received=False)
    assert result.twist is not None
    v, _ = result.twist
    assert v > 0.0
    assert "LookAheadPursuit" in result.correction_strat


def test_near_endpoint_rotate(controller):
    """Robot offset near end with carrot orthogonal → NearEndpointStrategy (v == 0)."""
    path = straight_path(n=5, length=1.0)
    fx, fy, ftheta = path[-1]
    # Place robot beside the endpoint, facing away so carrot bearing > π/2
    pose = (fx - 0.05, fy + 0.2, math.pi)  # facing west, endpoint is east
    result = controller.step(pose, path, result_received=False)
    assert result.twist is not None
    v, omega = result.twist
    assert v == pytest.approx(0.0)
    assert "NearEndpoint" in result.correction_strat


def test_no_strategy_fallback(controller):
    """Empty path → no strategy fires, fallback result has no twist."""
    result = controller.step((0.0, 0.0, 0.0), [], result_received=False)
    assert result.twist is None
    assert not result.should_exit
