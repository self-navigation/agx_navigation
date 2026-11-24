"""Deterministic complete-plan decisions without ROS or command publication.

The first implementation uses declared raw segment boundaries and static maps.
Callers provide motion/error bounds for the chosen execution model and an
explicit solver/limits result. No geometric claim is inferred from a field key.
The caller must schedule the returned brake deadline and refresh this assessment
after a changed map, execution epoch, or state assumption.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import operator

import numpy as np

from agx_planning.maps import MapSnapshot
from agx_planning.pmp_planner.plan_store import PlanRecord, PlanStore, _same_state
from .braking import BrakeConfig, BrakeReservation, PrefixBrakeReservation, reserve_prefix_brake
from .validity import Footprint, GridCollisionChecker, ValidityResult


@dataclass(frozen=True)
class Decision:
    action: str
    status: str
    message: str
    anchor_index: int | None = None
    execute_at: float | None = None
    brake: BrakeReservation | None = None
    validity: ValidityResult | None = None
    plan_id: str | None = None

    @property
    def accepted(self):
        return self.status in ("remaining_valid", "staged", "committed")


@dataclass(frozen=True)
class _Assessment:
    snapshot: MapSnapshot
    checker: GridCollisionChecker
    parent_id: str
    parent_epoch: float
    current_index: int
    assessed_at: float
    motion_bounds: np.ndarray
    error_bounds: np.ndarray
    reserve: PrefixBrakeReservation
    record: PlanRecord


def _bounds(values, count, name):
    array = np.asarray(values, dtype=float)
    if array.shape != (count,) or not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError(f"{name} must contain {count} finite nonnegative values")
    return np.frombuffer(array.tobytes(), dtype=float)


class ReplanningCoordinator:
    """Keep the active plan until a complete checked replacement can activate.

    STOP_REPLAN with a future execute_at reserves a fallback while work runs.
    A deadline_missed result is FAILED: a late call cannot claim that the old
    braking certificate still applies. This object never publishes commands.
    """

    def __init__(self, store: PlanStore, c_v, c_w, brake_config: BrakeConfig, *,
                 footprint=Footprint(), state_tolerance=1e-6, time_tolerance=1e-6):
        if not isinstance(store, PlanStore) or not isinstance(brake_config, BrakeConfig):
            raise ValueError("coordinator requires a plan store and brake configuration")
        if not all(math.isfinite(v) and v > 0 for v in (c_v, c_w)):
            raise ValueError("drive coefficients must be finite and positive")
        if not all(math.isfinite(v) and v >= 0 for v in (state_tolerance, time_tolerance)):
            raise ValueError("activation tolerances must be finite and nonnegative")
        self.store, self.c_v, self.c_w = store, float(c_v), float(c_w)
        self.brake_config, self.footprint = brake_config, footprint
        self.state_tolerance, self.time_tolerance = state_tolerance, time_tolerance
        self._assessment = None
        self._pending_action = None
        self._pending_assessment = None
        self._pending_record = None

    @property
    def reservation(self):
        return self._assessment.reserve if self._assessment else None

    @staticmethod
    def _segment_indices(plan):
        starts = [segment.start_time for segment in plan.segments] + [float(plan.times[-1])]
        return [i for i, time in enumerate(plan.times)
                if any(abs(time - start) <= 1e-9 for start in starts)]

    def _fallback(self, status, message):
        assessment = self._assessment
        if (assessment is None or not assessment.reserve.feasible
                or self.store.active is not assessment.record):
            return Decision("FAILED", status, message)
        reserve = assessment.reserve
        return Decision("STOP_REPLAN", status, message,
                        assessment.current_index + reserve.anchor_index,
                        reserve.decision_time, reserve.brake, reserve.prefix_validity,
                        assessment.parent_id)

    def assess(self, snapshot, *, now, current_index, measured_state,
               interval_motion_bounds, pose_error_bounds=None,
               quality_trigger=False, candidate_indices=None):
        """Validate all remaining motion and reserve a discrete stopping anchor.

        now and current_index must identify the same raw boundary. Arbitrary
        measurement-time propagation is deliberately not assumed here. Bounds
        cover the complete active plan; optional candidate_indices are absolute
        raw-boundary indices and restrict the declared braking anchor set.
        """
        self._assessment = None
        self._pending_action = None
        self._pending_assessment = None
        self._pending_record = None
        try:
            plan = self.store.active
            if plan is None or not isinstance(snapshot, MapSnapshot):
                raise ValueError("an active plan and immutable map snapshot are required")
            current_index = operator.index(current_index)
            if not 0 <= current_index < len(plan.times):
                raise ValueError("current index is outside the active plan")
            if (not math.isfinite(now)
                    or abs(now - plan.start_epoch - plan.times[current_index]) > self.time_tolerance):
                raise ValueError("assessment must use the current raw boundary epoch")
            state = np.asarray(measured_state, float)
            if state.shape != (5,) or not _same_state(state, plan.raw_states[current_index], self.state_tolerance):
                raise ValueError("measured full state differs from the declared active boundary")
            if plan.field_key.geometry != snapshot.geometry:
                raise ValueError("changed map geometry requires a fresh stopped planning request")
            motion = _bounds(interval_motion_bounds, len(plan.commands), "motion bounds")
            errors = _bounds(np.zeros(len(plan.times)) if pose_error_bounds is None else pose_error_bounds,
                             len(plan.times), "pose error bounds")
            checker = GridCollisionChecker(snapshot.obstacles, snapshot.geometry.resolution,
                                            snapshot.geometry.origin[:2], map_revision=snapshot.identity,
                                            footprint=self.footprint)
            indices = (list(range(current_index, len(plan.times))) if candidate_indices is None
                       else [operator.index(i) for i in candidate_indices])
            if any(i < current_index or i >= len(plan.times) for i in indices):
                raise ValueError("brake candidates must be unexecuted raw boundaries")
            times = plan.start_epoch + plan.times[current_index:]
            reserve = reserve_prefix_brake(
                checker, plan.raw_states[current_index:], times, self.c_v, self.c_w,
                self.brake_config, interval_motion_bounds=motion[current_index:],
                pose_error_bounds=errors[current_index:],
                candidate_indices=[i - current_index for i in indices],
            )
            # Geometric KEEP uses the same declared uncertainty model as the
            # prefix reserve, including heading error over travelled distance.
            travelled = np.r_[0.0, np.cumsum(motion[current_index:])]
            inflation = (errors[current_index:] + self.brake_config.position_uncertainty
                         + (checker.footprint.radius + travelled) * self.brake_config.heading_uncertainty)
            whole = checker.check_path(plan.raw_states[current_index:, :3], times,
                                       interval_motion_bounds=motion[current_index:],
                                       pose_error_bounds=inflation)
            self._assessment = _Assessment(snapshot, checker, plan.plan_id, plan.start_epoch,
                                            current_index, float(now), motion, errors, reserve, plan)
            self.store.discard_pending()
            if not reserve.feasible:
                return Decision("FAILED", "no_safe_reserve", "no declared delay and brake path is certified",
                                validity=whole, plan_id=plan.plan_id)
            if whole.valid and not quality_trigger:
                return Decision("KEEP", "remaining_valid", "the complete remaining path is valid",
                                current_index + reserve.anchor_index, reserve.decision_time,
                                reserve.brake, whole, plan.plan_id)
            return self._fallback("replacement_required", "complete a moving replacement before the reserved brake deadline")
        except (TypeError, ValueError, IndexError) as error:
            return Decision("FAILED", "invalid_input", str(error))

    def _current(self):
        assessment, active = self._assessment, self.store.active
        if (assessment is None or active is None or not assessment.reserve.feasible
                or active is not assessment.record):
            raise ValueError("the current active plan has no matching braking assessment")
        return assessment, active

    @staticmethod
    def _field_matches_snapshot(key, snapshot):
        return (key.map_session == snapshot.session and key.map_revision == snapshot.revision
                and key.geometry == snapshot.geometry and key.occupancy_policy_id == snapshot.policy.identity
                and key.domain == "complete")

    def stage_candidate(self, candidate, *, action, anchor_index, now,
                        interval_motion_bounds, limits_ok, pose_error_bounds=None):
        """Check the full candidate, including every reused repair suffix interval.

        limits_ok declares solver convergence, required dynamic limits, and the
        required terminal target. It cannot substitute for the checks performed
        here. REPAIR requires typed composition provenance, which is rechecked
        against the actual active parent and its complete retained suffix.
        """
        try:
            if not math.isfinite(now):
                return Decision("FAILED", "invalid_input", "candidate time must be finite")
            assessment, active = self._current()
            if math.isfinite(now) and now > assessment.reserve.decision_time + self.time_tolerance:
                return Decision("FAILED", "deadline_missed", "candidate arrived after the certified braking anchor")
            if action not in ("REPAIR", "SPLICE") or not isinstance(candidate, PlanRecord):
                raise ValueError("candidate action must be REPAIR or SPLICE with a full record")
            if limits_ok is not True:
                raise ValueError("candidate failed its solver, dynamic-limit, or terminal-target gate")
            if (action == "REPAIR") != (candidate.composition is not None):
                raise ValueError("candidate action differs from its repair provenance")
            anchor_index = operator.index(anchor_index)
            if anchor_index not in self._segment_indices(active) or anchor_index < assessment.current_index:
                raise ValueError("candidate departure must be an unexecuted segment boundary")
            anchor = float(active.start_epoch + active.times[anchor_index])
            if (not math.isfinite(now) or now < assessment.assessed_at - self.time_tolerance
                    or now > anchor or anchor > assessment.reserve.decision_time):
                raise ValueError("candidate cannot meet its departure and stopping deadlines")
            if not self._field_matches_snapshot(candidate.field_key, assessment.snapshot):
                raise ValueError("candidate field does not identify the assessed complete map")
            if (candidate.field_key.goal_cell != active.field_key.goal_cell
                    or candidate.field_key.recipe_id != active.field_key.recipe_id
                    or candidate.field_key.numerical_version != active.field_key.numerical_version):
                raise ValueError("candidate goal or field recipe differs from the active request")
            if candidate.composition is not None and candidate.composition.departure_index != anchor_index:
                raise ValueError("repair departure provenance differs from the requested anchor")
            if candidate.composition is not None:
                candidate.validate_repair_parent(active)
            motion = _bounds(interval_motion_bounds, len(candidate.commands), "candidate motion bounds")
            errors = _bounds(np.zeros(len(candidate.times)) if pose_error_bounds is None else pose_error_bounds,
                             len(candidate.times), "candidate pose error bounds")
            travelled = np.r_[0.0, np.cumsum(motion)]
            inflation = (errors + self.brake_config.position_uncertainty
                         + (assessment.checker.footprint.radius + travelled) * self.brake_config.heading_uncertainty)
            validity = assessment.checker.check_path(
                candidate.raw_states[:, :3], candidate.start_epoch + candidate.times,
                interval_motion_bounds=motion, pose_error_bounds=inflation,
            )
            if not validity.valid:
                raise ValueError("candidate or reused suffix is geometrically invalid: " + validity.reason)
            candidate_reserve = reserve_prefix_brake(
                assessment.checker, candidate.raw_states, candidate.start_epoch + candidate.times,
                self.c_v, self.c_w, self.brake_config, interval_motion_bounds=motion,
                pose_error_bounds=errors,
            )
            if not candidate_reserve.feasible:
                raise ValueError("candidate has no certified braking support")
            # Prefix safety is already certified up to the later reserved brake
            # anchor. Store.stage checks parent, session, full state, and epoch.
            self.store.stage(candidate, anchor_index=anchor_index,
                             validation_field_key=candidate.field_key,
                             deadline=assessment.reserve.decision_time)
            self._pending_action = action
            self._pending_assessment = _Assessment(
                assessment.snapshot, assessment.checker, candidate.plan_id, candidate.start_epoch,
                0, anchor, motion, errors, candidate_reserve, candidate,
            )
            self._pending_record = candidate
            return Decision(action, "staged", "complete replacement passed the activation gates",
                            anchor_index, anchor, assessment.reserve.brake, validity, candidate.plan_id)
        except (TypeError, ValueError, IndexError, MemoryError) as error:
            return self._fallback("candidate_rejected", str(error))

    def commit(self, *, now, measured_state, snapshot):
        """Activate only at the staged anchor and unchanged validation map."""
        try:
            if not math.isfinite(now):
                return Decision("FAILED", "invalid_input", "activation time must be finite")
            assessment, active = self._current()
            if not isinstance(snapshot, MapSnapshot) or snapshot.identity != assessment.snapshot.identity:
                self._assessment = None
                return Decision("FAILED", "map_changed", "refresh raw-map validity before activation")
            if math.isfinite(now) and now > assessment.reserve.decision_time + self.time_tolerance:
                return Decision("FAILED", "deadline_missed", "activation missed the certified braking anchor")
            pending = self.store.pending
            if pending is None or self._pending_action is None:
                raise ValueError("there is no checked replacement to activate")
            if pending is not self._pending_record:
                raise ValueError("the pending record is not the checked candidate")
            state = np.asarray(measured_state, float)
            if state.shape != (5,) or not _same_state(state, pending.raw_states[0], self.state_tolerance):
                self._assessment = None
                self._pending_action = None
                self._pending_assessment = None
                self._pending_record = None
                return Decision("FAILED", "state_changed", "measured state invalidates the nominal braking reserve")
            if now > assessment.reserve.decision_time:
                raise ValueError("the reserved stopping deadline has passed")
            committed = self.store.commit_pending(
                now=now, measured_state=measured_state, current_field_key=pending.field_key,
                session_id=active.session_id, state_tolerance=self.state_tolerance,
                time_tolerance=self.time_tolerance,
            )
            result = Decision(self._pending_action, "committed", "replacement activated at its checked full-state anchor",
                              execute_at=float(now), plan_id=committed.plan_id)
            self._assessment = self._pending_assessment
            self._pending_action = None
            self._pending_assessment = None
            self._pending_record = None
            return result
        except (TypeError, ValueError) as error:
            return self._fallback("candidate_rejected", str(error))

    def deadline_decision(self, *, now, snapshot):
        """Return the reserved stop at its declared deadline; never reuse it late."""
        try:
            assessment, _ = self._current()
            if not isinstance(snapshot, MapSnapshot) or snapshot.identity != assessment.snapshot.identity:
                self._assessment = None
                return Decision("FAILED", "map_changed", "refresh raw-map validity before using a brake reserve")
            if not math.isfinite(now) or now < assessment.assessed_at - self.time_tolerance:
                raise ValueError("deadline check must use a finite current execution time")
            if now > assessment.reserve.decision_time + self.time_tolerance:
                return Decision("FAILED", "deadline_missed", "the caller missed the certified braking anchor")
            if now >= assessment.reserve.decision_time - self.time_tolerance:
                self.store.discard_pending()
                self._pending_action = None
                self._pending_assessment = None
                self._pending_record = None
                return self._fallback("brake_due", "execute the reserved brake now, then replan from the measured stop")
            return self._fallback("waiting", "retain the validated prefix and finish replacement before the brake deadline")
        except (TypeError, ValueError) as error:
            return Decision("FAILED", "invalid_input", str(error))
