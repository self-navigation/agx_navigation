"""Owned plan records, full BVP horizons, and checked pending-plan activation.

Times in a plan are relative to start_epoch. Command k acts on
[times[k], times[k + 1]); raw_states contains both interval boundaries.
This module checks identities and joints. Geometric validity belongs to the
caller and must be established before supplying validation_field_key.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, is_dataclass, replace
import hashlib
import json
import math
import operator
import os
from pathlib import Path
import tempfile
from types import MappingProxyType

import numpy as np
from scipy.interpolate import PPoly

from agx_planning.vector_field.contracts import FieldKey


def _owned(value, ndim, name):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim or not np.isfinite(array).all():
        raise ValueError(f"{name} must have {ndim} dimensions and finite values")
    # A bytes backing store prevents callers from making a view writable.
    return np.frombuffer(array.tobytes(order="C"), dtype=np.float64).reshape(array.shape)


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _metadata(value):
    if not isinstance(value, Mapping) or any(not isinstance(k, str) for k in value):
        raise ValueError("metadata must be a JSON object with string keys")
    copied = json.loads(json.dumps(_plain(value), allow_nan=False))

    def freeze(item):
        if isinstance(item, dict):
            return MappingProxyType({key: freeze(v) for key, v in item.items()})
        if isinstance(item, list):
            return tuple(freeze(v) for v in item)
        return item

    return freeze(copied)


def _name(value, name):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")


def _same_state(left, right, tolerance):
    delta = np.asarray(left, float) - np.asarray(right, float)
    if delta.shape[-1:] != (5,) or not np.isfinite(delta).all():
        return False
    delta = delta.copy()
    delta[..., 2] = (delta[..., 2] + np.pi) % (2 * np.pi) - np.pi
    return bool(np.all(np.abs(delta) <= tolerance))


@dataclass(frozen=True, eq=False)
class SegmentSolution:
    """A complete 10-channel BVP spline, independent of its committed prefix."""

    field_key: FieldKey
    start_time: float
    knots: np.ndarray
    coefficients: np.ndarray
    metadata: Mapping = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.field_key, FieldKey):
            raise ValueError("segment requires an identified FieldKey")
        if not math.isfinite(self.start_time) or self.start_time < 0:
            raise ValueError("segment start_time must be finite and nonnegative")
        knots = _owned(self.knots, 1, "knots")
        coefficients = _owned(self.coefficients, 3, "coefficients")
        if (len(knots) < 2 or knots[0] != 0 or np.any(np.diff(knots) <= 0)
                or coefficients.shape[0] < 1
                or coefficients.shape[1:] != (len(knots) - 1, 10)):
            raise ValueError("spline requires increasing knots from zero and 10 channels")
        object.__setattr__(self, "knots", knots)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    @classmethod
    def from_ppoly(cls, solution, field_key, start_time=0.0, metadata=None):
        return cls(field_key, start_time, solution.x, solution.c, metadata or {})

    @property
    def duration(self):
        return float(self.knots[-1])

    @property
    def array_bytes(self):
        return self.knots.nbytes + self.coefficients.nbytes

    def evaluate(self, times):
        """Return (10, N), or (10,) for a scalar; never extrapolate."""
        times = np.asarray(times, float)
        if (times.ndim > 1 or not np.isfinite(times).all()
                or np.any(times < 0) or np.any(times > self.duration)):
            raise ValueError("spline evaluation is outside its stored horizon")
        values = PPoly(self.coefficients, self.knots, extrapolate=False)(times)
        if not np.isfinite(values).all():
            raise ValueError("stored spline evaluation is non-finite")
        return np.moveaxis(values, -1, 0)

    def warm_seed(self, times, shift, initial_state, mode="state_costate", tail=None):
        """Shift the stored horizon; use a caller's fresh tail beyond its end.

        tail receives only shifted query times beyond the stored horizon and
        must return finite (10, M) values. It must not extrapolate this spline.
        The finite requested time grid bounds the requested fresh tail.
        """
        times = np.asarray(times, float)
        initial_state = np.asarray(initial_state, float)
        if (times.ndim != 1 or len(times) == 0 or not np.isfinite(times).all()
                or times[0] != 0 or np.any(np.diff(times) <= 0)
                or not math.isfinite(shift) or not 0 <= shift <= self.duration
                or initial_state.shape != (5,) or not np.isfinite(initial_state).all()):
            raise ValueError("invalid warm-seed times, shift, or initial state")
        if mode not in ("states", "state_costate"):
            raise ValueError("warm-seed mode must be states or state_costate")
        query = times + shift
        if not np.isfinite(query).all():
            raise ValueError("shifted warm-seed times must be finite")
        covered = query <= self.duration
        values = np.empty((10, len(times)), dtype=float)
        values[:, covered] = self.evaluate(query[covered])
        if not covered.all():
            if tail is None:
                raise ValueError("warm seed exceeds the stored horizon; a fresh tail is required")
            extra = np.asarray(tail(query[~covered].copy()), float)
            if extra.shape != (10, int((~covered).sum())) or not np.isfinite(extra).all():
                raise ValueError("fresh tail must return finite (10, M) values")
            values[:, ~covered] = extra
        values[2] = np.unwrap(values[2])
        values[2] += 2 * np.pi * round((initial_state[2] - values[2, 0]) / (2 * np.pi))
        values[:5, 0] = initial_state
        if mode == "states":
            values[5:] = 0.0
        return values

    def __reduce__(self):
        return type(self), (self.field_key, self.start_time, self.knots,
                            self.coefficients, _plain(self.metadata))


@dataclass(frozen=True)
class RepairComposition:
    """Explicit source identity for a new middle joined to a retained suffix."""

    parent_plan_id: str
    repair_plan_id: str
    departure_index: int
    landing_index: int
    join_time: float
    retained_field_keys: tuple[FieldKey, ...]

    def __post_init__(self):
        _name(self.parent_plan_id, "parent_plan_id")
        _name(self.repair_plan_id, "repair_plan_id")
        departure, landing = operator.index(self.departure_index), operator.index(self.landing_index)
        if departure < 0 or landing <= departure:
            raise ValueError("repair landing must follow its departure")
        if not math.isfinite(self.join_time) or self.join_time <= 0:
            raise ValueError("repair join time must be finite and positive")
        keys = tuple(self.retained_field_keys)
        if any(not isinstance(key, FieldKey) for key in keys):
            raise ValueError("retained source keys must be identified")
        object.__setattr__(self, "retained_field_keys", keys)

    def to_dict(self):
        return {"operation": "repair", "parent_plan_id": self.parent_plan_id,
                "repair_plan_id": self.repair_plan_id, "departure_index": self.departure_index,
                "landing_index": self.landing_index, "join_time": self.join_time,
                "retained_field_keys": [key.to_dict() for key in self.retained_field_keys]}

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        if value.pop("operation") != "repair":
            raise ValueError("unsupported composition operation")
        value["retained_field_keys"] = tuple(FieldKey.from_dict(key) for key in value["retained_field_keys"])
        return cls(**value)


@dataclass(frozen=True, eq=False)
class PlanRecord:
    """Raw states, commands, and complete source splines.

    For a repair composition, field_key is the requested current validation
    identity. Each segment keeps its original solve identity. Neither key alone
    establishes geometric validity; the coordinator must check the whole path.
    """

    session_id: str
    plan_id: str
    parent_id: str | None
    field_key: FieldKey
    start_epoch: float
    times: np.ndarray
    raw_states: np.ndarray
    commands: np.ndarray
    accels: np.ndarray
    costates: np.ndarray
    segments: tuple[SegmentSolution, ...] = ()
    status: str = "success"
    metadata: Mapping = field(default_factory=dict)
    composition: RepairComposition | None = None

    def __post_init__(self):
        _name(self.session_id, "session_id")
        _name(self.plan_id, "plan_id")
        _name(self.status, "status")
        if self.parent_id is not None:
            _name(self.parent_id, "parent_id")
            if self.parent_id == self.plan_id:
                raise ValueError("a plan cannot be its own parent")
        if not isinstance(self.field_key, FieldKey):
            raise ValueError("plan requires an identified FieldKey")
        if not math.isfinite(self.start_epoch):
            raise ValueError("start_epoch must be finite")
        for name, ndim in (("times", 1), ("raw_states", 2), ("commands", 2),
                           ("accels", 2), ("costates", 2)):
            object.__setattr__(self, name, _owned(getattr(self, name), ndim, name))
        count = len(self.commands)
        if (self.times.shape != (count + 1,) or self.times[0] != 0
                or np.any(np.diff(self.times) <= 0)
                or self.raw_states.shape != (count + 1, 5)
                or self.commands.shape != (count, 2) or self.accels.shape != (count, 2)
                or self.costates.shape != (count, 5)):
            raise ValueError("plan requires N intervals and N+1 increasing raw state boundaries")
        segments = tuple(self.segments)
        if any(not isinstance(segment, SegmentSolution) for segment in segments):
            raise ValueError("segments must be SegmentSolution records")
        composition = self.composition
        if composition is not None:
            if not isinstance(composition, RepairComposition) or composition.parent_plan_id != self.parent_id:
                raise ValueError("repair composition requires its declared parent")
            if not np.any(np.isclose(self.times, composition.join_time, rtol=0, atol=1e-9)):
                raise ValueError("repair join must be a raw state boundary")
            retained = tuple(segment.field_key for segment in segments
                             if segment.start_time >= composition.join_time - 1e-9)
            if retained != composition.retained_field_keys:
                raise ValueError("retained spline identities differ from repair provenance")
        previous = -1.0
        for index, segment in enumerate(segments):
            if ((composition is None or segment.start_time < composition.join_time - 1e-9)
                    and segment.field_key != self.field_key):
                raise ValueError("all segments must use the plan field key")
            if segment.start_time <= previous or segment.start_time >= self.times[-1]:
                raise ValueError("segment starts must be ordered within the committed plan")
            matches = np.flatnonzero(np.isclose(self.times, segment.start_time, rtol=0, atol=1e-9))
            if len(matches) != 1:
                raise ValueError("segment start must match one raw boundary")
            end = segments[index + 1].start_time if index + 1 < len(segments) else self.times[-1]
            if end - segment.start_time > segment.duration + 1e-9:
                raise ValueError("committed segment exceeds its stored horizon")
            selection = ((self.times >= segment.start_time - 1e-9)
                         & (self.times <= end + 1e-9))
            queries = np.clip(self.times[selection] - segment.start_time, 0, segment.duration)
            if not _same_state(segment.evaluate(queries)[:5].T, self.raw_states[selection], 1e-6):
                raise ValueError("raw state boundaries disagree with their stored spline")
            previous = segment.start_time
        if segments and segments[0].start_time != 0:
            raise ValueError("stored horizons must start at the first plan boundary")
        object.__setattr__(self, "segments", segments)
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    @classmethod
    def from_chunks(cls, chunks, *, initial_state=None, **identity):
        """Copy production chunks and retain each full, uncommitted BVP horizon."""
        if initial_state is not None:
            initial_state = np.asarray(initial_state, float)
            if initial_state.shape != (5,) or not np.isfinite(initial_state).all():
                raise ValueError("initial state must contain five finite raw values")
        chunks = list(chunks)
        if not chunks:
            if initial_state is None:
                raise ValueError("an empty plan requires its initial raw state")
            return cls(**identity, times=[0.0], raw_states=np.asarray(initial_state).reshape(1, 5),
                       commands=np.empty((0, 2)), accels=np.empty((0, 2)),
                       costates=np.empty((0, 5)))
        states, commands, accels, costates, times, segments = [], [], [], [], [0.0], []
        elapsed = 0.0
        for index, chunk in enumerate(chunks):
            if chunk.chunk_idx != index:
                raise ValueError("chunks must be contiguous and ordered from zero")
            if getattr(chunk, "field_key", None) != identity["field_key"]:
                raise ValueError("chunk field identity is missing or differs from the plan")
            if getattr(chunk, "raw_states", None) is None or getattr(chunk, "solution", None) is None:
                raise ValueError("legacy chunks lack raw states or full horizons")
            raw = np.asarray(chunk.raw_states, float)
            count = len(chunk.wheel_cmds)
            if (count == 0 or raw.shape != (count + 1, 5)
                    or not math.isfinite(chunk.dt_sample) or chunk.dt_sample <= 0):
                raise ValueError("invalid chunk sample geometry")
            if states and not _same_state(states[-1][-1], raw[0], 1e-6):
                raise ValueError("raw state discontinuity between chunks")
            if index == 0 and initial_state is not None and not _same_state(raw[0], initial_state, 1e-6):
                raise ValueError("first chunk does not pin the requested initial state")
            diagnostics = getattr(chunk, "solve_diagnostics", None)
            metadata = {"chunk_idx": index}
            if diagnostics is not None:
                metadata["solve_diagnostics"] = asdict(diagnostics) if is_dataclass(diagnostics) else diagnostics
            segments.append(SegmentSolution.from_ppoly(chunk.solution, identity["field_key"],
                                                        elapsed, metadata))
            states.append(raw if index == 0 else raw[1:])
            commands.append(chunk.wheel_cmds)
            accels.append(chunk.accels)
            costates.append(chunk.costates)
            times.extend(elapsed + np.arange(1, count + 1) * chunk.dt_sample)
            elapsed = float(times[-1])
        return cls(**identity, times=times, raw_states=np.concatenate(states),
                   commands=np.concatenate(commands), accels=np.concatenate(accels),
                   costates=np.concatenate(costates), segments=tuple(segments))

    @classmethod
    def compose_repair(cls, parent, middle, *, departure_index, landing_index, plan_id):
        """Join a complete middle to a full retained suffix at segment boundaries.

        The retained suffix is explicitly retimed to the middle's arrival.
        This factory checks full-state joints and source provenance, not collision
        clearance or dynamic limits. It supports static-map validity only.
        """
        if not isinstance(parent, cls) or not isinstance(middle, cls):
            raise ValueError("repair composition requires full plan records")
        departure_index, landing_index = operator.index(departure_index), operator.index(landing_index)
        if not 0 <= departure_index < landing_index < len(parent.times):
            raise ValueError("repair indices must select ordered parent boundaries")
        if (middle.parent_id != parent.plan_id or middle.session_id != parent.session_id
                or middle.status != "success" or parent.status != "success"
                or not middle.capabilities["full_horizons"] or not parent.capabilities["full_horizons"]
                or middle.composition is not None or not len(middle.commands)):
            raise ValueError("repair middle must be a successful fresh child of the parent")
        departure_time = float(parent.times[departure_index])
        landing_time = float(parent.times[landing_index])
        boundaries = [segment.start_time for segment in parent.segments] + [float(parent.times[-1])]
        if not any(abs(departure_time - value) <= 1e-9 for value in boundaries):
            raise ValueError("repair departure must be a stored segment boundary")
        if not any(abs(landing_time - value) <= 1e-9 for value in boundaries):
            raise ValueError("repair landing must be a stored segment boundary")
        if (abs(middle.start_epoch - parent.start_epoch - departure_time) > 1e-9
                or not _same_state(middle.raw_states[0], parent.raw_states[departure_index], 1e-6)
                or not _same_state(middle.raw_states[-1], parent.raw_states[landing_index], 1e-6)):
            raise ValueError("repair must match both full-state endpoints and its departure epoch")
        join = float(middle.times[-1])
        retained = tuple(replace(segment, start_time=join + (segment.start_time - landing_time))
                         for segment in parent.segments if segment.start_time >= landing_time - 1e-9)
        composition = RepairComposition(parent.plan_id, middle.plan_id, departure_index,
                                        landing_index, join, tuple(segment.field_key for segment in retained))
        return cls(middle.session_id, plan_id, parent.plan_id, middle.field_key, middle.start_epoch,
                   np.r_[middle.times, join + parent.times[landing_index + 1:] - landing_time],
                   np.concatenate((middle.raw_states, parent.raw_states[landing_index + 1:])),
                   np.concatenate((middle.commands, parent.commands[landing_index:])),
                   np.concatenate((middle.accels, parent.accels[landing_index:])),
                   np.concatenate((middle.costates, parent.costates[landing_index:])),
                   middle.segments + retained, "success",
                   {"middle_metadata": _plain(middle.metadata), "retimed_static_suffix": True}, composition)

    def validate_repair_parent(self, parent):
        """Check loaded or copied repair provenance against the actual parent.

        A frozen dataclass can still be copied with changed constructor values.
        Typed metadata therefore is not a substitute for checking retained data.
        """
        info = self.composition
        if (not isinstance(parent, PlanRecord) or info is None
                or info.parent_plan_id != parent.plan_id or self.parent_id != parent.plan_id
                or self.session_id != parent.session_id):
            raise ValueError("repair provenance does not identify the active parent")
        departure, landing = info.departure_index, info.landing_index
        if not 0 <= departure < landing < len(parent.times):
            raise ValueError("repair provenance selects invalid parent boundaries")
        landing_time = float(parent.times[landing])
        parent_starts = [segment.start_time for segment in parent.segments] + [float(parent.times[-1])]
        if any(not any(abs(time - start) <= 1e-9 for start in parent_starts)
               for time in (parent.times[departure], landing_time)):
            raise ValueError("repair provenance requires parent segment boundaries")
        if (abs(self.start_epoch - parent.start_epoch - parent.times[departure]) > 1e-9
                or not _same_state(self.raw_states[0], parent.raw_states[departure], 1e-6)):
            raise ValueError("repair departure differs from the declared parent state")
        matches = np.flatnonzero(np.isclose(self.times, info.join_time, rtol=0, atol=1e-9))
        if len(matches) != 1:
            raise ValueError("repair join must select one raw boundary")
        join = int(matches[0])
        expected_times = info.join_time + (parent.times[landing:] - landing_time)
        if (self.times[join:].shape != expected_times.shape
                or not np.allclose(self.times[join:], expected_times, rtol=0, atol=1e-9)
                or not _same_state(self.raw_states[join:], parent.raw_states[landing:], 1e-6)):
            raise ValueError("retained repair state suffix differs from the declared parent landing")
        for name in ("commands", "accels", "costates"):
            if not np.array_equal(getattr(self, name)[join:], getattr(parent, name)[landing:]):
                raise ValueError("retained repair command suffix differs from its parent")
        expected_segments = [segment for segment in parent.segments if segment.start_time >= landing_time - 1e-9]
        retained_segments = [segment for segment in self.segments if segment.start_time >= info.join_time - 1e-9]
        if len(expected_segments) != len(retained_segments):
            raise ValueError("retained repair spline count differs from its parent")
        for original, retained in zip(expected_segments, retained_segments):
            if (retained.field_key != original.field_key
                    or abs(retained.start_time - info.join_time - (original.start_time - landing_time)) > 1e-9
                    or not np.array_equal(retained.knots, original.knots)
                    or not np.array_equal(retained.coefficients, original.coefficients)
                    or _plain(retained.metadata) != _plain(original.metadata)):
                raise ValueError("retained repair spline provenance differs from its parent")
        return True

    @property
    def array_bytes(self):
        return sum(getattr(self, name).nbytes for name in
                   ("times", "raw_states", "commands", "accels", "costates")) + sum(
                       segment.array_bytes for segment in self.segments)

    @property
    def capabilities(self):
        return {"raw_states": True, "full_horizons": bool(self.segments) or len(self.commands) == 0}

    def __reduce__(self):
        return type(self), (self.session_id, self.plan_id, self.parent_id, self.field_key,
                            self.start_epoch, self.times, self.raw_states, self.commands,
                            self.accels, self.costates, self.segments, self.status,
                            _plain(self.metadata), self.composition)


@dataclass(frozen=True, eq=False)
class LegacyPlan:
    """Readable old arrays with no implied raw-state or spline capabilities."""

    arrays: Mapping
    metadata: Mapping = field(default_factory=dict)

    def __post_init__(self):
        copied = {}
        for key, value in self.arrays.items():
            array = np.asarray(value)
            if array.dtype.hasobject:
                raise ValueError("legacy object arrays are unsupported")
            copied[key] = np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
        object.__setattr__(self, "arrays", MappingProxyType(copied))
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    @property
    def capabilities(self):
        return {"raw_states": False, "full_horizons": False}

    def __reduce__(self):
        return type(self), (dict(self.arrays), _plain(self.metadata))


def _paths(path):
    path = Path(path)
    if path.suffix in (".npz", ".json"):
        return path.with_suffix(".npz"), path.with_suffix(".json")
    return Path(str(path) + ".npz"), Path(str(path) + ".json")


def save_plan(plan, path, *, overwrite=False):
    """Write schema version 1 NPZ arrays and a checked JSON sidecar."""
    if not isinstance(plan, PlanRecord):
        raise TypeError("only identified PlanRecord objects can be saved in the new schema")
    npz_path, json_path = _paths(path)
    if not overwrite and (npz_path.exists() or json_path.exists()):
        raise FileExistsError("plan output already exists")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {name: getattr(plan, name) for name in ("times", "raw_states", "commands", "accels", "costates")}
    segments = []
    for index, segment in enumerate(plan.segments):
        arrays[f"segment_{index}_knots"] = segment.knots
        arrays[f"segment_{index}_coefficients"] = segment.coefficients
        segments.append({"start_time": segment.start_time, "field_key": segment.field_key.to_dict(),
                         "metadata": _plain(segment.metadata)})
    document = {"schema_version": 1, "session_id": plan.session_id, "plan_id": plan.plan_id,
                "parent_id": plan.parent_id, "field_key": plan.field_key.to_dict(),
                "start_epoch": plan.start_epoch, "status": plan.status,
                "metadata": _plain(plan.metadata), "segments": segments,
                "composition": plan.composition.to_dict() if plan.composition else None}
    temporary = []
    try:
        with tempfile.NamedTemporaryFile(dir=npz_path.parent, suffix=".npz", delete=False) as stream:
            temporary.append(Path(stream.name))
            np.savez_compressed(stream, **arrays)
        document["arrays_sha256"] = hashlib.sha256(temporary[0].read_bytes()).hexdigest()
        with tempfile.NamedTemporaryFile(mode="w", encoding="ascii", dir=json_path.parent,
                                         suffix=".json", delete=False) as stream:
            temporary.append(Path(stream.name))
            json.dump(document, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary[0], npz_path)
        os.replace(temporary[1], json_path)
    finally:
        for candidate in temporary:
            candidate.unlink(missing_ok=True)
    return npz_path, json_path


def load_plan(path):
    """Load checked schema 1 data, or explicitly identify an old NPZ archive."""
    npz_path, json_path = _paths(path)
    if not json_path.exists():
        with np.load(npz_path, allow_pickle=False) as archive:
            return LegacyPlan({name: archive[name] for name in archive.files})
    document = json.loads(json_path.read_text())
    if document.get("schema_version") in (None, 0):
        with np.load(npz_path, allow_pickle=False) as archive:
            return LegacyPlan({name: archive[name] for name in archive.files}, document)
    if document.get("schema_version") != 1:
        raise ValueError("unsupported plan schema version")
    if hashlib.sha256(npz_path.read_bytes()).hexdigest() != document.get("arrays_sha256"):
        raise ValueError("plan arrays and JSON sidecar do not match")
    key = FieldKey.from_dict(document["field_key"])
    with np.load(npz_path, allow_pickle=False) as archive:
        expected = {"times", "raw_states", "commands", "accels", "costates"}
        segments = []
        for index, item in enumerate(document["segments"]):
            names = (f"segment_{index}_knots", f"segment_{index}_coefficients")
            expected.update(names)
            source_key = FieldKey.from_dict(item["field_key"]) if "field_key" in item else key
            segments.append(SegmentSolution(source_key, item["start_time"], archive[names[0]],
                                             archive[names[1]], item["metadata"]))
        if set(archive.files) != expected:
            raise ValueError("plan archive arrays do not match its schema")
        arrays = {name: archive[name] for name in ("times", "raw_states", "commands", "accels", "costates")}
        return PlanRecord(document["session_id"], document["plan_id"], document["parent_id"],
                          key, document["start_epoch"], **arrays, segments=tuple(segments),
                          status=document["status"], metadata=document["metadata"],
                          composition=(RepairComposition.from_dict(document["composition"])
                                       if document.get("composition") else None))


@dataclass(frozen=True)
class _Pending:
    plan: PlanRecord
    parent_epoch: float
    anchor_index: int
    anchor_time: float
    deadline: float
    validation_field_key: FieldKey


class PlanStore:
    """Bound retained arrays; preserve active playback until checked activation."""

    def __init__(self, max_records=4, max_bytes=256 * 1024**2):
        self.max_records = operator.index(max_records)
        self.max_bytes = operator.index(max_bytes)
        if self.max_records < 2 or self.max_bytes <= 0:
            raise ValueError("plan store needs at least two records and a positive array budget")
        self.clear()

    def clear(self):
        self._active = None
        self._pending = None
        self._previous = []

    @property
    def active(self):
        return self._active

    @property
    def pending(self):
        return self._pending.plan if self._pending else None

    @property
    def previous(self):
        return tuple(self._previous)

    @property
    def stats(self):
        records = [p for p in (self.active, self.pending) if p is not None] + self._previous
        return {"records": len(records), "array_bytes": sum(p.array_bytes for p in records),
                "previous_records": len(self._previous), "pending": self._pending is not None}

    @staticmethod
    def _executable(plan):
        if not isinstance(plan, PlanRecord) or not plan.capabilities["full_horizons"]:
            raise ValueError("execution requires identified raw states and full BVP horizons")
        if plan.status != "success":
            raise ValueError("only a successful complete plan can enter playback")

    def _retained_previous(self, required):
        previous = self._previous.copy()
        while (len(required) + len(previous) > self.max_records
               or sum(p.array_bytes for p in required + previous) > self.max_bytes):
            if not previous:
                raise MemoryError("active and pending plan arrays exceed the store budget")
            previous.pop(0)
        return previous

    def install_initial(self, plan):
        self._executable(plan)
        if self.active is not None or plan.parent_id is not None:
            raise ValueError("initial installation requires an empty store and no parent plan")
        previous = self._retained_previous([plan])
        self._active, self._previous = plan, previous

    def stage(self, plan, *, anchor_index, validation_field_key, deadline):
        self._executable(plan)
        anchor_index = operator.index(anchor_index)
        active = self.active
        if (active is None or plan.session_id != active.session_id
                or plan.parent_id != active.plan_id or plan.plan_id == active.plan_id):
            raise ValueError("pending plan has a stale parent or session")
        if not 0 <= anchor_index < len(active.times):
            raise ValueError("anchor index is outside the active plan")
        anchor_time = active.start_epoch + active.times[anchor_index]
        if (not math.isfinite(deadline) or deadline < anchor_time
                or abs(plan.start_epoch - anchor_time) > 1e-9
                or not _same_state(plan.raw_states[0], active.raw_states[anchor_index], 1e-6)):
            raise ValueError("pending plan does not match its active anchor or deadline")
        if validation_field_key != plan.field_key:
            raise ValueError("pending plan and validation field differ")
        previous = self._retained_previous([active, plan])
        pending = _Pending(plan, active.start_epoch, anchor_index, float(anchor_time),
                           float(deadline), validation_field_key)
        self._pending, self._previous = pending, previous

    def discard_pending(self):
        self._pending = None

    def commit_pending(self, *, now, measured_state, current_field_key, session_id,
                       state_tolerance=1e-6, time_tolerance=1e-6):
        pending, active = self._pending, self.active
        if (not math.isfinite(now) or not math.isfinite(state_tolerance)
                or not math.isfinite(time_tolerance) or state_tolerance < 0 or time_tolerance < 0):
            raise ValueError("activation times and tolerances must be finite and nonnegative")
        if pending is None or active is None:
            raise ValueError("no pending plan to activate")
        if (session_id != active.session_id or pending.plan.session_id != session_id
                or pending.plan.parent_id != active.plan_id or pending.parent_epoch != active.start_epoch):
            raise ValueError("pending plan has a stale parent or session")
        if current_field_key != pending.validation_field_key:
            raise ValueError("pending validation belongs to a different field")
        if abs(now - pending.anchor_time) > time_tolerance or now > pending.deadline:
            raise ValueError("pending plan missed its activation time or deadline")
        state = np.asarray(measured_state, float)
        if state.shape != (5,) or not _same_state(state, pending.plan.raw_states[0], state_tolerance):
            raise ValueError("measured state differs from the pending anchor")
        # All checks precede mutation. Stage already reserved enough array space.
        self._previous.append(active)
        self._active, self._pending = pending.plan, None
        return self._active
