"""Run offline PMP rollouts in a child process, outside the node's GIL.

The child is ROS-free and talks over one duplex multiprocessing.Pipe:
parent -> child ("field", array), ("plan", x0, goal), ("quit",); child ->
parent ("chunk", RolloutChunk), ("done", RolloutResult), ("error", repr).
Field updates received during a rollout are queued for the next request; one
rollout never mixes field revisions. Typed FieldSnapshot payloads carry the
source key. Legacy arrays remain explicitly unidentified. Preempt and cancel set an mp.Event that
the child polls between solves; if it does not answer within stop_grace_s the
parent kills and respawns it. Uses the spawn start method, since forking a
process that holds rclpy/DDS threads is unsafe.
"""

from __future__ import annotations

import multiprocessing as mp
import threading
import time
from typing import Callable, Iterator, Optional

import numpy as np


def _child_main(conn, stop_event, cfg) -> None:
    from agx_planning.pmp_planner.rollout import (
        RolloutResult, parse_field_array, rollout_generator,
    )
    from agx_planning.pmp_planner.shooting_solver import PMPShootingSolver
    from agx_planning.vector_field import VectorFieldGrid

    from agx_planning.vector_field.contracts import FieldSnapshot

    solver = PMPShootingSolver(cfg, VectorFieldGrid())
    missing = object()
    pending_field = missing
    field_error = "No field has been supplied"

    def absorb(msg) -> bool:
        """Install an idle field, or invalidate it explicitly on bad input."""
        nonlocal field_error
        if msg[0] == "field":
            try:
                value = msg[1]
                field = (value.build_grid(cfg) if isinstance(value, FieldSnapshot)
                         else parse_field_array(value, cfg))
                if field is None or not field.ready:
                    raise ValueError("Invalid field payload")
                solver.field = field.frozen_copy()
                field_error = None
            except (TypeError, ValueError, IndexError, OverflowError) as error:
                # An invalid new request must not leave a stale valid field.
                solver.field = VectorFieldGrid()
                field_error = f"Invalid field payload: {error}"
        elif msg[0] == "quit":
            return False
        else:
            raise ValueError(f"Unknown solver message: {msg[0]}")
        return True

    while True:
        if pending_field is not missing:
            absorb(("field", pending_field))
            pending_field = missing
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            return
        if msg[0] != "plan":
            try:
                if not absorb(msg):
                    return
            except Exception as error:
                conn.send(("error", repr(error)))
            continue
        _, x0, goal, *expected = msg
        expected_key = expected[0] if expected else None
        quit_after = False

        def stop_fn() -> Optional[str]:
            nonlocal quit_after, pending_field
            while conn.poll():
                incoming = conn.recv()
                if incoming[0] == "field":
                    # Coalesce only future input. The current solver keeps the
                    # exact frozen grid chosen at this request's start.
                    pending_field = incoming[1]
                elif incoming[0] == "quit":
                    quit_after = True
                    return "stopped"
                else:
                    raise ValueError("Overlapping or unknown solver request")
            return "stopped" if stop_event.is_set() else None

        solver.reset_warm_start()
        try:
            if field_error is not None:
                conn.send(("done", RolloutResult(status="failed", message=field_error)))
                continue
            if expected_key is not None and solver.field.field_key != expected_key:
                conn.send(("done", RolloutResult(status="failed", message="Requested field key is unavailable")))
                continue
            solver.field.require_goal(goal)
            gen = rollout_generator(solver, cfg, x0, goal, stop_fn)
            while True:
                try:
                    conn.send(("chunk", next(gen)))
                except StopIteration as stop:
                    conn.send(("done", stop.value))
                    break
        except Exception as error:
            conn.send(("error", repr(error)))
        if quit_after:
            return


class SolverProcess:
    """Parent-side handle. Thread-safe for one planner thread plus one field
    thread; rollouts themselves are serialised by the caller (``_exec_lock``)."""

    def __init__(self, cfg, stop_grace_s: float = 5.0):
        self._cfg = cfg
        self._grace = stop_grace_s
        self._ctx = mp.get_context("spawn")
        self._send_lock = threading.Lock()
        self._field = None
        self._closed = False
        self._spawn()

    def _spawn(self) -> None:
        with self._send_lock:
            if self._closed:
                return
            self._conn, child = self._ctx.Pipe(duplex=True)
            self._stop = self._ctx.Event()
            self._proc = self._ctx.Process(
                target=_child_main, args=(child, self._stop, self._cfg),
                name="pmp_solver", daemon=True)
            self._proc.start()
            child.close()
            if self._field is not None:
                self._conn.send(("field", self._field))

    def _send(self, msg) -> None:
        with self._send_lock:
            self._conn.send(msg)

    def set_field(self, data) -> None:
        """Copy legacy data or retain an immutable identified FieldSnapshot."""
        from agx_planning.vector_field.contracts import FieldSnapshot
        value = data if isinstance(data, FieldSnapshot) else np.array(data, copy=True)
        with self._send_lock:
            if self._closed:
                raise RuntimeError("SolverProcess is closed")
            self._field = value
            try:
                self._conn.send(("field", value))
            except (BrokenPipeError, OSError):
                pass  # A restarted child receives the latest owned payload.

    def rollout(self, x0, goal, stop_fn: Callable[[], Optional[str]], *, field_key=None
                ) -> Iterator:
        """Yield RolloutChunks, or None when nothing is ready yet (the caller
        should sleep and resume); the generator's return value is the
        RolloutResult (use with GeneratorReturnCatcher, like rollout_generator)."""
        from agx_planning.pmp_planner.rollout import RolloutResult

        if self._closed:
            raise RuntimeError("SolverProcess is closed")
        if not self._proc.is_alive():
            self._restart()
        self._stop.clear()
        try:
            with self._send_lock:
                expected = field_key if field_key is not None else getattr(self._field, "key", None)
                self._conn.send(("plan", np.array(x0, dtype=float, copy=True),
                                 np.array(goal, dtype=float, copy=True), expected))
        except (BrokenPipeError, OSError):
            self._restart()
            return RolloutResult(status="failed", message="Solver process connection closed")
        reason: Optional[str] = None
        stop_t = 0.0
        while True:
            if reason is None:
                reason = stop_fn()
                if reason is not None:
                    self._stop.set()
                    stop_t = time.monotonic()
            elif time.monotonic() - stop_t > self._grace:
                self._restart()
                return RolloutResult(status=reason, message=reason.capitalize())
            # Never block here: the caller is a coroutine on a single-threaded
            # executor, so an idle tick yields None and the caller sleeps.
            try:
                ready = self._conn.poll(0)
            except (OSError, ValueError):
                self._restart()
                return RolloutResult(status="failed", message="Solver process connection closed")
            if not ready:
                if not self._proc.is_alive():
                    self._restart()
                    return RolloutResult(status="failed",
                                         message="Solver process died")
                yield None
                continue
            try:
                kind, payload = self._conn.recv()
            except (EOFError, OSError):
                self._restart()
                return RolloutResult(status="failed", message="Solver process connection closed")
            if kind == "chunk":
                if reason is None:
                    yield payload
            elif kind == "done":
                if payload.status == "stopped":
                    return RolloutResult(status=reason or "preempted",
                                         message=(reason or "preempted").capitalize())
                return payload
            else:
                return RolloutResult(status="failed",
                                     message=f"Solver process error: {payload}")

    def _restart(self) -> None:
        if self._proc.is_alive():
            self._proc.terminate()
        self._proc.join(2.0)
        if self._proc.is_alive():
            self._proc.kill()
            self._proc.join(2.0)
        self._conn.close()
        if not self._closed:
            self._spawn()

    def close(self) -> None:
        with self._send_lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._conn.send(("quit",))
            except (BrokenPipeError, OSError):
                pass
        self._proc.join(2.0)
        if self._proc.is_alive():
            self._proc.kill()
            self._proc.join(2.0)
        self._conn.close()
