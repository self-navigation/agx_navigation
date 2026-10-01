"""Run offline PMP rollouts in a child process, outside the node's GIL.

The child is ROS-free and talks over one duplex multiprocessing.Pipe:
parent -> child ("field", array), ("plan", x0, goal), ("quit",); child ->
parent ("chunk", RolloutChunk), ("done", RolloutResult), ("error", repr).
A field update may arrive mid-rollout. Preempt and cancel set an mp.Event that
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

    solver = PMPShootingSolver(cfg, VectorFieldGrid())

    def absorb(msg) -> bool:
        """Handle a non-plan message; return False on quit."""
        if msg[0] == "field":
            f = parse_field_array(msg[1], cfg)
            if f is not None:
                solver.field = f
        elif msg[0] == "quit":
            return False
        return True

    while True:
        try:
            msg = conn.recv()
        except EOFError:
            return
        if msg[0] != "plan":
            if not absorb(msg):
                return
            continue
        _, x0, goal = msg
        quit_after = False

        def stop_fn() -> Optional[str]:
            nonlocal quit_after
            # Field updates mid-rollout take effect on the next segment,
            # as they did when the solver lived in the node.
            while conn.poll():
                if not absorb(conn.recv()):
                    quit_after = True
                    return "stopped"
            return "stopped" if stop_event.is_set() else None

        solver.reset_warm_start()
        try:
            gen = rollout_generator(solver, cfg, x0, goal, stop_fn)
            while True:
                try:
                    conn.send(("chunk", next(gen)))
                except StopIteration as stop:
                    conn.send(("done", stop.value))
                    break
        except Exception as e:  # report, keep serving
            conn.send(("error", repr(e)))
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
        self._field: Optional[np.ndarray] = None
        self._spawn()

    def _spawn(self) -> None:
        self._conn, child = self._ctx.Pipe(duplex=True)
        self._stop = self._ctx.Event()
        self._proc = self._ctx.Process(
            target=_child_main, args=(child, self._stop, self._cfg),
            name="pmp_solver", daemon=True)
        self._proc.start()
        child.close()
        if self._field is not None:
            self._send(("field", self._field))

    def _send(self, msg) -> None:
        with self._send_lock:
            self._conn.send(msg)

    def set_field(self, data: np.ndarray) -> None:
        self._field = data
        try:
            self._send(("field", data))
        except (BrokenPipeError, OSError):
            pass  # child being respawned; _spawn resends the latest field

    def rollout(self, x0, goal, stop_fn: Callable[[], Optional[str]]
                ) -> Iterator:
        """Yield RolloutChunks; the generator's return value is the
        RolloutResult (use with GeneratorReturnCatcher, like rollout_generator)."""
        from agx_planning.pmp_planner.rollout import RolloutResult

        if not self._proc.is_alive():
            self._spawn()
        self._stop.clear()
        self._send(("plan", np.asarray(x0, float), np.asarray(goal, float)))
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
            # poll() blocks in C without the GIL -- the whole point.
            if not self._conn.poll(0.05):
                if not self._proc.is_alive():
                    self._restart()
                    return RolloutResult(status="failed",
                                         message="Solver process died")
                continue
            kind, payload = self._conn.recv()
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
        self._proc.terminate()
        self._proc.join(2.0)
        if self._proc.is_alive():
            self._proc.kill()
        self._spawn()

    def close(self) -> None:
        try:
            self._send(("quit",))
        except Exception:
            pass
        self._proc.join(2.0)
        if self._proc.is_alive():
            self._proc.kill()
