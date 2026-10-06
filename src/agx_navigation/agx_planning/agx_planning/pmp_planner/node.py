"""Vector-field guided PMP planner for a skid-steer robot in wheel space.

Applies Pontryagin's Maximum Principle and solves the TPBVP with
scipy.integrate.solve_bvp. State (p_x, p_y, theta, w_l, w_r), where w_l, w_r
are the left/right wheel-pair speeds; control is the per-wheel acceleration
(a_l, a_r). Body velocities are derived as v = (w_l + w_r) * r/2 and omega =
(w_r - w_l) * r/track_eff, with track_eff = track * slip_chi lumping lateral
skid into the kinematics. The cost combines a travel-time potential, field
alignment, a gated speed reference, a heading-coupled brake, soft wheel-speed
barriers, per-wheel effort and an optional footprint barrier (w_fp) that keeps
the chassis outline off the walls. The position-costate coupling is gated by
pos_gate, a heuristic change to the adjoint against understeer.

Offline only: the node serves the `pmp_planner/plan_to_goal` action, rolling
out the whole trajectory from rest in a child process (solver_process.py) and
streaming each dt_segment chunk (poses, wheel-speed setpoints, accelerations
and costates) as feedback; a new goal pre-empts the running rollout. It also
publishes the rolled-out Path on /pmp_planner/trajectory. It never publishes
wheel commands; the runtime_corrector is their only writer.
"""

from dataclasses import dataclass
from typing import Optional
import threading
import time
from time import perf_counter

import numpy as np

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.clock import Clock, ClockType
import rclpy.task
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Float32MultiArray

from agx_planning_msgs.action import PlanToGoal
from agx_planning.utils import declare_and_load_dataclass, GeneratorReturnCatcher
from agx_planning.vector_field import VectorFieldGrid
from agx_planning.pmp_planner import (
    PMPShootingSolver,
    PlannerConfig,
    TurnDiagnosticLogger,
    RolloutChunk,
    RolloutResult,
    compute_diag_values,
    parse_field_array,
    rollout_generator,
)


@dataclass
class NodeConfig:
    map_frame: str = "map"
    robot_frame: str = "base_link"
    # Set to a file path (e.g. /tmp/pmp_diag.csv) to enable the diagnostic
    # logger. Empty string disables it. The logger writes planned heading
    # profiles and actual odom to CSV for post-analysis; see TurnDiagnosticLogger.
    diag_log_path: str = ""
    # How much to wait for a vector field before giving up.
    # Set to higher than average value,
    # because the very first message takes longer to receive.
    vector_field_timeout: float = 10.0


class PlannerNode(Node):
    """Offline planner: exposes a ROS2 action server
    `pmp_planner/plan_to_goal`. Each goal carries an explicit
    (start_x, start_y, start_theta) and (target_x, target_y, target_theta);
    wheel speeds are zero-initialized (planning from rest). Each committed
    dt_segment-second BVP segment is streamed back as action feedback. A
    new goal arriving mid-rollout preempts the current one: _action_handle_accepted
    fires _exec_stop, the in-flight rollout exits with "preempted", and the
    new goal then runs once the previous finishes (_exec_busy). Path-masked replan
    detection lives in the interpreter, not here -- the planner only sees
    fresh action goals.

    NOTE: the node runs on a SingleThreadedExecutor and must stay that way.
    rclpy's MultiThreadedExecutor busy-spins under use_sim_time with a
    1 kHz /clock (~115% CPU on an empty node, measured 2026-10-01) and
    starves every callback of the GIL -- the field callback was entered
    4-8 s late. Offline mode therefore never blocks a callback: the execute
    callback is a coroutine that awaits _sleep() between checks, so two
    goals (outgoing + preempting) and field delivery interleave on the
    one thread, and the solve itself runs in a child process.
    """

    def __init__(self):
        super().__init__("pmp_planner")

        self.cfg = declare_and_load_dataclass(self, PlannerConfig())
        self.node_cfg = declare_and_load_dataclass(self, NodeConfig())

        if self.cfg.mode != "offline":
            raise ValueError(
                f"PlannerConfig.mode must be 'offline' (online mode was "
                f"removed 2026-10-06), got {self.cfg.mode!r}"
            )

        # --- Shared state ---
        self._field = VectorFieldGrid()

        # _field_lock / _field_event: _on_field swaps the grid under the
        # lock; a rollout waits on _field_event before starting.
        self._field_lock = threading.Lock()
        self._field_event = threading.Event()

        # The field subscription gets its own (reentrant) group so it is
        # never queued behind the action callbacks' group.
        self._io_cb_group = ReentrantCallbackGroup()

        self._diag_logger: Optional[TurnDiagnosticLogger] = None
        if self.node_cfg.diag_log_path:
            try:
                self._diag_logger = TurnDiagnosticLogger(self.node_cfg.diag_log_path)
                self.get_logger().info(
                    f"Diagnostic logger active -> {self.node_cfg.diag_log_path}"
                )
            except OSError as e:
                self.get_logger().error(f"Cannot open diag log: {e}")

        self._solver = PMPShootingSolver(self.cfg, self._field)

        # --- Subscriptions / publishers ---
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=1,
        )
        self.create_subscription(
            Float32MultiArray, "/vector_field/planner_data", self._on_field, qos,
            callback_group=self._io_cb_group,
        )
        self._traj_pub = self.create_publisher(Path, "/pmp_planner/trajectory", 10)

        self._init_offline()

        self.get_logger().info(f"Planner running in '{self.cfg.mode}' mode.")

    def _init_offline(self):
        """Offline-mode wiring: action server, exec lock/stop, trajectory_id.

        No TF, no /odom, no wheel publisher, no /goal_pose -- the action
        goal carries start and target inline. The interpreter (action
        client) owns chassis-pose snapshots and goal-source subscriptions;
        the executor owns wheel-command publication.
        """
        # _exec_busy serialises rollouts so a preempting goal waits for
        # the previous to finish before starting (a flag suffices: every
        # callback runs on the executor's one thread). _exec_stop is the
        # signal that wakes a still-running rollout: _action_handle_accepted
        # sets it on a new goal arriving, _do_rollout_action's per-iter
        # check sees it and exits with "preempted".
        self._exec_busy = False
        self._exec_stop = threading.Event()
        self._trajectory_id: int = 0

        # The rollout runs in its own process (solver_process.py), so the
        # node's one thread only shuttles messages.
        from agx_planning.pmp_planner.solver_process import SolverProcess
        self._solver_proc = SolverProcess(self.cfg)

        # Reentrant so two execute coroutines (outgoing + preempting) can
        # both be in flight; with one thread this only affects scheduling.
        self._action_cb_group = ReentrantCallbackGroup()
        self._steady = Clock(clock_type=ClockType.STEADY_TIME)

        # Feedback QoS: the planner solves BVPs much faster than the
        # chassis plays them back (a 30-second sim trajectory at
        # dt_segment=1.25s = ~24 segments solved in well under a second
        # of wall clock), so the feedback queue fills with many unconsumed
        # chunks during the burst. depth=64 prevents drops that would
        # manifest as missing samples and incomplete path coverage.
        feedback_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=64,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._action_server = ActionServer(
            self,
            PlanToGoal,
            "pmp_planner/plan_to_goal",
            execute_callback=self._action_execute,
            goal_callback=self._action_goal_callback,
            cancel_callback=self._action_cancel_callback,
            handle_accepted_callback=self._action_handle_accepted,
            callback_group=self._action_cb_group,
            feedback_pub_qos_profile=feedback_qos,
        )
        self.get_logger().info(
            f"Indirect-PMP planner running OFFLINE; action "
            f"'pmp_planner/plan_to_goal'; horizon {self.cfg.T_horizon}s, "
            f"dt_segment {self.cfg.dt_segment}s, chunk samples at "
            f"{self.cfg.control_rate} Hz."
        )

    def destroy_node(self):
        # Wake any in-flight rollout so it can return promptly.
        if hasattr(self, "_exec_stop"):
            self._exec_stop.set()
        if hasattr(self, "_solver_proc"):
            self._solver_proc.close()
        if self._diag_logger is not None:
            self._diag_logger.close()
        super().destroy_node()

    # ---------------- Subscriptions ----------------

    def _on_field(self, msg: Float32MultiArray):
        """Parse the field message and atomically swap in the new VectorFieldGrid.

        Layout (canonical):
          [h, w, origin_x, origin_y, resolution,
           travel_time(H*W), grad_x(H*W), grad_y(H*W), grad_mag(H*W)]
        Backward compatibility:
          - 1-channel (T only): F_unit auto-derived from -grad T.
          - 3-channel (T, gx, gy): grad_mag missing, ignored.

        Path-masked replan detection that used to live here has moved to the interpreter (the action client). The planner is
        now a pure (start, goal, field) -> trajectory function; replanning
        is a fresh action goal.
        """
        _t0 = time.monotonic()
        self.get_logger().info("field msg: callback entered")
        data = np.asarray(msg.data, dtype=np.float32)
        _t1 = time.monotonic()
        new_field = parse_field_array(data, self.cfg)
        _t2 = time.monotonic()
        if new_field is None:
            self.get_logger().warn(
                f"Field size mismatch: got {data.size} floats, "
                f"expected header + n, 3n, or 4n body elements",
                throttle_duration_sec=5.0,
            )
            return

        with self._field_lock:
            # Atomic swap. CPython's GIL makes the bare assignment atomic, so
            # any concurrent reader (a rollout running on the action-server
            # thread) sees either the old or new
            # grid, never a torn update.
            self._field = new_field
            # The solver holds its own reference to the field; rebind it so
            # the next solve uses the new instance. The solver's version-counter
            # additionally drops the warm start because the new instance starts
            # at version=1, never matching the cached _last_field_version.
            self._solver.field = new_field
            if hasattr(self, "_solver_proc"):
                self._solver_proc.set_field(data)
            _t3 = time.monotonic()
            self._field_event.set()

        self.get_logger().warn(
            f"Got field. Size: {self._field._tt.shape}  asarray="
            f"{(_t1 - _t0) * 1e3:.0f}ms parse={(_t2 - _t1) * 1e3:.0f}ms "
            f"swap+forward={(_t3 - _t2) * 1e3:.0f}ms",
            throttle_duration_sec=5.0,
        )

    # ---------------- Action server (offline mode) ----------------

    def _action_goal_callback(self, goal_request) -> GoalResponse:
        # Accept all syntactically-valid goals; semantic validation
        # (frame_id matches map_frame, field is ready, ...) happens in
        # _action_execute_inner so we can return a meaningful result
        # message rather than just rejecting up-front.
        return GoalResponse.ACCEPT

    def _action_cancel_callback(self, goal_handle) -> CancelResponse:
        # Wake the in-flight rollout so the cancellation observed via
        # goal_handle.is_cancel_requested takes effect promptly.
        self._exec_stop.set()
        return CancelResponse.ACCEPT

    def _action_handle_accepted(self, goal_handle):
        """Called on every accepted goal. If a previous rollout is in
        flight, set _exec_stop so it exits with "preempted"; the new
        execute coroutine will then wait on _exec_busy until that one
        finishes. goal_handle.execute() itself is non-blocking (it
        schedules _action_execute as an executor task)."""
        self._exec_stop.set()
        goal_handle.execute()

    async def _sleep(self, seconds: float) -> None:
        """Yield the executor for `seconds` of WALL time (steady clock: a
        ROS-clock timer would stall with the sim and race /clock). The only
        way a callback here may wait -- never block the one thread."""
        fut = rclpy.task.Future()

        def fire():
            timer.cancel()
            fut.set_result(None)

        timer = self.create_timer(seconds, fire, clock=self._steady)
        try:
            await fut
        finally:
            self.destroy_timer(timer)

    async def _action_execute(self, goal_handle):
        """Execute coroutine wrapper: serialise rollouts via _exec_busy so
        a preempting goal cleanly waits for the previous to finish
        before clearing _exec_stop and starting its own rollout."""
        while self._exec_busy:
            await self._sleep(0.01)
        self._exec_busy = True
        try:
            self._exec_stop.clear()
            return await self._action_execute_inner(goal_handle)
        finally:
            self._exec_busy = False

    async def _action_execute_inner(self, goal_handle):
        """Validate the goal, wait for the field, run one rollout, and
        translate the rollout's status string into the appropriate
        action terminal state.

        The trajectory_id counter is bumped here (not earlier) so a goal
        that aborts during validation gets result.trajectory_id = 0,
        which the interpreter uses to distinguish "no rollout was
        attempted" from "the rollout we were watching just ended".

        PlanToGoal carries a 3D start pose (x, y, theta). The 5D BVP also
        needs initial wheel speeds; these are zero-initialized -- planning
        from rest. The first BVP segment will converge to the correct
        velocity profile regardless.
        """
        req = goal_handle.request
        result = PlanToGoal.Result()

        # Frame validation. The planner does its math in map_frame; if
        # the client sent poses in another frame, refuse rather than
        # silently planning in the wrong frame entirely.
        if req.frame_id != self.node_cfg.map_frame:
            err = (
                f"frame_id {req.frame_id!r} does not match planner "
                f"map_frame {self.node_cfg.map_frame!r}"
            )
            self.get_logger().warn(err)
            goal_handle.abort()
            result.success = False
            result.message = err
            result.trajectory_id = 0
            return result

        # 5D initial state: 3D pose from action goal, wheel speeds zero-init.
        x0 = np.array([req.start_x, req.start_y, req.start_theta, 0.0, 0.0])
        goal = np.array([req.target_x, req.target_y, req.target_theta])
        self.get_logger().info(
            f"Action goal: start=({x0[0]:.2f}, {x0[1]:.2f}, {x0[2]:.2f}) "
            f"-> target=({goal[0]:.2f}, {goal[1]:.2f}, {goal[2]:.2f})"
        )

        # Atomically check readiness and arm the event.
        # If the field arrives between the check and the first .wait() call,
        # the event is already set and .wait() returns immediately.
        with self._field_lock:
            if not self._field.ready:
                self._field_event.clear()

        # Wait for the field if it's not yet ready. Bounded so a
        # misconfigured system (no /vector_field/planner_data publisher)
        # doesn't hang the action indefinitely.
        deadline = time.monotonic() + self.node_cfg.vector_field_timeout
        while not self._field.ready:
            if self._exec_stop.is_set():
                # Preempted by a newer goal (or shutdown) before we even
                # got a field. Honour cancel-vs-preempt distinction.
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    msg = "Cancelled while waiting for field"
                else:
                    goal_handle.abort()
                    msg = "Preempted while waiting for field"
                result.success = False
                result.message = msg
                result.trajectory_id = 0
                return result
            if time.monotonic() > deadline:
                goal_handle.abort()
                result.success = False
                result.message = "Timeout waiting for vector field"
                result.trajectory_id = 0
                return result

            # Yield so _on_field can run on the executor's one thread, and
            # wake periodically to re-check _exec_stop and the deadline.
            remaining = deadline - time.monotonic()
            await self._sleep(min(0.05, max(0.001, remaining)))

        self._trajectory_id += 1
        traj_id = self._trajectory_id
        # Snapshot the field for log clarity. _on_field's atomic swap may
        # rebind self._field mid-rollout; subsequent BVP segments pick up
        # whichever instance is current at that point, mirroring the original
        # behaviour.
        self._solver.field = self._field
        self._solver.reset_warm_start()

        try:
            status = await self._do_rollout_action(goal_handle, traj_id, x0, goal)
        except Exception as e:
            self.get_logger().error(f"Offline rollout crashed: {e!r}")
            goal_handle.abort()
            result.success = False
            result.message = f"Rollout exception: {e!r}"
            result.trajectory_id = int(traj_id)
            return result

        if status == "success":
            goal_handle.succeed()
            result.success = True
            result.message = "Goal reached"
        elif status == "cancelled":
            goal_handle.canceled()
            result.success = False
            result.message = "Cancelled"
        elif status == "preempted":
            # New goal arrived mid-rollout. Distinct from a true failure
            # only via the message string -- both terminate as ABORTED.
            goal_handle.abort()
            result.success = False
            result.message = "Preempted"
        else:
            # "failed": BVP failure / stagnation / sim-time cap.
            goal_handle.abort()
            result.success = False
            result.message = "Plan failure (BVP / stagnation / sim-time cap)"
        result.trajectory_id = int(traj_id)
        return result

    async def _do_rollout_action(
        self, goal_handle, traj_id: int, x0: np.ndarray, goal: np.ndarray
    ) -> str:
        """Thin adapter: wire ROS 2 cancel/preempt signals into rollout_generator
        and handle per-chunk publishing and diagnostics.

        Returns the RolloutResult.status string so _action_execute_inner can
        map it to the appropriate action terminal state.
        """

        def stop_fn() -> Optional[str]:
            # Cancel takes precedence: more specific terminal state than preempt.
            if goal_handle.is_cancel_requested:
                return "cancelled"
            return "preempted" if self._exec_stop.is_set() else None

        all_poses: list[np.ndarray] = []

        gen = GeneratorReturnCatcher(
            self._solver_proc.rollout(x0, goal, stop_fn)
        )
        for chunk in gen:
            if chunk is None:  # solver busy: let other callbacks run
                await self._sleep(0.05)
                continue
            self._handle_rollout_chunk(chunk, traj_id, all_poses, goal_handle)

        terminal: RolloutResult = gen.value
        if terminal.status == "success":
            self.get_logger().info(
                f"Offline rollout traj_id={traj_id}: {terminal.message}"
            )
            self._publish_cumulative_path(all_poses)
        else:
            self.get_logger().warn(
                f"Offline rollout traj_id={traj_id}: {terminal.message}"
            )
        return terminal.status

    def _handle_rollout_chunk(
        self,
        chunk: RolloutChunk,
        traj_id: int,
        all_poses: list[np.ndarray],
        goal_handle,
    ):
        """Process one RolloutChunk: log diagnostics, publish feedback and path."""
        if self._diag_logger is not None and chunk.costates.shape[0] > 0:
            lam_th_0, lam_om_0, alpha_cmd_0 = compute_diag_values(
                chunk.costates[0], self.cfg
            )
            # Diag CSV keeps its body-space schema: omega / v columns are
            # the body equivalents of the PUBLISHED wheel commands.
            v_cmd, om_cmd = self.cfg.wheels_to_body(
                chunk.wheel_cmds[:, 0], chunk.wheel_cmds[:, 1]
            )
            self._diag_logger.log_plan(
                traj_id=traj_id,
                chunk=chunk.chunk_idx,
                thetas_deg=np.degrees(chunk.poses[:, 2]),
                omegas=om_cmd,
                vs=v_cmd,
                lam_th_0=lam_th_0,
                lam_om_0=lam_om_0,
                alpha_cmd_0=alpha_cmd_0,
            )
        _t0 = perf_counter()
        self._publish_chunk_feedback(goal_handle, traj_id, chunk)
        _fb_ms = (perf_counter() - _t0) * 1e3

        all_poses.append(chunk.poses)
        _t1 = perf_counter()
        self._publish_cumulative_path(all_poses)
        _path_ms = (perf_counter() - _t1) * 1e3

        self.get_logger().info(
            f"chunk {chunk.chunk_idx} published:"
            f"  feedback={_fb_ms:.0f}ms"
            f"  cum_path={_path_ms:.0f}ms"
            f"  poses_total={sum(p.shape[0] for p in all_poses)}"
        )

    # ---------------- Publishing ----------------

    def _publish_cumulative_path(self, all_poses: list[np.ndarray]):
        """Offline-mode cumulative trajectory publication for visualization."""
        if not all_poses:
            return
        now = self.get_clock().now().to_msg()
        path = Path()
        path.header.stamp = now
        path.header.frame_id = self.node_cfg.map_frame
        for block in all_poses:
            for k in range(block.shape[0]):
                pose = PoseStamped()
                pose.header.stamp = now
                pose.header.frame_id = self.node_cfg.map_frame
                pose.pose.position.x = float(block[k, 0])
                pose.pose.position.y = float(block[k, 1])
                yaw = float(block[k, 2])
                pose.pose.orientation.z = float(np.sin(yaw / 2.0))
                pose.pose.orientation.w = float(np.cos(yaw / 2.0))
                path.poses.append(pose)
        self._traj_pub.publish(path)

    def _publish_chunk_feedback(
        self,
        goal_handle,
        traj_id: int,
        chunk: RolloutChunk,
    ):
        """Emit one trajectory chunk as PlanToGoal action feedback.

        All arrays in the chunk are parallel (row i = tick i):
          wheel_cmds (N, 2) -- published per-side setpoints [w_l, w_r]
          accels     (N, 2) -- BVP-optimal wheel accelerations [a_l*, a_r*]
          poses      (N, 3) -- planned pose [px, py, theta]
          costates   (N, 5) -- PMP costates [lx, ly, lth, lwl, lwr]

        Empty chunks are silently skipped: there's no "is_final" flag in
        the action feedback (the action result signals end-of-trajectory),
        so an empty feedback message would carry no information. The
        intra-segment-hit case in rollout_generator guards against
        ever yielding an empty truncation.
        """
        if chunk.wheel_cmds.shape[0] == 0:
            return
        fb = PlanToGoal.Feedback()
        fb.trajectory_id = int(traj_id)
        fb.chunk_index = int(chunk.chunk_idx)
        fb.dt = float(chunk.dt_sample)
        # tolist() because rosidl-generated message slots for float32[]
        # expect a Python list (or array.array), not an ndarray.
        p32 = chunk.poses.astype(np.float32)
        w32 = chunk.wheel_cmds.astype(np.float32)
        a32 = chunk.accels.astype(np.float32)
        l32 = chunk.costates.astype(np.float32)
        fb.pose_x = p32[:, 0].tolist()
        fb.pose_y = p32[:, 1].tolist()
        fb.pose_theta = p32[:, 2].tolist()
        fb.wheel_left = w32[:, 0].tolist()
        fb.wheel_right = w32[:, 1].tolist()
        fb.accel_left = a32[:, 0].tolist()
        fb.accel_right = a32[:, 1].tolist()
        fb.lam_x = l32[:, 0].tolist()
        fb.lam_y = l32[:, 1].tolist()
        fb.lam_theta = l32[:, 2].tolist()
        fb.lam_wheel_left = l32[:, 3].tolist()
        fb.lam_wheel_right = l32[:, 4].tolist()
        goal_handle.publish_feedback(fb)

    # ---------------- PMP introspection (for evaluation) ----------------

    def extract_costates(self) -> Optional[np.ndarray]:
        """Return the last costate trajectory (m, 5):
        lambda_x, lambda_y, lambda_th, lambda_wl, lambda_wr."""
        return self._solver._last_costate

    def extract_predicted_trajectory(self) -> Optional[np.ndarray]:
        """Return the last optimal state trajectory (m, 5):
        px, py, theta, w_l, w_r."""
        return self._solver._last_state
