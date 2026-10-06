"""GMPC as a Nav2-shaped tracking controller for the comparison harness.

What it is. A ``nav2_msgs/action/FollowPath`` server named ``follow_path`` --
the same action, name and goal layout ``controller_server`` exposes -- so
``tools/compare_run.py`` sends it our PMP plan exactly as it does for the
``pmp-mppi`` / ``pmp-rpp`` arms (Type A in #33: the planner is held fixed at
our plan, only the tracker differs). The pose comes from TF ``map -> base_link``
(what Nav2's controller reads), odometry liveness from ``/odom/filtered``, and
the command goes out as a ``TwistStamped`` on ``/cmd_vel``, the topic Nav2's
velocity smoother feeds ``twist_to_wheels`` through. Nothing in this package is
on our stack's path; it is a passenger that is only ever launched by
``agx_baselines/launch/gmpc.launch.py``.

Reference timing. The path's pose stamps, when strictly increasing, are the
reference time law (compare_run stamps the PMP plan's own ``dt_sample`` grid,
so the arm tracks the plan AS A TRAJECTORY, which is what GMPC is for). An
unstamped path -- what a Nav2 planner would send -- gets a constant-speed time
law from ``cruise_speed``.

Terminal paths. Every one publishes an explicit zero twist first:
``twist_to_wheels`` and the joint controller both latch their last command,
so silence keeps the wheels turning (CLAUDE.md). Outcomes map onto the
FollowPath error codes compare_run already decodes: ``NONE`` on arrival
(within ``goal_tolerance_xy`` of the final pose once the reference has run
out), ``FAILED_TO_MAKE_PROGRESS`` if the hold phase times out short,
``NO_VALID_CONTROL`` on a solver failure or a tracking error beyond
``max_tracking_error``, ``TF_ERROR`` when the pose or odometry goes stale,
``INVALID_PATH`` for an empty path.

Executor. SingleThreaded, on purpose (CLAUDE.md: MultiThreadedExecutor
busy-spins under the 1 kHz sim clock). The control loop is a sim-time timer;
the action's execute callback is a coroutine that merely awaits the loop's
completion future, so nothing blocks the executor.
"""

from __future__ import annotations

import math

import numpy as np
import rclpy
from geometry_msgs.msg import TwistStamped
from nav2_msgs.action import FollowPath
from nav_msgs.msg import Odometry
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.task import Future
from tf2_ros import Buffer, TransformListener

from agx_baselines.gmpc.tracker import PlanTracker, wrap_angle

ERR = FollowPath.Result  # error-code constants


def _yaw(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class GmpcController(Node):
    def __init__(self):
        super().__init__("gmpc_controller")
        p = self.declare_parameters("", [
            ("control_rate", 20.0),
            ("horizon", 10),
            ("q_weights", [20000.0, 20000.0, 2000.0]),
            ("r_weight", 0.3),
            ("v_min", -0.5), ("v_max", 0.5),
            ("w_min", -1.5), ("w_max", 1.5),
            ("cruise_speed", 0.4),
            ("goal_tolerance_xy", 0.15),
            ("goal_hold_s", 10.0),
            ("max_tracking_error", 3.0),
            ("odom_timeout", 2.0),
            ("map_frame", "map"),
            ("base_frame", "base_link"),
            ("cmd_vel_topic", "/cmd_vel"),
            ("odom_topic", "/odom/filtered"),
        ])
        self.cfg = {x.name: x.value for x in p}
        self.dt = 1.0 / float(self.cfg["control_rate"])

        self.cmd_pub = self.create_publisher(TwistStamped, self.cfg["cmd_vel_topic"], 10)
        self.create_subscription(Odometry, self.cfg["odom_topic"], self._on_odom, 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)

        self._last_odom_t: float | None = None  # ROS clock seconds
        self._active = None  # dict with goal_handle, tracker, t0, future ...
        self.timer = self.create_timer(self.dt, self._tick)

        self.server = ActionServer(
            self, FollowPath, "follow_path",
            execute_callback=self._execute,
            goal_callback=self._on_goal,
            cancel_callback=lambda _h: CancelResponse.ACCEPT,
        )
        self.get_logger().info(
            f"GMPC up: dt={self.dt:.3f}s N={self.cfg['horizon']} "
            f"Q={self.cfg['q_weights']} R={self.cfg['r_weight']}")

    # -- helpers --------------------------------------------------------------
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_odom(self, _msg: Odometry) -> None:
        self._last_odom_t = self._now()

    def _pose(self):
        """(x, y, yaw, stamp_s) of base in map, or None."""
        try:
            tf = self.tf_buffer.lookup_transform(
                self.cfg["map_frame"], self.cfg["base_frame"], rclpy.time.Time())
        except Exception:  # noqa: BLE001 -- tf2 raises a zoo of exceptions
            return None
        t = tf.transform.translation
        st = tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9
        return (t.x, t.y, _yaw(tf.transform.rotation), st)

    def _publish(self, v: float, w: float) -> None:
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.cfg["base_frame"]
        msg.twist.linear.x = float(v)
        msg.twist.angular.z = float(w)
        self.cmd_pub.publish(msg)

    def _stop(self) -> None:
        # Explicit zero on EVERY terminal path; three times because the first
        # publish after a (re)match can be dropped and a latched non-zero
        # command keeps the wheels spinning.
        for _ in range(3):
            self._publish(0.0, 0.0)

    # -- action plumbing ------------------------------------------------------
    def _on_goal(self, _req) -> GoalResponse:
        if self._active is not None:
            self.get_logger().warn("follow_path goal rejected: one is already active")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    async def _execute(self, goal_handle):
        path = goal_handle.request.path
        result = FollowPath.Result()
        if len(path.poses) == 0:
            result.error_code = ERR.INVALID_PATH
            result.error_msg = "empty path"
            self._stop()
            goal_handle.abort()
            return result

        poses = np.array([[ps.pose.position.x, ps.pose.position.y, _yaw(ps.pose.orientation)]
                          for ps in path.poses], dtype=float)
        stamps = np.array([ps.header.stamp.sec + ps.header.stamp.nanosec * 1e-9
                           for ps in path.poses], dtype=float)
        times = stamps if (len(stamps) > 1 and np.all(np.diff(stamps) > 0)) else None
        try:
            tracker = PlanTracker(
                poses, times, self.dt,
                horizon=int(self.cfg["horizon"]),
                q=tuple(float(x) for x in self.cfg["q_weights"]),
                r=float(self.cfg["r_weight"]),
                v_bounds=(float(self.cfg["v_min"]), float(self.cfg["v_max"])),
                w_bounds=(float(self.cfg["w_min"]), float(self.cfg["w_max"])),
                cruise_speed=float(self.cfg["cruise_speed"]))
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"GMPC setup failed: {exc!r}")
            result.error_code = ERR.NO_VALID_CONTROL
            result.error_msg = f"setup: {exc!r}"
            self._stop()
            goal_handle.abort()
            return result

        self.get_logger().info(
            f"follow_path: {len(path.poses)} poses, "
            f"{'stamped' if times is not None else 'unstamped'} -> "
            f"{tracker.n} samples, {tracker.duration:.1f} s reference")
        done: Future = Future()
        self._active = {"gh": goal_handle, "tracker": tracker, "t0": self._now(),
                        "future": done, "hold_t0": None, "n_solves": 0,
                        "solve_s": 0.0, "max_err": 0.0}
        await done
        st = self._active
        self._active = None
        code, msg = done.result()
        self._stop()
        result.error_code = code
        result.error_msg = msg
        self.get_logger().info(
            f"follow_path terminal: code={code} '{msg}' solves={st['n_solves']} "
            f"mean_solve={1e3 * st['solve_s'] / max(1, st['n_solves']):.1f} ms "
            f"max_track_err={st['max_err']:.3f} m")
        if code == ERR.NONE:
            goal_handle.succeed()
        elif goal_handle.is_cancel_requested:
            goal_handle.canceled()
        else:
            goal_handle.abort()
        return result

    def _finish(self, code: int, msg: str) -> None:
        self._stop()
        fut = self._active["future"]
        if not fut.done():
            fut.set_result((code, msg))

    # -- control loop ---------------------------------------------------------
    def _tick(self) -> None:
        st = self._active
        if st is None or st["future"].done():
            return
        gh = st["gh"]
        if gh.is_cancel_requested:
            self._finish(ERR.UNKNOWN, "cancelled")
            return
        now = self._now()
        if self._last_odom_t is None or now - self._last_odom_t > self.cfg["odom_timeout"]:
            self._finish(ERR.TF_ERROR, "odometry stale")
            return
        pose = self._pose()
        if pose is None:
            # TF may lag a beat at start; fail only once it is stale for real.
            if now - st["t0"] > self.cfg["odom_timeout"]:
                self._finish(ERR.TF_ERROR, "no map->base transform")
            return
        x, y, yaw, _stamp = pose
        tracker: PlanTracker = st["tracker"]
        t = now - st["t0"]
        ref = tracker.reference_at(t)
        err = math.hypot(ref[0] - x, ref[1] - y)
        st["max_err"] = max(st["max_err"], err)
        if err > self.cfg["max_tracking_error"]:
            self._finish(ERR.NO_VALID_CONTROL, f"tracking error {err:.2f} m")
            return

        goal = tracker.final_pose()
        dist_goal = math.hypot(goal[0] - x, goal[1] - y)
        if t >= tracker.duration:
            if st["hold_t0"] is None:
                st["hold_t0"] = now
            if dist_goal <= self.cfg["goal_tolerance_xy"]:
                self._finish(ERR.NONE, "arrived")
                return
            if now - st["hold_t0"] > self.cfg["goal_hold_s"]:
                self._finish(ERR.FAILED_TO_MAKE_PROGRESS,
                             f"hold timeout, {dist_goal:.2f} m from goal")
                return

        try:
            v, w, solve_s = tracker.solve((x, y, wrap_angle(yaw)), t)
        except Exception as exc:  # noqa: BLE001 -- qpOASES / casadi failures
            self._finish(ERR.NO_VALID_CONTROL, f"solver: {exc!r}")
            return
        st["n_solves"] += 1
        st["solve_s"] += solve_s
        self._publish(v, w)

        fb = FollowPath.Feedback()
        fb.distance_to_goal = float(dist_goal)
        fb.speed = float(abs(v))
        gh.publish_feedback(fb)


def main(args=None):
    rclpy.init(args=args)
    node = GmpcController()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node._stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
