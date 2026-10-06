"""Supervisory layer node for the Nav2 and GMPC arms.

Wraps SupervisorCore and the baked-map hull clearance (walls.py) so other
stacks get the same checks as ours. With supervisor:=true, collision_monitor
and GMPC publish to cmd_in_topic (/cmd_vel_nav_out) and this node becomes the
only writer of /cmd_vel. Commands are relayed scaled by speed_scale, or as an
explicit zero every tick while stopped, since downstream latches the last
command. Pose is TF map -> base_link. In mode=navigate the reference is Nav2's
/plan (periodic re-plans to the same goal are ignored); in mode=follow_path it
is /supervisor/plan, the exact path sent to FollowPath.

On a trigger (one in flight, replan_cooldown apart, max_replans per goal) it
cancels and re-sends NavigateToPose, or in follow_path computes a new path with
ComputePathToPose and sends it to FollowPath.

While stopped after a re-plan, a command passes only if it opens clearance at
the pose predicted escape_horizon seconds ahead. In the slow band a command
that does not close clearance passes unscaled.

Outputs: /supervisor/state (Decision.as_array() plus [replans, stops,
replan_in_flight, applied_scale]), /supervisor/terminal (JSON, TRANSIENT_LOCAL:
replans, stops, last_reason, goals_sent, goal_status, goal_error_code, retimes)
and /supervisor/retime (plan index at each waypoint). Uses a
SingleThreadedExecutor with async callbacks only.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from action_msgs.srv import CancelGoal
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav2_msgs.action import ComputePathToPose, FollowPath, NavigateToPose
from nav_msgs.msg import Path
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Float64MultiArray, Int32, String
from tf2_ros import Buffer, TransformListener

from agx_planning.supervisor.core import Decision, SupervisorConfig, SupervisorCore
from agx_planning.supervisor.walls import SignedWallDistance
from agx_planning.utils import declare_and_load_dataclass

LATCHED = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE,
                     durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)

_STATUS = {GoalStatus.STATUS_SUCCEEDED: "succeeded", GoalStatus.STATUS_ABORTED: "aborted",
           GoalStatus.STATUS_CANCELED: "canceled"}


@dataclass
class SupervisorNodeConfig:
    mode: str = "navigate"                       # navigate | follow_path
    cmd_in_topic: str = "/cmd_vel_nav_out"
    cmd_out_topic: str = "/cmd_vel"
    plan_topic: str = ""                         # "" -> /plan (navigate) or /supervisor/plan
    map_yaml: str = ""                           # baked map; "" disables the wall layer
    map_frame: str = "map"
    base_frame: str = "base_link"
    rate: float = 20.0                           # [Hz] decision tick
    navigate_action: str = "navigate_to_pose"
    follow_path_action: str = "follow_path"
    compute_path_action: str = "compute_path_to_pose"
    planner_id: str = "GridBased"
    controller_id: str = "FollowPath"
    goal_checker_id: str = "general_goal_checker"
    progress_checker_id: str = "progress_checker"
    replan_cooldown: float = 2.0                 # [s] sim time between re-plans
    max_replans: int = 20                        # per goal
    new_goal_dist: float = 0.5                   # [m] /plan end moved this far = new goal
    escape_horizon: float = 0.3                  # [s] look-ahead for the escape check


def _yaw(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _path_arrays(msg: Path):
    poses = np.array([[p.pose.position.x, p.pose.position.y, _yaw(p.pose.orientation)]
                      for p in msg.poses], dtype=float)
    st = np.array([p.header.stamp.sec + p.header.stamp.nanosec * 1e-9 for p in msg.poses])
    times = st if (len(st) > 1 and np.all(np.diff(st) > 0)) else None
    return poses, times


def _pose_msg(frame: str, x: float, y: float, yaw: float, stamp) -> PoseStamped:
    ps = PoseStamped()
    ps.header.frame_id = frame
    ps.header.stamp = stamp
    ps.pose.position.x = float(x)
    ps.pose.position.y = float(y)
    ps.pose.orientation.z = math.sin(yaw / 2.0)
    ps.pose.orientation.w = math.cos(yaw / 2.0)
    return ps


class SupervisorNode(Node):
    def __init__(self):
        super().__init__("supervisor_node")
        self.cfg = declare_and_load_dataclass(self, SupervisorNodeConfig())
        core_cfg = declare_and_load_dataclass(self, SupervisorConfig(enabled=True), prefix="core.")
        if self.cfg.mode not in ("navigate", "follow_path"):
            raise ValueError(f"mode must be navigate|follow_path, got {self.cfg.mode!r}")
        self.core = SupervisorCore(core_cfg)
        self.walls = None
        if self.cfg.map_yaml:
            self.walls = SignedWallDistance.from_map_yaml(self.cfg.map_yaml)
            self.get_logger().info(f"wall layer on: {self.cfg.map_yaml}")
        else:
            self.get_logger().warn("map_yaml empty: wall layer OFF (clearance = inf)")

        self.cmd_pub = self.create_publisher(TwistStamped, self.cfg.cmd_out_topic, 10)
        self.state_pub = self.create_publisher(Float64MultiArray, "/supervisor/state", 10)
        self.term_pub = self.create_publisher(String, "/supervisor/terminal", LATCHED)
        self.retime_pub = self.create_publisher(Int32, "/supervisor/retime", 10)
        self.create_subscription(TwistStamped, self.cfg.cmd_in_topic, self._on_cmd, 10)
        plan_topic = self.cfg.plan_topic or (
            "/plan" if self.cfg.mode == "navigate" else "/supervisor/plan")
        self.create_subscription(Path, plan_topic, self._on_plan,
                                 LATCHED if self.cfg.mode == "follow_path" else 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)

        if self.cfg.mode == "navigate":
            self.nav_client = ActionClient(self, NavigateToPose, self.cfg.navigate_action)
            self.cancel_client = self.create_client(
                CancelGoal, f"/{self.cfg.navigate_action.strip('/')}/_action/cancel_goal")
        else:
            self.path_client = ActionClient(self, ComputePathToPose, self.cfg.compute_path_action)
            self.follow_client = ActionClient(self, FollowPath, self.cfg.follow_path_action)

        self.has_plan = False
        self.goal = None                 # (x, y, yaw) of the plan's end
        self.await_plan = False          # install the next /plan (after our re-plan)
        self.pose = None                 # (x, y, yaw)
        self.clearance = math.inf
        self.dec: Decision | None = None
        self.stop = False
        self.scale = 1.0
        self.applied = 1.0
        self.in_flight = False
        self.replan_since_stop = False
        self.last_replan_t = -math.inf
        self.goal_replans = 0
        self._latest_gh = None
        self.term = {"replans": 0, "stops": 0, "last_reason": "", "goals_sent": 0,
                     "goal_status": "none", "goal_error_code": None, "retimes": 0}
        self._publish_terminal()
        self.timer = self.create_timer(1.0 / self.cfg.rate, self._tick)
        self.get_logger().info(
            f"supervisor up: mode={self.cfg.mode} plan={plan_topic} "
            f"{self.cfg.cmd_in_topic} -> {self.cfg.cmd_out_topic}")

    # ---- helpers ------------------------------------------------------
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _lookup_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(self.cfg.map_frame, self.cfg.base_frame,
                                                 rclpy.time.Time())
        except Exception:  # noqa: BLE001 -- tf2's exception zoo
            return None
        t = tf.transform.translation
        return (t.x, t.y, _yaw(tf.transform.rotation))

    def _clear(self, x, y, yaw) -> float:
        return self.walls.body_clearance(x, y, yaw) if self.walls is not None else math.inf

    def _publish_cmd(self, v: float, w: float, src: TwistStamped | None = None) -> None:
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = src.header.frame_id if src is not None else self.cfg.base_frame
        msg.twist.linear.x = float(v)
        msg.twist.angular.z = float(w)
        self.cmd_pub.publish(msg)

    def _publish_terminal(self) -> None:
        self.term_pub.publish(String(data=json.dumps(self.term)))

    def _set_term(self, **kw) -> None:
        changed = any(self.term.get(k) != v for k, v in kw.items())
        self.term.update(kw)
        if changed:
            self._publish_terminal()

    # ---- inputs -------------------------------------------------------
    def _install(self, poses, times) -> None:
        self.core.set_plan(poses, times, t_now=self._now())
        self.has_plan = True
        self.goal = tuple(poses[-1])
        self.get_logger().info(
            f"plan installed: {len(poses)} poses, {self.core.n_waypoints} waypoints, "
            f"{'stamped' if times is not None else 'unstamped'}")

    def _on_plan(self, msg: Path) -> None:
        if not msg.poses:
            return
        poses, times = _path_arrays(msg)
        if self.cfg.mode == "follow_path":
            self.goal_replans = 0
            self._install(poses, times)
            return
        end = poses[-1]
        new_goal = self.goal is None or math.hypot(end[0] - self.goal[0],
                                                   end[1] - self.goal[1]) > self.cfg.new_goal_dist
        if new_goal:
            self.goal_replans = 0
        if new_goal or self.await_plan or not self.has_plan:
            self.await_plan = False
            self._install(poses, times)

    def _on_cmd(self, msg: TwistStamped) -> None:
        v, w = msg.twist.linear.x, msg.twist.angular.z
        d1 = self._predicted_clearance(v, w)
        if self.stop:
            if self.replan_since_stop and d1 > self.clearance + 1e-3:
                self.applied = 1.0
                self._publish_cmd(v, w, msg)   # opens clearance: let it out
            else:
                self.applied = 0.0
                self._publish_cmd(0.0, 0.0, msg)
            return
        # The slow band brakes APPROACH only: a command that does not close
        # clearance passes unscaled. Without this a robot spawned (or stopped)
        # just above stop_dist has scale ~0 and is pinned there for good --
        # the first smoke (G1, plan 00369) started at 0.150 m and never moved.
        s = 1.0 if d1 >= self.clearance - 1e-3 else self.scale
        self.applied = s
        self._publish_cmd(v * s, w * s, msg)

    def _predicted_clearance(self, v: float, w: float) -> float:
        """Hull clearance escape_horizon seconds along (v, w); -inf if unknown."""
        if self.pose is None or self.walls is None:
            return -math.inf
        x, y, yaw = self.pose
        h = self.cfg.escape_horizon
        return self._clear(x + v * h * math.cos(yaw), y + v * h * math.sin(yaw), yaw + w * h)

    # ---- decision tick ------------------------------------------------
    def _tick(self) -> None:
        pose = self._lookup_pose()
        if pose is None:
            return
        self.pose = pose
        x, y, yaw = pose
        t = self._now()
        self.clearance = self._clear(x, y, yaw)
        extra = [float(self.core.replans), float(self.core.stops), float(self.in_flight),
                 float(self.applied)]
        if self.has_plan:
            d = self.core.update(x, y, yaw, t, self.clearance)
            self.dec = d
            self.scale, stop = d.speed_scale, d.stop
            arr = d.as_array()
            if d.reached and d.retime_index is not None:
                self.retime_pub.publish(Int32(data=int(d.retime_index)))
                self._set_term(retimes=self.term["retimes"] + 1)
            if d.trigger:
                self._maybe_replan(t, d.reason)
        else:
            c = self.core.cfg
            span = max(c.slow_dist - c.stop_dist, 1e-9)
            fin = math.isfinite(self.clearance)
            self.scale = float(np.clip((self.clearance - c.stop_dist) / span, 0, 1)) if fin else 1.0
            stop = fin and self.clearance <= c.stop_dist
            arr = [math.nan] * len(Decision.FIELDS)
            i = Decision.FIELDS.index
            arr[i("clearance")] = float(self.clearance)
            arr[i("speed_scale")] = self.scale
            arr[i("stop")] = float(stop)
        if stop and not self.stop:
            self.replan_since_stop = False
            self.get_logger().warn(f"STOP: hull clearance {self.clearance:.3f} m")
        self.stop = stop
        if stop:
            self._publish_cmd(0.0, 0.0)
        self._set_term(replans=self.core.replans, stops=self.core.stops)
        self.state_pub.publish(Float64MultiArray(data=[float(a) for a in arr] + extra))

    # ---- re-plan ------------------------------------------------------
    def _maybe_replan(self, t: float, reason: str) -> None:
        if self.in_flight or self.goal is None:
            return
        if t - self.last_replan_t < self.cfg.replan_cooldown:
            return
        if self.goal_replans >= self.cfg.max_replans:
            return
        self.in_flight = True
        self.last_replan_t = t
        self.goal_replans += 1
        self._set_term(last_reason=reason)
        self.get_logger().warn(f"re-plan ({reason}) #{self.goal_replans} at clearance "
                               f"{self.clearance:.3f} m")
        coro = self._replan_navigate() if self.cfg.mode == "navigate" else self._replan_follow()
        self.executor.create_task(self._guard(coro))

    async def _guard(self, coro) -> None:
        try:
            await coro
        except Exception as exc:  # noqa: BLE001 -- never let a re-plan kill the node
            self.get_logger().error(f"re-plan failed: {exc!r}")
            self._set_term(goal_status="replan_failed")
        finally:
            self.in_flight = False
            if self.stop:
                self.replan_since_stop = True

    async def _replan_navigate(self) -> None:
        if self.cancel_client.service_is_ready():
            await self.cancel_client.call_async(CancelGoal.Request())  # zero id = all goals
        gx, gy, gyaw = self.goal
        goal = NavigateToPose.Goal()
        goal.pose = _pose_msg(self.cfg.map_frame, gx, gy, gyaw, self.get_clock().now().to_msg())
        self.await_plan = True
        await self._send(self.nav_client, goal)

    async def _replan_follow(self) -> None:
        x, y, yaw = self.pose
        gx, gy, gyaw = self.goal
        stamp = self.get_clock().now().to_msg()
        req = ComputePathToPose.Goal()
        req.start = _pose_msg(self.cfg.map_frame, x, y, yaw, stamp)
        req.goal = _pose_msg(self.cfg.map_frame, gx, gy, gyaw, stamp)
        req.use_start = True
        req.planner_id = self.cfg.planner_id
        if not self.path_client.wait_for_server(timeout_sec=0.0):
            raise RuntimeError(f"{self.cfg.compute_path_action} not available")
        gh = await self.path_client.send_goal_async(req)
        if not gh.accepted:
            raise RuntimeError("ComputePathToPose rejected")
        res = await gh.get_result_async()
        path = res.result.path
        if res.status != GoalStatus.STATUS_SUCCEEDED or not path.poses:
            raise RuntimeError(f"ComputePathToPose status {res.status}, "
                               f"{len(path.poses)} poses")
        poses, _ = _path_arrays(path)
        poses[-1] = (gx, gy, gyaw)  # keep the original goal as the plan's end
        self._install(poses, None)
        fp = FollowPath.Goal()
        fp.path = path
        fp.controller_id = self.cfg.controller_id
        fp.goal_checker_id = self.cfg.goal_checker_id
        if hasattr(fp, "progress_checker_id"):
            fp.progress_checker_id = self.cfg.progress_checker_id
        await self._send(self.follow_client, fp)

    async def _send(self, client: ActionClient, goal) -> None:
        if not client.wait_for_server(timeout_sec=0.0):
            raise RuntimeError("action server not available")
        gh = await client.send_goal_async(goal)
        if not gh.accepted:
            self._set_term(goal_status="rejected")
            return
        self._latest_gh = gh
        self._set_term(goals_sent=self.term["goals_sent"] + 1, goal_status="active",
                       goal_error_code=None)
        # Do not hold in_flight for the whole drive: watch the result separately.
        self.executor.create_task(self._watch(gh))

    async def _watch(self, gh) -> None:
        res = await gh.get_result_async()
        if gh is self._latest_gh:  # a superseded goal's end is not the run's
            code = getattr(res.result, "error_code", None)
            self._set_term(goal_status=_STATUS.get(res.status, f"status_{res.status}"),
                           goal_error_code=None if code is None else int(code))


def main(args=None):
    rclpy.init(args=args)
    node = SupervisorNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        for _ in range(3):
            node._publish_cmd(0.0, 0.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
