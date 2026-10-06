#!/usr/bin/env python3
"""Is the ROS 2 stack actually up, and if not, which part is missing?

WHY THIS EXISTS. Every corrector number in this repo was measured through
`GazeboBridge`, which talks to Gazebo directly and bypasses the ROS graph
entirely. The runtime pipeline -- vector_field -> pmp_planner ->
runtime_corrector -> controllers -- is therefore the least exercised part of
the system, and it has known start-up races: a launch that "came up" can be
missing the map, the map->odom transform, or a subscriber on /goal_pose, and
the visible symptom is not an error but a robot that never moves, or a
'Timeout waiting for vector field'.

The fix for a race you cannot remove is to DETECT it cheaply and restart, so
this reports readiness as data rather than as a human squinting at a log:

    python3 tools/stack_ready.py --json          # one snapshot, exit 0 if ready
    python3 tools/stack_ready.py --wait 90       # block until ready, or fail

Exit codes: 0 ready, 1 not ready (details on stdout), 2 could not even init.
`--json` prints one object with a per-check breakdown, which is what an
automated caller should branch on -- the human-readable table is the default.

DESIGN NOTES.

* Checks are split into GRAPH checks (does a publisher/subscriber exist) and
  LIVENESS checks (is data actually flowing). Both are needed and they fail
  differently: a node that crashed after advertising leaves the graph check
  passing forever, while a node that is merely slow to start fails the graph
  check and recovers. Only the liveness checks can tell "the planner is alive"
  from "the planner registered and died".

* `/clock` liveness is checked by VALUE, not by message count. A paused or
  crashed Gazebo keeps a `/clock` publisher and, in some failure modes, keeps
  republishing the same stamp -- so counting messages says "alive" for a world
  that is not stepping. Requiring the value to ADVANCE is the only test that
  distinguishes them, and it is exactly the failure that makes a fixture sit
  forever with the robot motionless.

* Subscriber COUNTS on /goal_pose are a first-class check, not a nicety. A goal
  published before the subscribers have matched reaches only those already
  there: lose vector_field and you get 'Timeout waiting for vector field',
  lose runtime_corrector and nothing drives at all -- and neither prints
  anything at the time. This is documented in CLAUDE.md as a trap for humans
  publishing goals by hand; here it is a precondition that can be waited on.

* This node uses SYSTEM time deliberately (`use_sim_time` is NOT set). Its job
  includes deciding whether sim time is running, so a node whose own timeouts
  are driven by sim time would hang forever on exactly the failure it exists
  to detect.

* `--mode nav2` selects the nav2 stack's checks (lifecycle nodes ACTIVE, the
  navigate_to_pose action server, the /cmd_vel chain) instead of the vec-pmp
  ones; the default `--mode vec-pmp` keeps the original check set unchanged,
  because every corrector number in the repo was gated on exactly it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from nav_msgs.msg import OccupancyGrid, Odometry
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray
from tf2_msgs.msg import TFMessage


LATCHED = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)


@dataclass
class Check:
    name: str
    ok: bool = False
    detail: str = "not observed"
    required: bool = True


@dataclass
class Report:
    ready: bool
    checks: list = field(default_factory=list)
    waited_s: float = 0.0

    def to_json(self) -> str:
        return json.dumps(
            {
                "ready": self.ready,
                "waited_s": round(self.waited_s, 1),
                "checks": [
                    {"name": c.name, "ok": c.ok, "detail": c.detail, "required": c.required}
                    for c in self.checks
                ],
            },
            indent=2,
        )

    def to_text(self) -> str:
        lines = []
        for c in self.checks:
            mark = "ok  " if c.ok else ("FAIL" if c.required else "warn")
            lines.append(f"  [{mark}] {c.name:24} {c.detail}")
        head = "STACK READY" if self.ready else "STACK NOT READY"
        return f"{head}  (after {self.waited_s:.0f}s)\n" + "\n".join(lines)


class StackProbe(Node):
    def __init__(self, expect_planner: bool, mode: str = "vec-pmp", require_amcl: bool = False):
        super().__init__("stack_ready_probe")
        self.expect_planner = expect_planner
        self.mode = mode

        self._clock_stamps: list[float] = []
        self._map = None
        self._tf_frames: set[str] = set()
        self._joint_count = 0
        self._odom_count = 0
        self._field_count = 0
        self._amcl_pose_count = 0
        # One reusable client per lifecycle node, created lazily in
        # _lifecycle_state (service names are static per stack).
        self._lifecycle_clients: dict[str, object] = {}

        self.create_subscription(Clock, "/clock", self._on_clock, 10)
        self.create_subscription(OccupancyGrid, "/map", self._on_map, LATCHED)
        self.create_subscription(TFMessage, "/tf", self._on_tf, 50)
        self.create_subscription(TFMessage, "/tf_static", self._on_tf, LATCHED)
        self.create_subscription(JointState, "/joint_states", self._on_joints, 10)
        self.create_subscription(Odometry, "/odom", self._on_odom, 10)
        self.create_subscription(
            Float32MultiArray, "/vector_field/planner_data", self._on_field, 1
        )
        self.require_amcl = require_amcl
        if require_amcl:
            from geometry_msgs.msg import PoseWithCovarianceStamped

            # amcl's pose estimate, REQUIRED: it flows only once amcl has both
            # a scan and its initial pose. Before that, map->odom can exist but
            # the global costmap still reports "Timed out waiting for transform
            # base_link to map", and a goal sent then races localization.
            # (It was once subscribed as PoseStamped -- the wrong type, so it
            # silently never matched and this check could never pass.)
            self.create_subscription(
                PoseWithCovarianceStamped, "/amcl_pose", self._on_amcl_pose,
                QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                           reliability=QoSReliabilityPolicy.RELIABLE),
            )

    # --- callbacks: count and remember, never block ------------------------
    def _on_clock(self, msg: Clock) -> None:
        t = msg.clock.sec + msg.clock.nanosec * 1e-9
        self._clock_stamps.append(t)
        if len(self._clock_stamps) > 200:
            del self._clock_stamps[:100]

    def _on_map(self, msg: OccupancyGrid) -> None:
        self._map = msg

    def _on_tf(self, msg: TFMessage) -> None:
        for tr in msg.transforms:
            self._tf_frames.add(f"{tr.header.frame_id.lstrip('/')}->{tr.child_frame_id.lstrip('/')}")

    def _on_joints(self, _msg: JointState) -> None:
        self._joint_count += 1

    def _on_odom(self, _msg: Odometry) -> None:
        self._odom_count += 1

    def _on_field(self, _msg: Float32MultiArray) -> None:
        self._field_count += 1

    def _on_amcl_pose(self, _msg) -> None:
        self._amcl_pose_count += 1

    def _lifecycle_state(self, node_name: str) -> str | None:
        """Current lifecycle label of `node_name`, or None if unanswerable.

        Graph presence proves a node EXISTS; only the lifecycle state proves it
        is ACTIVE and therefore serving. The nav2 stack autostarts through
        nav2_lifecycle_manager, so a node stuck in 'unconfigured'/'inactive'
        (a crashed plugin load, a bad parameter file) looks alive on every
        topic count and still navigates nothing.
        """
        from lifecycle_msgs.srv import GetState

        cli = self._lifecycle_clients.get(node_name)
        if cli is None:
            cli = self.create_client(GetState, f"/{node_name}/get_state")
            self._lifecycle_clients[node_name] = cli
        if not cli.service_is_ready():
            return None
        fut = cli.call_async(GetState.Request())
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not fut.done():
            rclpy.spin_once(self, timeout_sec=0.05)
        try:
            return fut.result().current_state.label
        except Exception:  # noqa: BLE001 -- an unanswered probe is a missing state
            return None

    # --- the checks --------------------------------------------------------
    def evaluate(self) -> list:
        checks = []

        # Sim time must ADVANCE, not merely be published -- see module docstring.
        if len(self._clock_stamps) < 2:
            checks.append(Check("sim clock", False, f"{len(self._clock_stamps)} msgs on /clock"))
        else:
            span = self._clock_stamps[-1] - self._clock_stamps[0]
            checks.append(
                Check(
                    "sim clock",
                    span > 0.05,
                    f"advanced {span:.2f}s over {len(self._clock_stamps)} msgs"
                    if span > 0.05
                    else f"NOT advancing (span {span:.4f}s) -- world paused or dead",
                )
            )

        if self._map is None:
            checks.append(Check("map", False, "no /map received (latched publisher missing?)"))
        else:
            info = self._map.info
            checks.append(
                Check("map", True, f"{info.width}x{info.height} @ {info.resolution:.3f} m/px")
            )

        # map->odom is what `localization` provides; without it the corrector
        # has no pose in the planning frame and drives to wheel-odometry bias.
        has_map_odom = any(f.startswith("map->") for f in self._tf_frames)
        has_odom_base = any("odom->" in f for f in self._tf_frames)
        checks.append(
            Check(
                "tf map->odom",
                has_map_odom,
                "present" if has_map_odom else f"MISSING (saw {len(self._tf_frames)} transforms)",
            )
        )
        checks.append(Check("tf odom->base", has_odom_base, "present" if has_odom_base else "MISSING"))

        checks.append(
            Check(
                "joint_states",
                self._joint_count > 0,
                f"{self._joint_count} msgs" if self._joint_count else "silent -- controllers not up",
            )
        )
        checks.append(
            Check("odometry", self._odom_count > 0, f"{self._odom_count} msgs" if self._odom_count else "silent")
        )
        checks.append(
            Check(
                "wheel controller",
                self.count_subscribers("/wheel_velocity_controller/commands") >= 1,
                "subscriber present on the command topic",
            )
        )

        if self.require_amcl:
            # amcl pose: required (see the subscription note above).
            checks.append(
                Check(
                    "amcl_pose",
                    self._amcl_pose_count > 0,
                    f"{self._amcl_pose_count} msgs"
                    if self._amcl_pose_count
                    else "not yet published (no scan / not localized yet)",
                )
            )

        if self.mode == "nav2":
            return checks + self._nav2_checks()
        if self.mode == "gmpc":
            return checks + self._follow_path_checks("gmpc_controller")

        # --- vec-pmp mode (the original checks, unchanged) -------------------
        # Graph checks: who is listening for a goal, and who commands wheels.
        goal_subs = self.count_subscribers("/goal_pose")
        checks.append(
            Check(
                "goal_pose subscribers",
                goal_subs >= 2,
                f"{goal_subs} (want >=2: vector_field + runtime_corrector)",
            )
        )

        # QoS compatibility on /goal_pose. A TRANSIENT_LOCAL subscriber cannot
        # receive from a VOLATILE publisher, and the failure is silent on the
        # publishing side: the goal goes out, the robot drives, and the mismatched
        # node simply never hears anything. This was found by writing
        # tools/drive_goal.py with the wrong durability, where it presented as
        # "the run never finishes" rather than as a QoS problem. The whole stack
        # declares RELIABLE + VOLATILE depth 1 on this topic; anything else is a bug.
        try:
            durs = {
                ep.qos_profile.durability
                for ep in self.get_publishers_info_by_topic("/goal_pose")
                + self.get_subscriptions_info_by_topic("/goal_pose")
            }
            mixed = len(durs) > 1
            checks.append(
                Check(
                    "goal_pose QoS",
                    not mixed,
                    "consistent durability across all endpoints"
                    if not mixed
                    else f"MIXED durability {sorted(str(d) for d in durs)} -- "
                    "some endpoint will silently receive nothing",
                )
            )
        except Exception as exc:  # noqa: BLE001 -- introspection is best-effort
            checks.append(Check("goal_pose QoS", True, f"not checked ({exc})", required=False))

        # The planner only publishes once a goal exists, so its liveness is a
        # warning before a goal and a requirement after one.
        field_pubs = self.count_publishers("/vector_field/planner_data")
        checks.append(
            Check(
                "vector_field",
                field_pubs >= 1,
                f"{field_pubs} publisher(s), {self._field_count} field msgs",
                required=True,
            )
        )
        if self.expect_planner:
            checks.append(
                Check(
                    "planner_data flowing",
                    self._field_count > 0,
                    f"{self._field_count} msgs (a goal must be active for this)",
                )
            )
        return checks

    def _nav2_checks(self) -> list:
        """Readiness for the nav2 arms: lifecycle ACTIVE, action server up.

        These are the checks the vec-pmp ones cannot make. The nav2 stack
        autostarts through a lifecycle manager, so every failure mode that
        matters shows up as a node sitting in unconfigured/inactive while all
        its topics LOOK fine; and the thing a goal is sent to is the
        navigate_to_pose ACTION, whose graph presence is its own check.
        """
        checks = []
        for node_name in ("controller_server", "planner_server", "bt_navigator"):
            state = self._lifecycle_state(node_name)
            checks.append(
                Check(
                    f"{node_name} lifecycle",
                    state == "active",
                    f"state: {state}" if state else "get_state unanswered (node down?)",
                )
            )

        # The action server. send_goal/get_result are SERVICES in the ROS 2
        # action protocol (only status/feedback are topics), so a topic-graph
        # count of send_goal returns 0 forever and this check would fail a
        # perfectly healthy stack -- which is exactly what the first nav2
        # bring-up did. Query the node's visible service servers instead.
        srv_names = set()
        try:
            srv_names = {
                name
                for (name, _t) in self.get_service_names_and_types_by_node(
                    "bt_navigator", "/")
            }
        except Exception:  # noqa: BLE001 -- graph introspection is best-effort
            pass
        status_pubs = self.count_publishers("/navigate_to_pose/_action/status")
        send_goal_srv = "/navigate_to_pose/_action/send_goal" in srv_names
        checks.append(
            Check(
                "navigate_to_pose action",
                send_goal_srv and status_pubs >= 1,
                f"send_goal service {'present' if send_goal_srv else 'ABSENT'}, "
                f"{status_pubs} status pub(s)",
            )
        )

        # The command chain must be complete: something publishing /cmd_vel
        # (collision_monitor) and twist_to_wheels subscribed to it. Both arms'
        # motion funnels through this edge; a missing side here means the
        # robot would navigate beautifully and never move.
        cmd_vel_pubs = self.count_publishers("/cmd_vel")
        cmd_vel_subs = self.count_subscribers("/cmd_vel")
        checks.append(
            Check(
                "cmd_vel chain",
                cmd_vel_pubs >= 1 and cmd_vel_subs >= 1,
                f"{cmd_vel_pubs} pub(s), {cmd_vel_subs} sub(s) on /cmd_vel",
            )
        )

        return checks


    def _follow_path_checks(self, server_node: str) -> list:
        """Readiness for a bare follow_path tracker (the GMPC arm, agx_baselines).

        No lifecycle, no bt_navigator: the goal goes straight to a FollowPath
        action server owned by `server_node`, and its output must reach
        twist_to_wheels over /cmd_vel. Same two edges the nav2 checks verify,
        minus the nav2-only nodes.
        """
        checks = []
        srv_names = set()
        try:
            srv_names = {name for (name, _t) in
                         self.get_service_names_and_types_by_node(server_node, "/")}
        except Exception:  # noqa: BLE001 -- graph introspection is best-effort
            pass
        status_pubs = self.count_publishers("/follow_path/_action/status")
        send_goal_srv = "/follow_path/_action/send_goal" in srv_names
        checks.append(Check(
            "follow_path action",
            send_goal_srv and status_pubs >= 1,
            f"send_goal service {'present' if send_goal_srv else 'ABSENT'} on "
            f"{server_node}, {status_pubs} status pub(s)"))
        cmd_vel_pubs = self.count_publishers("/cmd_vel")
        cmd_vel_subs = self.count_subscribers("/cmd_vel")
        checks.append(Check(
            "cmd_vel chain", cmd_vel_pubs >= 1 and cmd_vel_subs >= 1,
            f"{cmd_vel_pubs} pub(s), {cmd_vel_subs} sub(s) on /cmd_vel"))
        return checks


def probe(timeout: float, settle: float, expect_planner: bool, mode: str = "vec-pmp",
          require_amcl: bool = False) -> Report:
    rclpy.init()
    node = StackProbe(expect_planner=expect_planner, mode=mode, require_amcl=require_amcl)
    t0 = time.monotonic()
    checks: list = []
    try:
        while True:
            # Spin in short slices so a wait is responsive and a snapshot is
            # still given `settle` seconds to hear the latched/periodic topics.
            deadline = time.monotonic() + settle
            while time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.1)
            checks = node.evaluate()
            ready = all(c.ok for c in checks if c.required)
            if ready or time.monotonic() - t0 >= timeout:
                return Report(ready=ready, checks=checks, waited_s=time.monotonic() - t0)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--wait", type=float, default=0.0,
                    help="keep probing until ready, up to this many seconds (default: one snapshot)")
    ap.add_argument("--settle", type=float, default=3.0,
                    help="seconds to listen before each evaluation")
    ap.add_argument("--expect-planner", action="store_true",
                    help="also require planner_data to be FLOWING (only true once a goal is active)")
    ap.add_argument("--mode", choices=["vec-pmp", "nav2", "gmpc"], default="vec-pmp",
                    help="which stack's readiness to check (vec-pmp keeps the "
                         "original checks exactly; nav2 requires the nav2 "
                         "lifecycle nodes ACTIVE, the navigate_to_pose action "
                         "server, and a complete /cmd_vel chain)")
    ap.add_argument("--require-amcl", action="store_true",
                    help="require /amcl_pose (localization:=amcl, either mode): "
                         "map->odom alone exists before amcl has localized")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    try:
        report = probe(timeout=args.wait, settle=args.settle,
                       expect_planner=args.expect_planner, mode=args.mode,
                       require_amcl=args.require_amcl)
    except Exception as exc:  # noqa: BLE001 -- a probe must never traceback at a caller
        print(json.dumps({"ready": False, "error": str(exc)}) if args.json else f"probe failed: {exc}")
        return 2

    print(report.to_json() if args.json else report.to_text())
    return 0 if report.ready else 1


if __name__ == "__main__":
    sys.exit(main())
