"""Gazebo-backed Bridge for the RL corrector env (the real training backend).

This is the GazeboBridge half of the bridge.py contract (the other half is the
pure KinematicBridge). The env is unchanged: it calls reset()/step()/close() and
gets back a StateReading, exactly as with the kinematic bridge.

WHY TWO TRANSPORTS
------------------
* gz.transport (in-process, gz.msgs10): GROUND-TRUTH pose, teleport, deterministic
  stepping, terrain spawn/remove. We must use Gazebo's true model pose for the
  reward, NOT /odom: /odom is *wheel* odometry, and on the slip patches we train
  for the wheels spin while the robot stays put -- wheel odometry would report
  phantom progress and hide the very slip the policy exists to correct.
* rclpy: publish the 4-wheel command to the ros2_control velocity controller and
  read the body twist (v, omega) from /odom. The twist is a *rate*, so it is
  teleport-safe even though the integrated /odom pose is not; we only borrow the
  twist, never the pose.

RUNTIME-CONFIRMATION ITEMS (the plan flags these; verify on a live sim with
`gz service -l` / `gz topic -l`):
  * world name (default "ordjo_world", from worlds/ordjo_world.world)
  * model name (spawned "scout_mini"; -allow_renaming may suffix it -> pass
    model_name= explicitly if so)
  * service names below follow the gz-sim convention /world/<world>/<verb>.

STEPPING
--------
Two modes (deterministic= flag):
  * wall-clock (default; env Phase 2): sim runs real-time (`gz sim -r`); step()
    publishes the command, then spins rclpy for control_dt so /odom and the gz
    pose callback update. Throughput ~= real time.
  * deterministic (env Phase 4 / training): sim paused; step() multi_steps the
    world by round(control_dt / physics_step) ticks. Reproducible and decoupled
    from wall-clock, so headless + rtf>1 gives throughput. gz_ros2_control runs
    inside the sim loop, so stepping the world also advances the controllers.
"""

import math
import time
from typing import List, Optional, Tuple

from .bridge import StateReading

# gz transport / messages. Imported at module load: this file is only imported
# when someone actually wants the Gazebo backend (env stays import-light).
import gz.transport13 as gz_transport
from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.world_control_pb2 import WorldControl
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.entity_factory_pb2 import EntityFactory
from gz.msgs10.entity_pb2 import Entity

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray

# Controller joint order (wheel_velocity_controller.yaml) == StateReading wheel
# order == [front_left, rear_left, front_right, rear_right] == [fl, rl, fr, rr].
_WHEEL_NAMES = ("front_left_wheel", "rear_left_wheel",
                "front_right_wheel", "rear_right_wheel")

# Standard ROS control topic and localization topic (agx_bringup Topics).
_WHEEL_COMMANDS_TOPIC = "/wheel_velocity_controller/commands"
_ODOM_TOPIC = "/odom"
_JOINT_STATES_TOPIC = "/joint_states"


def _yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    """Planar yaw from a quaternion (z-up)."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class _BridgeNode(Node):
    """rclpy side: publishes wheel commands, caches the latest /odom twist and
    (optionally) per-wheel speeds. Pose is NOT taken from here -- see module doc."""

    def __init__(self, use_wheel_speeds: bool) -> None:
        super().__init__("rl_corrector_gz_bridge")
        self._use_wheel_speeds = use_wheel_speeds

        # Match the forward command controller's QoS (reliable/volatile/keep-last)
        # so no setpoint is dropped, mirroring the deployed corrector node.
        cmd_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self._cmd_pub = self.create_publisher(
            Float64MultiArray, _WHEEL_COMMANDS_TOPIC, cmd_qos
        )
        self.create_subscription(Odometry, _ODOM_TOPIC, self._on_odom, 10)
        if use_wheel_speeds:
            self.create_subscription(
                JointState, _JOINT_STATES_TOPIC, self._on_joints, 10
            )

        self.v: float = 0.0
        self.omega: float = 0.0
        self.wheel_speeds: Optional[List[float]] = [0.0] * 4 if use_wheel_speeds else None
        self._odom_seen = False

    def publish_wheels(self, wheels) -> None:
        msg = Float64MultiArray()
        msg.data = [float(w) for w in wheels]
        self._cmd_pub.publish(msg)

    def _on_odom(self, msg: Odometry) -> None:
        # Body-frame twist only (rate -> teleport-safe). Pose is ignored here.
        self.v = float(msg.twist.twist.linear.x)
        self.omega = float(msg.twist.twist.angular.z)
        self._odom_seen = True

    def _on_joints(self, msg: JointState) -> None:
        idx = {n: i for i, n in enumerate(msg.name)}
        if all(n in idx for n in _WHEEL_NAMES):
            self.wheel_speeds = [float(msg.velocity[idx[n]]) for n in _WHEEL_NAMES]

    @property
    def odom_seen(self) -> bool:
        return self._odom_seen


class GazeboBridge:
    """Drives one Gazebo instance for the RL corrector env."""

    def __init__(
        self,
        cfg,
        world_name: str = "ordjo_world",
        model_name: str = "scout_mini",
        physics_step: float = 0.01,
        deterministic: bool = False,
        reset_z: float = 0.20,
        settle_steps: int = 5,
        service_timeout_ms: int = 3000,
        spin_warmup_s: float = 10.0,
    ) -> None:
        self.cfg = cfg
        self.world = world_name
        self.model = model_name
        self.physics_step = float(physics_step)
        self.deterministic = bool(deterministic)
        self.reset_z = float(reset_z)
        self.settle_steps = int(settle_steps)
        self.timeout_ms = int(service_timeout_ms)
        # Short timeout for the flaky-ack world services (set_pose/control): the
        # command executes fast; we cap the wait so a false-negative ack can't
        # stall a step/reset for the full service timeout. See _set_pose.
        self._ack_ms = min(800, int(service_timeout_ms))

        # gz-sim service endpoints (convention: /world/<world>/<verb>).
        self._svc_set_pose = f"/world/{world_name}/set_pose"
        self._svc_control = f"/world/{world_name}/control"
        self._svc_create = f"/world/{world_name}/create"
        self._svc_remove = f"/world/{world_name}/remove"
        # pose/info (NOT dynamic_pose/info): it is published for ALL entities at
        # a steady rate, so the robot pose stays fresh even when it is momentarily
        # at rest (dynamic_pose only carries *moving* entities), and it preserves
        # entity names so we can match the model. See _set_pose for the ack quirk.
        self._topic_pose = f"/world/{world_name}/pose/info"

        # --- gz transport: ground-truth pose + world services ---
        self._gz = gz_transport.Node()
        self._pose_xyth: Optional[Tuple[float, float, float]] = None
        self._gz.subscribe(Pose_V, self._topic_pose, self._on_pose)

        # --- rclpy: command out + twist in ---
        if not rclpy.ok():
            rclpy.init()
        self._node = _BridgeNode(use_wheel_speeds=cfg.use_wheel_speeds)
        self._exec = SingleThreadedExecutor()
        self._exec.add_node(self._node)

        # Names of terrain patches we spawned, so we can remove them next reset.
        self._terrain_models: List[str] = []

        # Wait for the first ground-truth pose + odom so step 0 is valid.
        self._wait_ready(spin_warmup_s)
        if self.deterministic:
            self._world_control(pause=True)

    # ------------------------------------------------------------------
    # gz callbacks / helpers
    # ------------------------------------------------------------------

    def _on_pose(self, msg: Pose_V) -> None:
        """Cache the robot's ground-truth world pose from pose/info.

        pose/info lists every entity (ground/sun/links/visuals); the top-level
        model pose has name == model_name. Match the model exactly.
        """
        for p in msg.pose:
            if p.name == self.model:
                q = p.orientation
                self._pose_xyth = (
                    p.position.x, p.position.y,
                    _yaw_from_quat(q.x, q.y, q.z, q.w),
                )
                return

    def _spin(self, duration_s: float) -> None:
        """Spin rclpy for a wall-clock duration (delivers cmd flush + /odom)."""
        end = time.monotonic() + duration_s
        while time.monotonic() < end:
            self._exec.spin_once(timeout_sec=max(0.0, end - time.monotonic()))

    def _wait_ready(self, timeout_s: float) -> None:
        end = time.monotonic() + timeout_s
        while time.monotonic() < end:
            self._exec.spin_once(timeout_sec=0.05)
            if self._pose_xyth is not None and self._node.odom_seen:
                return
        missing = []
        if self._pose_xyth is None:
            missing.append(f"ground-truth pose on {self._topic_pose}")
        if not self._node.odom_seen:
            missing.append(f"twist on {_ODOM_TOPIC}")
        raise RuntimeError(
            "GazeboBridge timed out waiting for: " + ", ".join(missing)
            + ". Is the sim up (gz_sim + sim_control) and the model named "
            f"'{self.model}' in world '{self.world}'?"
        )

    # NOTE on the gz-transport ack quirk: while a pose/info subscription is live
    # in this process, the world services (set_pose/control) reliably *execute*
    # but their Boolean response intermittently fails to arrive, so request()
    # returns ok=False even though the command took effect. We therefore use a
    # short ack timeout (don't block the full default on a false negative) and
    # verify side effects by reading ground-truth state, never by trusting `ok`.

    def _world_control(self, pause: Optional[bool] = None, multi_step: int = 0) -> None:
        req = WorldControl()
        if pause is not None:
            req.pause = pause
        if multi_step > 0:
            req.multi_step = multi_step
        # Best-effort: the step/pause executes regardless of the (flaky) ack.
        self._gz.request(self._svc_control, req, WorldControl, Boolean, self._ack_ms)

    def _set_pose(self, x: float, y: float, z: float, yaw: float,
                  tol: float = 0.05, tries: int = 4) -> bool:
        """Teleport and CONFIRM by ground-truth pose (the ack is unreliable).

        Returns True once the model's measured (x, y) is within `tol` of target.
        """
        req = Pose()
        req.name = self.model
        req.position.x, req.position.y, req.position.z = float(x), float(y), float(z)
        req.orientation.x = 0.0
        req.orientation.y = 0.0
        req.orientation.z = math.sin(yaw / 2.0)
        req.orientation.w = math.cos(yaw / 2.0)
        for _ in range(tries):
            self._gz.request(self._svc_set_pose, req, Pose, Boolean, self._ack_ms)
            # Confirm via pose/info. CRITICAL in deterministic mode: the sim is
            # paused, and pose/info only republishes when the world ADVANCES, so a
            # bare poll would keep reading the stale pre-teleport pose forever (and
            # fail once the robot has driven away from spawn). Step one physics
            # tick per poll so the post-teleport pose actually reaches the
            # callback; the tick also brakes residual velocity (wheels latched at
            # zero), and re-requesting set_pose each try yanks the body back to
            # target so it converges within tol.
            deadline = time.monotonic() + 0.5
            while time.monotonic() < deadline:
                if self.deterministic:
                    self._world_control(multi_step=1)
                self._exec.spin_once(timeout_sec=0.02)
                p = self._pose_xyth
                if p is not None and math.hypot(p[0] - x, p[1] - y) < tol:
                    return True
        return False

    def _advance(self, dt: float) -> None:
        """Advance the sim by dt under the active stepping mode, then refresh
        rclpy-side state. The gz pose callback runs on its own thread."""
        if self.deterministic:
            n = max(1, int(round(dt / self.physics_step)))
            self._world_control(multi_step=n)
            # Let the pose callback and /odom catch up post-step.
            self._spin(0.0 if n == 0 else min(0.05, dt))
        else:
            self._spin(dt)

    # ------------------------------------------------------------------
    # Bridge contract
    # ------------------------------------------------------------------

    def reset(self, start_pose, terrain=None) -> StateReading:
        x, y, th = (float(v) for v in start_pose)

        # 1. Halt the wheels so the teleport doesn't carry old motion through.
        self._node.publish_wheels([0.0, 0.0, 0.0, 0.0])
        self._spin(0.05)

        # 2. Swap terrain (Phase 3); no-op when terrain is None.
        self._apply_terrain(terrain, near=(x, y))

        # 3. Teleport to the episode start (slightly raised, then settle down).
        if not self._set_pose(x, y, self.reset_z, th):
            raise RuntimeError(f"set_pose failed on {self._svc_set_pose}")

        # 4. Settle: zero command for a few steps so residual twist damps out
        #    (set_pose moves the body but does not zero its velocity).
        for _ in range(self.settle_steps):
            self._node.publish_wheels([0.0, 0.0, 0.0, 0.0])
            self._advance(self.cfg.control_dt)

        return self._read_state()

    def step(self, wheels, dt: float) -> StateReading:
        self._node.publish_wheels(wheels)
        self._advance(dt)
        return self._read_state()

    def close(self) -> None:
        try:
            self._node.publish_wheels([0.0, 0.0, 0.0, 0.0])
            self._spin(0.02)
            if self.deterministic:
                self._world_control(pause=False)  # leave the sim runnable
        finally:
            self._remove_terrain()
            self._exec.remove_node(self._node)
            self._node.destroy_node()

    # ------------------------------------------------------------------
    # State + terrain
    # ------------------------------------------------------------------

    def _read_state(self) -> StateReading:
        # Drain any pending rclpy callbacks for the freshest twist.
        self._exec.spin_once(timeout_sec=0.0)
        if self._pose_xyth is None:
            raise RuntimeError("GazeboBridge lost the ground-truth pose stream.")
        return StateReading(
            pose=self._pose_xyth,
            v=self._node.v,
            omega=self._node.omega,
            wheel_speeds=self._node.wheel_speeds,
            contact=False,  # Phase: contact sensor wired later (optional hardening).
        )

    def _apply_terrain(self, terrain, near) -> None:
        """Remove last episode's patches and spawn this episode's. `terrain` is a
        list of patch dicts (see spawn_surface_patches schema) or None."""
        self._remove_terrain()
        if not terrain:
            return
        from .terrain import patch_sdf  # lazy: keeps gz-only deps off the hot path
        for idx, patch in enumerate(terrain):
            name = patch.get("name") or f"rl_patch_{idx}"
            sdf = patch_sdf(patch, name)
            req = EntityFactory()
            req.sdf = sdf
            req.name = name
            req.pose.position.x = float(patch.get("x", near[0]))
            req.pose.position.y = float(patch.get("y", near[1]))
            req.pose.position.z = float(patch.get("z", 0.001))
            # Best-effort + track regardless: like set_pose/control the create
            # executes even when the ack is a false negative, so record the name
            # so it is removed next reset either way.
            self._gz.request(self._svc_create, req, EntityFactory, Boolean, self._ack_ms)
            self._terrain_models.append(name)

    def _remove_terrain(self) -> None:
        for name in self._terrain_models:
            req = Entity()
            req.name = name
            req.type = Entity.MODEL
            self._gz.request(self._svc_remove, req, Entity, Boolean, self._ack_ms)
        self._terrain_models = []
