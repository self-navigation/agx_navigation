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
  * world name (default "rl_corrector", from worlds/rl_corrector.world)
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
from gz.msgs10.physics_pb2 import Physics

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState, Imu
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
from rosgraph_msgs.msg import Clock

# Controller joint order (wheel_velocity_controller.yaml) == StateReading wheel
# order == [front_left, rear_left, front_right, rear_right] == [fl, rl, fr, rr].
_WHEEL_NAMES = ("front_left_wheel", "rear_left_wheel",
                "front_right_wheel", "rear_right_wheel")

# Standard ROS control topic and localization topic (agx_bringup Topics).
_WHEEL_COMMANDS_TOPIC = "/wheel_velocity_controller/commands"
_ODOM_TOPIC = "/odom"
_JOINT_STATES_TOPIC = "/joint_states"
_CLOCK_TOPIC = "/clock"
_IMU_TOPIC = "/imu/data"


def _yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    """Planar yaw from a quaternion (z-up)."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class _BridgeNode(Node):
    """rclpy side: publishes wheel commands, caches the latest /odom twist and
    (optionally) per-wheel speeds. Pose is NOT taken from here -- see module doc."""

    def __init__(self, use_wheel_speeds: bool, use_imu: bool) -> None:
        super().__init__("rl_corrector_gz_bridge")
        self._use_wheel_speeds = use_wheel_speeds
        self._use_imu = use_imu

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
        if use_imu:
            # The on-robot IMU (bridged from gz). Read the same way deployment does
            # (ROS sensor_msgs/Imu), so the obs the policy sees matches train->deploy.
            # We keep the gz/odom split for pose/twist but take the IMU over ROS
            # because it is a *deployable* sensor, not privileged ground truth.
            self.create_subscription(Imu, _IMU_TOPIC, self._on_imu, 10)
        # Sim clock (bridged from gz by rl_clock_bridge). Deterministic stepping
        # gates each env step on this advancing by control_dt, so every step is a
        # true dt of SIM time -- the authoritative throttle, independent of how many
        # pose/info messages the paused-then-multi_stepped world happens to emit.
        self.create_subscription(Clock, _CLOCK_TOPIC, self._on_clock, 10)

        self.v: float = 0.0
        self.omega: float = 0.0
        self.wheel_speeds: Optional[List[float]] = [0.0] * 4 if use_wheel_speeds else None
        self.imu: Optional[Tuple[float, float, float]] = None
        self._odom_seen = False
        self.sim_time: Optional[float] = None

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

    def _on_clock(self, msg: Clock) -> None:
        self.sim_time = msg.clock.sec + msg.clock.nanosec * 1e-9

    def _on_imu(self, msg: Imu) -> None:
        # Body-frame yaw rate + linear acceleration (x fwd, y left). Matches the
        # (gyro_z, ax, ay) order obs.build_observation expects.
        self.imu = (
            float(msg.angular_velocity.z),
            float(msg.linear_acceleration.x),
            float(msg.linear_acceleration.y),
        )

    @property
    def odom_seen(self) -> bool:
        return self._odom_seen


class GazeboBridge:
    """Drives one Gazebo instance for the RL corrector env."""

    # rtf to request when unthrottling. "As fast as the CPU allows" -- paired with
    # real_time_update_rate=0 (unlimited) it removes the cap entirely.
    _FAST_RTF = 1000.0
    # World SDF defaults (worlds/rl_corrector.world), restored on close.
    _DEFAULT_RTF = 1.0
    _DEFAULT_UPDATE_RATE = 1000.0
    # Gravity to (re)assert on every set_physics. The gz Physics proto carries a
    # gravity field that defaults to (0,0,0), and /world/<world>/set_physics
    # applies EVERY field of the request, so a request that omits gravity zeroes
    # it -- the robot floats. We can only set physics, not read it back, but
    # gravity is a static world constant, so we just rewrite the known value.
    # Matches <gravity> in worlds/rl_corrector.world.
    _GRAVITY_Z = -9.8

    def __init__(
        self,
        cfg,
        world_name: str = "rl_corrector",
        model_name: str = "scout_mini",
        physics_step: float = 0.01,
        deterministic: bool = False,
        unthrottle: bool = False,
        # Measured settled height of the scout_mini's model origin on flat floor
        # (2026-08-02, reset_probe: z = 0.1806 on every one of 12 resets). Placing
        # AT it rather than above it means the robot barely falls, so it barely
        # slides -- which is where the reset's positional spread came from. Slip
        # patches are 1 mm decals, so this is right over a patch too.
        reset_z: float = 0.1806,
        settle_steps: int = 5,
        reset_tol: float = 0.15,
        # How close the settled pose must be to the requested one before the
        # rollout is allowed to start, and how many re-place attempts to spend
        # getting there. 1 mm is ~8x tighter than the un-refined reset and well
        # below the patch-edge scale that was amplifying the error.
        reset_place_tol: float = 0.001,
        reset_refine_iters: int = 4,
        # Vertical settle: keep stepping a stationary robot until its height
        # stops changing by more than this per control step. 1e-7 m is ~200x
        # below the ~2e-5 m run-to-run z spread the fixed settle was leaving.
        reset_settle_z_tol: float = 1e-7,
        reset_max_settle_iters: int = 60,
        service_timeout_ms: int = 3000,
        step_ack_ms: int = 10,
        spin_warmup_s: float = 10.0,
    ) -> None:
        self.cfg = cfg
        self.world = world_name
        self.model = model_name
        self.physics_step = float(physics_step)
        self.deterministic = bool(deterministic)
        self.unthrottle = bool(unthrottle)
        self.reset_z = float(reset_z)
        self.settle_steps = int(settle_steps)
        self.reset_place_tol = float(reset_place_tol)
        self.reset_refine_iters = int(reset_refine_iters)
        self.reset_settle_z_tol = float(reset_settle_z_tol)
        self.reset_max_settle_iters = int(reset_max_settle_iters)
        self.reset_settle_iters = 0
        self.reset_z_settled: Optional[float] = None
        # Distance the last reset actually settled from the requested pose --
        # read it to verify reproducibility instead of inferring it from spread.
        self.reset_offset = 0.0
        # Patch names the last _apply_terrain asked for but could not verify.
        self.terrain_missing: List[str] = []
        # Lenient margin for "did the teleport land?" checked against the settled
        # pose (vs the strict in-flight confirm tol in _set_pose).
        self.reset_tol = float(reset_tol)
        self.timeout_ms = int(service_timeout_ms)
        # Short timeout for the flaky-ack world services (set_pose/control): the
        # command executes fast; we cap the wait so a false-negative ack can't
        # stall a step/reset for the full service timeout. See _set_pose.
        self._ack_ms = min(800, int(service_timeout_ms))
        # Even shorter timeout for the HOT-PATH world-control step. That ack is
        # the flaky one (see the ack-quirk note), and deterministic mode advances
        # the world every env step, so blocking the full _ack_ms per step is what
        # pins throughput to ~1 step/s. We don't wait for the reply at all here:
        # the step is SENT synchronously, then _advance detects completion via a
        # fresh ground-truth pose. Used for the per-step multi_step and the
        # set_pose confirm-loop step (whose 0.5s deadline a full ack would blow).
        self._step_ack_ms = int(step_ack_ms)

        # gz-sim service endpoints (convention: /world/<world>/<verb>).
        self._svc_set_pose = f"/world/{world_name}/set_pose"
        self._svc_control = f"/world/{world_name}/control"
        self._svc_create = f"/world/{world_name}/create"
        self._svc_remove = f"/world/{world_name}/remove"
        self._svc_set_physics = f"/world/{world_name}/set_physics"
        # pose/info (NOT dynamic_pose/info): it is published for ALL entities at
        # a steady rate, so the robot pose stays fresh even when it is momentarily
        # at rest (dynamic_pose only carries *moving* entities), and it preserves
        # entity names so we can match the model. See _set_pose for the ack quirk.
        self._topic_pose = f"/world/{world_name}/pose/info"

        # --- gz transport: ground-truth pose + world services ---
        self._gz = gz_transport.Node()
        self._pose_xyth: Optional[Tuple[float, float, float]] = None
        self._pose_z: Optional[float] = None
        self._pose_quat: Optional[Tuple[float, float, float, float]] = None
        self._pose_stamp: Optional[float] = None
        # Steps that proceeded on a pose older than the step they were meant to
        # observe. Should be 0; a nonzero count means the rollout fed the
        # controller stale state and is not comparable with another.
        self.stale_pose_steps = 0
        self._entity_names: set = set()
        self._terrain_poses: dict = {}
        # --- state trace (see enable_trace) ---
        self._trace_fh = None
        self._trace_writer = None
        self._trace_row = 0
        self._trace_phase = "init"
        self._gz.subscribe(Pose_V, self._topic_pose, self._on_pose)

        # --- rclpy: command out + twist in ---
        if not rclpy.ok():
            rclpy.init()
        self._node = _BridgeNode(use_wheel_speeds=cfg.use_wheel_speeds,
                                 use_imu=cfg.use_imu)
        self._exec = SingleThreadedExecutor()
        self._exec.add_node(self._node)

        # --- determinism instrumentation ---------------------------------
        # Deterministic stepping guarantees "same initial state + same commands
        # -> same trajectory". It does NOT guarantee we re-establish the same
        # initial state, nor that every requested step actually happened. These
        # count the two ways that promise leaks, so a rollout can report whether
        # it was reproducible instead of leaving it to be inferred from spread.
        self.lost_steps = 0        # _advance gave up waiting for the sim clock
        self.total_steps = 0
        self.reset_ticks = 0       # physics ticks burned in the last reset's
                                   # confirm loop -- wall-clock-paced, so a
                                   # varying count means a varying start state

        # Names of terrain patches we spawned, so we can remove them next reset.
        self._terrain_models: List[str] = []
        # ...but that list only knows about THIS process. A previous process --
        # a trainer that was Ctrl-C'd, or one that exited mid-episode -- leaves
        # its patches in the running world, and the sim outlives any single
        # process by design (one Gazebo, many runs). Inheriting them silently
        # changes the plant: a leftover `rl_ground` from a --ground-friction run
        # is one low-friction slab under the whole trajectory, and it made an
        # entire corrector comparison unreproducible before it was noticed.
        # Names are fixed by terrain.py, so sweep them by name on startup rather
        # than querying the scene -- the removes are best-effort anyway.
        self._sweep_stale_terrain()

        # Teleport-confirm failures in a row. A single failure is usually just the
        # flaky pose read-back (the teleport itself executes), so we proceed; a
        # long run of them means a real problem (wrong model/world, dead sim), so
        # we raise rather than train forever on a robot that never moves.
        self._consec_reset_fail = 0
        self._max_consec_reset_fail = 10

        # Ensure the world is RUNNING before we wait. The sim can come up paused
        # (the gz GUI may start paused) or be left paused by a prior deterministic
        # run that was hard-killed before close() un-paused it. While paused,
        # pose/info still republishes (so ground-truth pose arrives), but /odom --
        # wheel_odometry integrating /joint_states -- only updates when the world
        # steps, so _wait_ready would block on the twist forever. Best-effort like
        # every world-control call (the ack is flaky; the unpause still executes).
        self._world_control(pause=False)

        # Lift the real-time cap so stepping runs as fast as the CPU allows. The
        # world SDF ships real_time_factor=1.0 + real_time_update_rate=1000, which
        # throttles BOTH wall-clock running and (via the update-rate target) the
        # paused-world multi_step, pinning training near 1x. We raise rtf and set
        # update_rate=0 (unlimited); restored in close(). Opt-in (training sets it)
        # so validation/visualization can keep real-time. Best-effort like every
        # world service -- the flaky ack doesn't stop the change taking effect.
        if self.unthrottle:
            self._set_physics(real_time_factor=self._FAST_RTF, real_time_update_rate=0.0)

        # Wait for the first ground-truth pose + odom (+ clock, used by the
        # deterministic step gate) so step 0 is valid.
        self._wait_ready(spin_warmup_s)
        if self.deterministic:
            self._ensure_paused()

    # ------------------------------------------------------------------
    # gz callbacks / helpers
    # ------------------------------------------------------------------

    def _entities_present(self, names) -> set:
        """Which of `names` currently exist in the world, per pose/info.

        pose/info enumerates every entity each time the world advances, so it
        doubles as the only cheap way to ASK the sim what is actually there --
        the create/remove services only ever report what they were asked to do.
        """
        seen = self._entity_names
        return {n for n in names if n in seen}

    def _on_pose(self, msg: Pose_V) -> None:
        """Cache the robot's ground-truth world pose from pose/info.

        pose/info lists every entity (ground/sun/links/visuals); the top-level
        model pose has name == model_name. Match the model exactly.
        """
        # Snapshot every top-level name in this message, so terrain spawning can
        # be verified instead of assumed (see _apply_terrain).
        self._entity_names = {p.name for p in msg.pose}
        # Sim time this pose describes. THE STEP GATE DEPENDS ON THIS: pose/info
        # arrives on a gz-transport thread while the step gate spins rclpy, so
        # without a stamp there is no way to tell a post-step pose from the
        # pre-step one still sitting in the cache. See _advance.
        self._pose_stamp = msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9
        for p in msg.pose:
            # Terrain poses, kept for the trace: a patch that lands a millimetre
            # off, or at a different yaw, is a different plant even though the
            # name-presence check in _apply_terrain passes.
            if p.name.startswith("rl_"):
                q = p.orientation
                self._terrain_poses[p.name] = (
                    p.position.x, p.position.y, p.position.z,
                    q.x, q.y, q.z, q.w,
                )
            if p.name == self.model:
                q = p.orientation
                self._pose_xyth = (
                    p.position.x, p.position.y,
                    _yaw_from_quat(q.x, q.y, q.z, q.w),
                )
                # z is not part of the planning pose, but it is the direct
                # readout of whether the robot has finished falling from
                # reset_z. Kept for reset diagnostics only.
                self._pose_z = p.position.z
                # Full orientation, for the trace only: the planning yaw hides
                # roll/pitch, and a rollout that diverges by tipping onto a patch
                # edge shows it in pitch several steps before it shows in x/y.
                self._pose_quat = (q.x, q.y, q.z, q.w)

    def _spin(self, duration_s: float) -> None:
        """Spin rclpy for a wall-clock duration (delivers cmd flush + /odom)."""
        end = time.monotonic() + duration_s
        while time.monotonic() < end:
            self._exec.spin_once(timeout_sec=max(0.0, end - time.monotonic()))

    def _wait_ready(self, timeout_s: float) -> None:
        # The sim clock is only needed by the deterministic step gate; don't make
        # wall-clock setups (which may run without the /clock bridge) block on it.
        need_clock = self.deterministic
        end = time.monotonic() + timeout_s
        need_imu = self.cfg.use_imu
        while time.monotonic() < end:
            self._exec.spin_once(timeout_sec=0.05)
            if (self._pose_xyth is not None and self._node.odom_seen
                    and (not need_clock or self._node.sim_time is not None)
                    and (not need_imu or self._node.imu is not None)):
                return
        missing = []
        if self._pose_xyth is None:
            missing.append(f"ground-truth pose on {self._topic_pose}")
        if not self._node.odom_seen:
            missing.append(f"twist on {_ODOM_TOPIC}")
        if need_clock and self._node.sim_time is None:
            missing.append(f"sim clock on {_CLOCK_TOPIC}")
        if need_imu and self._node.imu is None:
            missing.append(f"IMU on {_IMU_TOPIC} (is it bridged to ROS? "
                           "use_imu=True needs sensor_msgs/Imu there)")
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

    def _world_control(self, pause: Optional[bool] = None, multi_step: int = 0,
                       ack_ms: Optional[int] = None) -> None:
        req = WorldControl()
        if multi_step > 0 and pause is None and self.deterministic:
            # THE `pause` FIELD IS NOT OPTIONAL ON THE WIRE. WorldControl.pause is
            # a plain proto3 bool, so leaving it unset sends `pause: false`, and
            # gz applies it unconditionally alongside the step. Every multi_step
            # we issued therefore ALSO un-paused the world: it ran the n ticks we
            # asked for and then FREE-RAN until the next call, for however long
            # the CPU gave it. Measured 2026-08-02: control steps advancing
            # 0.42 s of sim time instead of control_dt=0.1, wall-clock-dependent
            # and different every run, with lost_steps and stale_pose_steps both
            # reporting 0 because nothing was being dropped -- extra physics was
            # being run. Deterministic mode must re-assert the pause on every
            # step; this is the whole basis of "same commands -> same rollout".
            pause = True
        if pause is not None:
            req.pause = pause
        if multi_step > 0:
            req.multi_step = multi_step
        # Best-effort: the step/pause executes regardless of the (flaky) ack.
        # Callers on the hot path pass a tiny ack_ms so a lost reply can't stall
        # them; correctness is re-established by reading ground-truth state.
        self._gz.request(self._svc_control, req, WorldControl, Boolean,
                         self._ack_ms if ack_ms is None else ack_ms)

    def _set_physics(self, real_time_factor: float,
                     real_time_update_rate: float) -> None:
        """Set the world's physics rates (rtf + update-rate cap). Keeps the SDF
        max_step_size so the deterministic n-ticks-per-step math is unchanged.
        Best-effort: like the other world services the ack is flaky, but the
        change takes effect and stepping speed is observable downstream."""
        req = Physics()
        req.max_step_size = self.physics_step
        req.real_time_factor = float(real_time_factor)
        req.real_time_update_rate = float(real_time_update_rate)
        # set_physics overwrites every field from the request and the proto
        # defaults gravity to (0,0,0); re-assert it or the world loses gravity.
        req.gravity.z = self._GRAVITY_Z
        self._gz.request(self._svc_set_physics, req, Physics, Boolean, self._ack_ms)

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
        if self.deterministic:
            return self._set_pose_stepped(req, x, y, tol, tries)

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
                    self._world_control(multi_step=1, ack_ms=self._step_ack_ms)
                    # Counted because this loop is paced by the WALL clock while
                    # stepping physics: how many ticks the robot gets to fall and
                    # settle from reset_z depends on machine timing, so the
                    # episode's true initial state varies run to run.
                    self.reset_ticks += 1
                self._exec.spin_once(timeout_sec=0.02)
                p = self._pose_xyth
                if p is not None and math.hypot(p[0] - x, p[1] - y) < tol:
                    return True
        return False

    # Physics ticks spent settling after a teleport, in deterministic mode.
    # FIXED, not "however many fit in 0.5 s of wall clock" -- see below.
    _confirm_ticks = 20

    def _set_pose_stepped(self, req, x: float, y: float, tol: float,
                          tries: int) -> bool:
        """Deterministic teleport: a FIXED number of physics ticks, every time.

        The wall-clock-paced version of this loop stepped physics until a 0.5 s
        deadline or the pose came within tol, so the number of ticks the robot
        spent falling and settling depended on machine timing -- observed 12 to
        31 across five back-to-back rollouts (`reset_ticks`). Those are real
        physics ticks, so the episode's true initial state differed run to run,
        and the runs with the low counts were the ones whose final error stood
        out. It was previously written off as self-correcting because each retry
        yanks the body back to target; that is true of x/y and false of the
        vertical and contact state, which is precisely the seed everything else
        amplifies.

        Spinning rclpy does NOT step the world, so waiting on the wall clock for
        a message is harmless here; only the tick count has to be fixed.
        """
        for _ in range(tries):
            self._gz.request(self._svc_set_pose, req, Pose, Boolean, self._ack_ms)
            stamp0 = self._pose_stamp
            for _ in range(self._confirm_ticks):
                self._world_control(multi_step=1, ack_ms=self._step_ack_ms)
                self.reset_ticks += 1
                self._exec.spin_once(timeout_sec=0.0)
            # Let the pose catch up to the ticks just taken, without stepping.
            end = time.monotonic() + 0.5
            while time.monotonic() < end:
                if stamp0 is None or (self._pose_stamp is not None
                                      and self._pose_stamp > stamp0):
                    break
                self._exec.spin_once(timeout_sec=0.005)
            p = self._pose_xyth
            if p is not None and math.hypot(p[0] - x, p[1] - y) < tol:
                return True
            # Only a genuine miss costs another (still fixed-size) round.
        return False

    def _advance(self, dt: float) -> None:
        """Advance the sim by dt under the active stepping mode, then refresh
        rclpy-side state. The gz pose callback runs on its own thread."""
        if self.deterministic:
            n = max(1, int(round(dt / self.physics_step)))
            t0 = self._node.sim_time
            # Fire the step without waiting on its (flaky/lost) ack -- that wait
            # is what throttled training to ~1 step/s.
            self._world_control(multi_step=n, ack_ms=self._step_ack_ms)
            # Gate on the SIM CLOCK advancing by the requested n ticks. pose/info
            # republishes per physics tick while the world advances, so the old
            # "first fresh pose" gate returned after ~1 of the n ticks -- letting
            # the env loop outrun physics (under-simulated steps, fake throughput,
            # e_cross collapsing to ~0). The clock is the authoritative measure of
            # how much sim time actually elapsed; wait for the full dt. Spinning
            # rclpy meanwhile keeps the /odom twist fresh. Bounded so a genuinely
            # dropped step can't hang the loop.
            ok = self._wait_clock_advance(t0, n * self.physics_step,
                                          deadline_s=max(0.2, 10.0 * dt))
            # A False here means the world did NOT advance the full dt before the
            # wall-clock deadline, and we carried on regardless -- the episode
            # silently lost physics time while the control index moved on. That is
            # a hidden, run-dependent input, not noise: it is why two "identical"
            # rollouts diverge under deterministic stepping. Counted rather than
            # raised, because the caller may legitimately prefer a degraded
            # rollout to a crash; measure_determinism reads these.
            if not ok:
                self.lost_steps += 1
            # The clock gate proves the WORLD advanced; it does not prove we can
            # SEE the result. Ground-truth pose arrives over gz transport on its
            # own thread, while the gate above spins rclpy -- so /clock routinely
            # reaches the target while _pose_xyth still holds the pre-step pose.
            # _read_state then hands the controller a stale pose, the controller
            # computes a command for a position the robot has already left, and
            # two identical rollouts diverge. Measured 2026-08-02: x/y/yaw
            # bit-identical for two steps, then one run's step-2 state was
            # literally its step-1 state, and the runs ended 0.53 m apart.
            # world_steps agreed exactly throughout -- the physics was never the
            # problem; the readout was.
            if t0 is not None:
                if not self._wait_pose_advance(
                        t0 + n * self.physics_step,
                        deadline_s=max(0.2, 10.0 * dt)):
                    self.stale_pose_steps += 1
            self.total_steps += 1
        else:
            self._spin(dt)

    # How long to watch the sim clock to decide the world really stopped, and how
    # many pause requests to spend before giving up. The world publishes /clock
    # every tick while running, so a free-running world betrays itself in well
    # under 100 ms even at rtf 1; 0.25 s is generous.
    _pause_check_s = 0.25
    _pause_attempts = 5

    def _ensure_paused(self) -> None:
        """Pause the world and VERIFY it stopped, retrying if it did not.

        WHY THIS IS NOT PARANOIA. Every world-control call is best-effort: the
        ack is unreliable, so the code has always fired pause and moved on. When
        that request is dropped the world keeps FREE-RUNNING, and deterministic
        mode silently becomes something much worse than non-deterministic --
        multi_step still advances the world, but so does real time, so each
        control step covers however much sim time the CPU happened to deliver.
        Measured 2026-08-02: control steps advancing 0.42 s of sim time instead
        of control_dt=0.1, tracking error 8-20 m instead of ~2, and no counter
        anywhere reporting a problem (lost_steps and stale_pose_steps were both
        0 -- the world was not dropping steps, it was doing extra ones).

        The sim clock is the honest test: a paused world stops publishing it.
        """
        for attempt in range(self._pause_attempts):
            self._world_control(pause=True)
            before = self._node.sim_time
            end = time.monotonic() + self._pause_check_s
            while time.monotonic() < end:
                self._exec.spin_once(timeout_sec=0.01)
            after = self._node.sim_time
            if before is None or after is None or after <= before + 1e-9:
                if attempt:
                    self._node.get_logger().warn(
                        "world needed %d pause attempts before it stopped"
                        % (attempt + 1))
                return
            self._node.get_logger().warn(
                "pause request %d did not take effect (clock advanced %.3f s); "
                "retrying" % (attempt + 1, after - before))
        raise RuntimeError(
            "could not pause world %r: the sim clock keeps advancing after %d "
            "pause requests. Deterministic stepping is meaningless against a "
            "free-running world -- every rollout would simulate a different "
            "amount of time. Is a second process (a GUI, another trainer) "
            "un-pausing it?" % (self.world, self._pause_attempts))

    def _wait_pose_advance(self, target_sim_t: float, deadline_s: float) -> bool:
        """Block until the cached ground-truth pose is stamped at/after `target`.

        The pose callback fires on the gz-transport thread, so watching the value
        would seem to be enough -- but this must still SPIN rclpy while it waits.
        A plain sleep() here starves /clock and /odom (single-threaded executor,
        same thread), so the next step's clock baseline is stale and the twist
        handed to the controller is older than the pose: measured 2026-08-02, a
        sleeping version of this loop drove tracking error from ~2.5 m to ~20 m.
        """
        # Half a physics tick of slack, same rationale as the clock gate.
        target = target_sim_t - 0.5 * self.physics_step
        end = time.monotonic() + deadline_s
        while time.monotonic() < end:
            stamp = self._pose_stamp
            if stamp is not None and stamp >= target:
                self._exec.spin_once(timeout_sec=0.0)   # drain twist to match
                return True
            self._exec.spin_once(timeout_sec=0.001)
        return False

    def _wait_clock_advance(self, t0: Optional[float], sim_dt: float,
                            deadline_s: float) -> bool:
        """Spin rclpy until the sim clock has advanced by `sim_dt` past `t0`.

        Returns True if it advanced, False if `deadline_s` of WALL time elapsed
        first -- in which case the caller proceeds with an under-advanced world.
        """
        # Half a tick of slack so float jitter / a clock sample landing mid-step
        # doesn't force an extra wait.
        target = None if t0 is None else t0 + sim_dt - 0.5 * self.physics_step
        end = time.monotonic() + deadline_s
        while time.monotonic() < end:
            self._exec.spin_once(timeout_sec=0.005)
            t = self._node.sim_time
            if target is None:
                # No clock baseline (shouldn't happen post-_wait_ready); fall back
                # to a one-shot drain so we don't busy-wait the whole deadline.
                if t is not None:
                    self._exec.spin_once(timeout_sec=0.0)
                    return True
            elif t is not None and t >= target:
                self._exec.spin_once(timeout_sec=0.0)  # one more drain for twist
                return True
        return False

    # ------------------------------------------------------------------
    # State trace (reproducibility instrumentation)
    # ------------------------------------------------------------------

    # Every scalar the world can hand back about the robot, plus the terrain
    # digest. The point is to find the FIRST step at which two supposedly
    # identical rollouts differ: an end-of-rollout metric only says that they
    # did, and by then the cause is 200 steps upstream. Ordering matters only in
    # that it must be stable across runs, so diff_trace can compare positionally.
    TRACE_COLUMNS = (
        "row", "phase", "sim_time", "world_steps", "lost_steps",
        "stale_pose_steps", "pose_stamp",
        "cmd0", "cmd1", "cmd2", "cmd3",
        "x", "y", "z", "yaw", "qx", "qy", "qz", "qw",
        "v", "omega", "w0", "w1", "w2", "w3",
        "imu_gz", "imu_ax", "imu_ay", "terrain",
    )

    def enable_trace(self, path: str) -> None:
        """Write one row per recorded snapshot to `path` (CSV).

        Rows are appended by step() and at each phase of reset(), so a trace
        covers the reset as well as the rollout -- the reset is where a
        divergence is most likely to be BORN even though it only becomes visible
        later. Compare two traces with tools/diff_trace.py.
        """
        import csv
        if self._trace_fh is not None:      # re-armed per rollout
            self._trace_fh.close()
        self._trace_fh = open(path, "w", newline="")
        self._trace_writer = csv.writer(self._trace_fh)
        self._trace_writer.writerow(self.TRACE_COLUMNS)
        self._trace_row = 0

    def _terrain_digest(self) -> str:
        """Every rl_* entity pose, rounded to 1e-6, as one stable string."""
        return "|".join(
            "%s:%s" % (name, ",".join("%.6f" % c for c in pose))
            for name, pose in sorted(self._terrain_poses.items())
        )

    def _trace(self, phase: str, wheels=None) -> None:
        if self._trace_writer is None:
            return
        # Drain callbacks so the snapshot reflects the world as of now, not as of
        # whenever the last spin happened to run.
        self._exec.spin_once(timeout_sec=0.0)
        p = self._pose_xyth or (float("nan"),) * 3
        q = self._pose_quat or (float("nan"),) * 4
        w = self._node.wheel_speeds or [float("nan")] * 4
        imu = self._node.imu or (float("nan"),) * 3
        cmd = list(wheels) if wheels is not None else [float("nan")] * 4
        self._trace_writer.writerow(
            [self._trace_row, phase,
             "%.6f" % (self._node.sim_time if self._node.sim_time is not None
                       else float("nan")),
             self.total_steps, self.lost_steps, self.stale_pose_steps,
             "%.6f" % (self._pose_stamp if self._pose_stamp is not None
                       else float("nan"))]
            + ["%.9g" % float(c) for c in cmd]
            + ["%.9g" % float(v) for v in (p[0], p[1],
                                           self._pose_z if self._pose_z is not None
                                           else float("nan"), p[2])]
            + ["%.9g" % float(c) for c in q]
            + ["%.9g" % float(v) for v in (self._node.v, self._node.omega)]
            + ["%.9g" % float(v) for v in w]
            + ["%.9g" % float(v) for v in imu]
            + [self._terrain_digest()]
        )
        self._trace_row += 1

    # ------------------------------------------------------------------
    # Bridge contract
    # ------------------------------------------------------------------

    def reset(self, start_pose, terrain=None) -> StateReading:
        x, y, th = (float(v) for v in start_pose)
        # Re-verify once per episode, not just at construction: a GUI client
        # attaching, or any other process touching /world/<w>/control, un-pauses
        # the world, and from then on every rollout is measured against a
        # different plant with nothing in the logs to say so.
        if self.deterministic:
            self._ensure_paused()
        self._trace("reset_enter")

        # 1. Halt the wheels so the teleport doesn't carry old motion through.
        self._node.publish_wheels([0.0, 0.0, 0.0, 0.0])
        self._spin(0.05)

        # 2. Swap terrain (Phase 3); no-op when terrain is None.
        self._apply_terrain(terrain, near=(x, y))
        self._trace("reset_terrain")

        # 3. Teleport to the episode start (slightly raised, then settle down).
        #    The gz set_pose service EXECUTES reliably even when its ack/read-back
        #    is flaky (see the ack-quirk note), so an unconfirmed teleport almost
        #    always still landed -- we settle and re-check rather than crash. Only
        #    a long streak of failures (sim dead / wrong model) is fatal.
        self.reset_ticks = 0
        confirmed = self._set_pose(x, y, self.reset_z, th)
        self._trace("reset_teleport")

        # 4. Settle: zero command for a few steps so residual twist damps out
        #    (set_pose moves the body but does not zero its velocity).
        for _ in range(self.settle_steps):
            self._node.publish_wheels([0.0, 0.0, 0.0, 0.0])
            self._advance(self.cfg.control_dt)
            self._trace("reset_settle", [0.0] * 4)

        st = self._read_state()

        # 5. Re-place and re-settle until the settled pose stops moving.
        #
        #    WHY THIS EXISTS. Steps 3-4 leave the robot up to ~8 mm from target:
        #    it is dropped from reset_z onto the floor and slides slightly as the
        #    contacts resolve, by an amount that depends on the wall-clock-paced
        #    number of ticks the confirm loop happened to take. 8 mm sounds
        #    negligible and is not: with slip patches on the path it decides
        #    whether a wheel catches a patch EDGE, a discontinuous change in
        #    friction under that wheel. That is how ten "identical" rollouts on
        #    2026-08-02 produced four results at 0.223 m and the rest scattered
        #    from 1.5 to 6.9 m -- reproducible modes, not smooth noise.
        #
        #    The sim itself is deterministic, so identical initial states DO give
        #    identical rollouts; the fix is simply to make the initial state
        #    actually identical. Placing at the settled height (see reset_z)
        #    removes most of the fall, and this loop polishes what remains.
        #    Converges in 1-2 passes; the bound is only there so a robot wedged
        #    against geometry cannot spin here forever.
        for _ in range(self.reset_refine_iters):
            if math.hypot(st.pose[0] - x, st.pose[1] - y) <= self.reset_place_tol:
                break
            self._set_pose(x, y, self.reset_z, th)
            for _ in range(self.settle_steps):
                self._node.publish_wheels([0.0, 0.0, 0.0, 0.0])
                self._advance(self.cfg.control_dt)
                self._trace("reset_refine", [0.0] * 4)
            st = self._read_state()
        self.reset_offset = math.hypot(st.pose[0] - x, st.pose[1] - y)

        # 6. Settle the VERTICAL state until it stops moving.
        #
        #    x/y/yaw come back bit-identical across resets, but z did not: the
        #    fixed-count settle above leaves the robot still micro-bouncing on
        #    its suspension by ~2e-5 m, differing run to run, and the IMU by
        #    ~0.02. That is the only remaining difference between two rollouts
        #    that are otherwise identical inputs, so it is the seed for whatever
        #    they eventually diverge by. Converge on it instead of hoping five
        #    steps was enough. Bounded, and it exits on the first stable pair, so
        #    a robot that has already settled pays one extra step.
        self.reset_settle_iters = 0
        z_prev = self._pose_z
        for _ in range(self.reset_max_settle_iters):
            self._node.publish_wheels([0.0, 0.0, 0.0, 0.0])
            self._advance(self.cfg.control_dt)
            self.reset_settle_iters += 1
            self._trace("reset_zsettle", [0.0] * 4)
            z_now = self._pose_z
            if (z_prev is not None and z_now is not None
                    and abs(z_now - z_prev) <= self.reset_settle_z_tol):
                break
            z_prev = z_now
        self.reset_z_settled = self._pose_z
        st = self._read_state()

        # Re-check against the settled ground-truth pose: this both rescues a
        # teleport whose mid-flight confirm raced the residual velocity, and
        # detects a genuinely-stuck robot.
        landed = confirmed or math.hypot(st.pose[0] - x, st.pose[1] - y) < self.reset_tol
        if landed:
            self._consec_reset_fail = 0
        else:
            self._consec_reset_fail += 1
            self._node.get_logger().warn(
                "reset teleport unconfirmed (%.2f m from target, %d in a row); "
                "proceeding." % (math.hypot(st.pose[0] - x, st.pose[1] - y),
                                 self._consec_reset_fail)
            )
            if self._consec_reset_fail >= self._max_consec_reset_fail:
                raise RuntimeError(
                    "set_pose failed %d resets in a row on %s -- is the sim alive "
                    "and is the model named %r in world %r? (a single failure is "
                    "tolerated; this many means a real problem)"
                    % (self._consec_reset_fail, self._svc_set_pose,
                       self.model, self.world)
                )
        return st

    def step(self, wheels, dt: float) -> StateReading:
        self._node.publish_wheels(wheels)
        self._advance(dt)
        st = self._read_state()
        self._trace("step", wheels)
        return st

    def close(self) -> None:
        try:
            self._node.publish_wheels([0.0, 0.0, 0.0, 0.0])
            self._spin(0.02)
            if self.deterministic:
                self._world_control(pause=False)  # leave the sim runnable
            if self.unthrottle:
                # Restore the SDF real-time cap so a sim reused after training
                # (e.g. the validate script) is not left running flat-out.
                self._set_physics(real_time_factor=self._DEFAULT_RTF,
                                  real_time_update_rate=self._DEFAULT_UPDATE_RATE)
        finally:
            if self._trace_fh is not None:
                self._trace_fh.close()
                self._trace_fh = None
                self._trace_writer = None
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
            imu=self._node.imu if self.cfg.use_imu else None,
        )

    def _settle_entity_changes(self, ticks: int = 2) -> None:
        """Let queued entity creations/removals actually take effect.

        gz-sim applies entity changes at the next world STEP, and in
        deterministic mode the world is PAUSED -- so a remove requested here has
        not happened yet when the next request goes out. Stepping is the only
        thing that commits it. Cheap: 2 physics ticks, not control steps.
        """
        if self.deterministic:
            for _ in range(ticks):
                self._world_control(multi_step=1, ack_ms=self._step_ack_ms)
        self._exec.spin_once(timeout_sec=0.02)

    def _wait_entities(self, names, max_ticks: int) -> bool:
        """Step the world until every name in `names` appears in pose/info.

        Returns True once they are all present, False if `max_ticks` ran out.
        Stepping is what commits a pending creation, and pose/info republishes
        per tick, so this both drives and observes the same process.
        """
        for _ in range(max_ticks):
            self._exec.spin_once(timeout_sec=0.01)
            if not set(names) - self._entities_present(names):
                return True
            if self.deterministic:
                self._world_control(multi_step=1, ack_ms=self._step_ack_ms)
            else:
                self._spin(0.02)
        self._exec.spin_once(timeout_sec=0.05)
        return not (set(names) - self._entities_present(names))

    def _wait_entities_gone(self, names, max_ticks: int) -> bool:
        """Step until none of `names` remains in pose/info (mirror of _wait_entities)."""
        for _ in range(max_ticks):
            self._exec.spin_once(timeout_sec=0.01)
            if not self._entities_present(names):
                return True
            if self.deterministic:
                self._world_control(multi_step=1, ack_ms=self._step_ack_ms)
            else:
                self._spin(0.02)
        self._exec.spin_once(timeout_sec=0.05)
        return not self._entities_present(names)

    def _apply_terrain(self, terrain, near) -> None:
        """Remove last episode's patches and spawn this episode's. `terrain` is a
        list of patch dicts (see spawn_surface_patches schema) or None.

        THE ORDERING HERE IS LOAD-BEARING. `create` on a name that still exists
        FAILS, and removals only commit on a world step (which never happens on
        its own while the world is paused). The original code removed and
        immediately re-created the same names, so whether the patches existed at
        all came down to wall-clock service timing: on 2026-08-02 roughly 4 of 10
        "identical" rollouts silently ran on BARE GROUND, scoring 0.2247 m --
        indistinguishable from --no-terrain -- while the rest scored 1.5-6.9 m.
        That, not chaos and not the reset, was the run-to-run variance.
        """
        self._remove_terrain()
        if not terrain:
            # Still step: the removals above must commit, or the NEXT episode
            # inherits them and its own create fails instead.
            self._settle_entity_changes()
            return
        from .terrain import patch_sdf  # lazy: keeps gz-only deps off the hot path
        by_name = {}
        for idx, patch in enumerate(terrain):
            name = patch.get("name") or f"rl_patch_{idx}"
            by_name[name] = (patch, idx)
        wanted = list(by_name)

        # Commit the removals before re-using the same names: a create against a
        # name that still exists is the one failure mode that is genuinely fatal
        # rather than merely slow.
        self._wait_entities_gone(wanted, self._terrain_remove_ticks)

        # Issue every create ONCE, then wait for the world to actually show them.
        #
        # Do NOT re-issue on a missing name. With the world paused the create
        # blocks for the whole ack timeout and returns FALSE, yet the entity is
        # created anyway and shows up some ticks later (measured 2026-08-02,
        # tuning/spawn_diag.py). So a "missing" patch is almost always a patch
        # in flight -- and re-creating it then genuinely fails, because by that
        # point the name exists. The retry that seemed obvious made it worse.
        #
        # The ack is useless here in both directions, so it is not consulted;
        # pose/info is the only honest answer to "is it there".
        for name in wanted:
            patch, _idx = by_name[name]
            sdf = patch_sdf(patch, name)
            req = EntityFactory()
            req.sdf = sdf
            req.name = name
            req.pose.position.x = float(patch.get("x", near[0]))
            req.pose.position.y = float(patch.get("y", near[1]))
            req.pose.position.z = float(patch.get("z", 0.001))
            self._gz.request(self._svc_create, req, EntityFactory, Boolean,
                             self._ack_ms)
            if name not in self._terrain_models:
                self._terrain_models.append(name)

        # Step until every patch is visible. This is the whole point: a patch
        # that lands a few steps INTO the rollout changes the plant mid-episode,
        # by a different amount each run -- which is exactly the run-to-run
        # spread we have been chasing. The rollout must not start until the
        # ground it is measured on is fully in place.
        self._wait_entities(wanted, self._terrain_spawn_ticks)
        self.terrain_missing = sorted(set(wanted) - self._entities_present(wanted))
        if self.terrain_missing:
            raise RuntimeError(
                "terrain patches failed to spawn: %s (asked for %s). The rollout "
                "would run on different ground than intended and must not be "
                "compared with others. This used to happen silently and was the "
                "source of the 0.22-vs-6.9 m run-to-run spread."
                % (self.terrain_missing, wanted))

    # Upper bound on `rl_patch_N` indices to sweep. along_path_terrain_sampler
    # spawns at most n_range[1] (3) and ground_friction_sampler adds `rl_ground`;
    # 8 leaves room for a wider n_range without another silent inheritance bug,
    # while keeping the worst case (every request timing out at _ack_ms) under a
    # second of startup cost.
    # Attempts to get every requested patch to actually appear. A create is
    # rejected while a same-named entity still exists and removals commit only on
    # a world step, so the first attempt fails ~half the time.
    # Ticks to spend waiting for a requested patch to actually appear. With the
    # world paused a create's ack times out (~800 ms) while the entity still
    # lands a few ticks later, so this is the real bound that matters.
    _terrain_spawn_ticks = 60
    # Ticks to spend waiting for removals to disappear before re-using the names.
    _terrain_remove_ticks = 60

    _STALE_PATCH_LIMIT = 8

    def _sweep_stale_terrain(self) -> None:
        """Remove terrain patches left in the world by an earlier process."""
        stale = ["rl_ground"] + [f"rl_patch_{i}" for i in range(self._STALE_PATCH_LIMIT)]
        for name in stale:
            req = Entity()
            req.name = name
            req.type = Entity.MODEL
            # Most of these do not exist; a remove of a missing model is a
            # harmless negative ack, same best-effort contract as _remove_terrain.
            self._gz.request(self._svc_remove, req, Entity, Boolean, self._ack_ms)

    def _remove_terrain(self) -> None:
        for name in self._terrain_models:
            req = Entity()
            req.name = name
            req.type = Entity.MODEL
            self._gz.request(self._svc_remove, req, Entity, Boolean, self._ack_ms)
        self._terrain_models = []
