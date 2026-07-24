"""Runtime corrector: the only writer of /wheel_velocity_controller/commands.

Nothing heavy is imported at package import time on purpose (same principle as
rl_corrector): `tvlqr` is pure numpy and must stay importable -- by unit tests
and by offline analysis -- without ROS present. The node, which needs rclpy and
the message packages, is imported inside main().

  tvlqr -- neighboring-optimal (TVLQR) feedback about the planned trajectory (pure)
  node  -- the ROS node: mode handling, playback, and the _emit()/_correct() seam
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # static analysers only; never imported at runtime
    from .node import WheelCorrectorNode

__all__ = [
    "WheelCorrectorNode",
    "main",
]


def __getattr__(name):
    """Lazily expose WheelCorrectorNode, so `from agx_planning.runtime_corrector
    import WheelCorrectorNode` still works without making ROS a hard import for
    the pure modules."""
    if name == "WheelCorrectorNode":
        from .node import WheelCorrectorNode

        return WheelCorrectorNode
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def main(args=None):
    """Console-script entry point (setup.py maps `runtime_corrector` here).

    Delegates to node.main rather than duplicating the spin/shutdown dance: the
    copy that used to live here lacked node.main's `if rclpy.ok()` guard, so a
    SIGTERM (which makes rclpy shut the context down itself) raised RCLError out
    of the finally block. That matters more than it looks -- the velocity
    controller LATCHES its last command, so a shutdown path that throws is a
    shutdown path that can leave the wheels spinning.
    """
    from .node import main as node_main

    node_main(args)


if __name__ == "__main__":
    main()
