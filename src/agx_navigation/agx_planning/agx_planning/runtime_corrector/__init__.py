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
    import rclpy

    from .node import WheelCorrectorNode

    rclpy.init(args=args)
    node = WheelCorrectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
