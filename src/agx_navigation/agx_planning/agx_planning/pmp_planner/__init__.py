import importlib.util

from .shooting_solver import PMPShootingSolver
from .config import PlannerConfig
from .diagnostic import TurnDiagnosticLogger
from .rollout import (
    RolloutChunk,
    RolloutResult,
    compute_diag_values,
    goal_reached,
    parse_field_array,
    rollout_generator,
)

__all__ = [
    "PMPShootingSolver",
    "PlannerConfig",
    "TurnDiagnosticLogger",
    "RolloutResult",
    "compute_diag_values",
    "goal_reached",
    "parse_field_array",
    "rollout_generator",
    "RolloutChunk",
]


ROS2_AVAILABLE = importlib.util.find_spec("rclpy") is not None

if ROS2_AVAILABLE:
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from .node import PlannerNode

    __all__.append("PlannerNode")

    def main(args=None):
        import logging
        import sys

        # rollout.py logs per-chunk solve times through stdlib logging, which
        # has no handler under ros2 run, so INFO was silently dropped.
        _h = logging.StreamHandler(sys.stderr)
        _h.setFormatter(logging.Formatter("[pmp_rollout] %(message)s"))
        _rl = logging.getLogger("agx_planning.pmp_planner.rollout")
        _rl.addHandler(_h)
        _rl.setLevel(logging.INFO)

        rclpy.init(args=args)
        node = PlannerNode()

        # NOT MultiThreadedExecutor: under use_sim_time it busy-spins and
        # starves every callback of the GIL (see PlannerNode docstring).
        executor = SingleThreadedExecutor()
        executor.add_node(node)

        try:
            executor.spin()
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()

    if __name__ == "__main__":
        main()
