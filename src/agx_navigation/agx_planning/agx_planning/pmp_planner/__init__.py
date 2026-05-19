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
    from rclpy.executors import MultiThreadedExecutor
    from .node import PlannerNode

    __all__.append("PlannerNode")

    def main(args=None):

        rclpy.init(args=args)
        node = PlannerNode()

        executor = MultiThreadedExecutor(num_threads=2)
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
