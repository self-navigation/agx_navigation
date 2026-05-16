from .shooting_solver import PMPShootingSolver
from .config import PlannerConfig
from .vf_grid import VectorFieldGrid
from .diagnostic import TurnDiagnosticLogger
from .rollout import (
    RolloutChunk,
    RolloutResult,
    compute_diag_values,
    goal_reached,
    parse_field_array,
    rollout_generator,
)
from .node import PlannerNode

__all__ = [
    "PMPShootingSolver",
    "PlannerConfig",
    "VectorFieldGrid",
    "TurnDiagnosticLogger",
    "PlannerNode",
    "RolloutResult",
    "compute_diag_values",
    "goal_reached",
    "parse_field_array",
    "rollout_generator",
    "RolloutChunk",
]


import rclpy
from rclpy.executors import MultiThreadedExecutor


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
