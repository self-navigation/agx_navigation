from .shooting_solver import PMPShootingSolver
from .config import PlannerConfig
from .vf_grid import VectorFieldGrid
from .diagnostic import TurnDiagnosticLogger
from .planner import PlannerNode

__all__ = [
    "PMPShootingSolver",
    "PlannerConfig",
    "VectorFieldGrid",
    "TurnDiagnosticLogger",
    "PlannerNode",
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
