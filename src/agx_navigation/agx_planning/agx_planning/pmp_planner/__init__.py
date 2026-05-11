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


def main(args=None):
    rclpy.init(args=args)
    node = PlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
