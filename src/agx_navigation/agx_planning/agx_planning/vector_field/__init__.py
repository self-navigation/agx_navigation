import importlib.util

from .grid import VectorFieldGrid
from .field import (
    SpeedConfig,
    CutLocusConfig,
    EarlyExitConfig,
    VectorFieldResult,
    world_to_grid,
    grid_to_world,
    compute_field,
    pack_field_array,
)

__all__ = [
    "VectorFieldGrid",
    "SpeedConfig",
    "CutLocusConfig",
    "EarlyExitConfig",
    "VectorFieldResult",
    "world_to_grid",
    "grid_to_world",
    "compute_field",
    "pack_field_array",
]

ROS2_AVAILABLE = importlib.util.find_spec("rclpy") is not None

if ROS2_AVAILABLE:
    import rclpy
    from .node import VectorFieldNode

    __all__.append("VectorFieldNode")

    def main(args=None):

        rclpy.init(args=args)
        node = VectorFieldNode()
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()

    if __name__ == "__main__":
        main()
