from .grid import VectorFieldGrid
from .field import (
    SpeedConfig,
    CutLocusConfig,
    VectorFieldResult,
    world_to_grid,
    grid_to_world,
    compute_field,
    pack_field_array,
)
from .node import VectorFieldNode

__all__ = [
    "VectorFieldGrid",
    "SpeedConfig",
    "CutLocusConfig",
    "VectorFieldResult",
    "world_to_grid",
    "grid_to_world",
    "compute_field",
    "pack_field_array",
    "VectorFieldNode",
]


import rclpy


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
