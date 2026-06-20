from .node import WheelCorrectorNode

__all__ = [
    "WheelCorrectorNode",
]

import rclpy


def main(args=None):
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
