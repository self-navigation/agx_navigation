from .strategies import RecoveryConfig, default_strategies
from .corrector import TrajectoryCorrectorNode
from .config import CorrectorConfig

__all__ = [
    "TrajectoryCorrectorNode",
    "CorrectorConfig",
    "RecoveryConfig",
    "default_strategies",
]

import rclpy


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryCorrectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
