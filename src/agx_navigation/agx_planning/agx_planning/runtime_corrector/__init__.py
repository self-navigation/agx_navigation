from .strategies import RecoveryConfig, default_strategies
from .corrector import TrajectoryCorrectorNode, CorrectorNodeConfig
from .config import CorrectorConfig
from .deviation_detector import DeviationDetector
from .correction_controller import CorrectionController, CorrectionResult, ExitKind
from .visualization import TrajectoryVisualizer
from .trajectory_buffer import TrajectoryBuffer, PlaybackSample

__all__ = [
    "TrajectoryCorrectorNode",
    "CorrectorConfig",
    "CorrectorNodeConfig",
    "RecoveryConfig",
    "default_strategies",
    "DeviationDetector",
    "CorrectionController",
    "CorrectionResult",
    "ExitKind",
    "TrajectoryVisualizer",
    "TrajectoryBuffer",
    "PlaybackSample",
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
