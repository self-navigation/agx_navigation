"""Mock ROS2 modules so pure-Python tests can import agx_planning submodules
without a ROS2 installation."""

import sys
from unittest.mock import MagicMock

_ROS_MODULES = [
    "rclpy",
    "rclpy.action",
    "rclpy.action.client",
    "rclpy.node",
    "rclpy.qos",
    "rclpy.time",
    "geometry_msgs",
    "geometry_msgs.msg",
    "nav_msgs",
    "nav_msgs.msg",
    "std_msgs",
    "std_msgs.msg",
    "builtin_interfaces",
    "builtin_interfaces.msg",
    "visualization_msgs",
    "visualization_msgs.msg",
    "tf2_ros",
    "tf_transformations",
    "agx_planning_msgs",
    "agx_planning_msgs.action",
    "agx_planning.utils",
]

for _mod in _ROS_MODULES:
    sys.modules.setdefault(_mod, MagicMock())
