"""Check field callbacks with ROS message and node stubs."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

_rclpy_mock = sys.modules.pop("rclpy", None)
try:
    from agx_planning.vector_field import FieldUpdater, VectorFieldConfig
finally:
    if _rclpy_mock is not None:
        sys.modules["rclpy"] = _rclpy_mock

with patch.object(sys.modules["rclpy.node"], "Node", object):
    from agx_planning.vector_field.node import NodeConfig, VectorFieldNode


@pytest.fixture
def node():
    node = VectorFieldNode.__new__(VectorFieldNode)
    node.node_cfg = NodeConfig()
    node.field_cfg = VectorFieldConfig(early_exit_enable=False)
    node._field_updater = FieldUpdater()
    node._field_result = None
    node.map_array = np.zeros((21, 21), dtype=np.int8)
    node.map_array[[0, -1], :] = 100
    node.map_array[:, [0, -1]] = 100
    node.map_msg = SimpleNamespace(info=SimpleNamespace(
        width=21, height=21, resolution=0.1,
        origin=SimpleNamespace(position=SimpleNamespace(x=0.0, y=0.0)),
    ))
    node.current_goal = SimpleNamespace(
        pose=SimpleNamespace(position=SimpleNamespace(x=1.5, y=1.0)),
    )
    node._get_current_pose_2d = lambda _: (0.5, 1.0)
    node.get_logger = MagicMock()
    node._publish_planner_data = MagicMock()
    node._publish_cost_to_go_grid = MagicMock()
    node._publish_arrows = MagicMock()
    return node


def test_repeated_request_reuses_and_publishes(node):
    node._recompute_and_publish()
    assert node._field_updater.last_update == "full_recompute"
    node._recompute_and_publish()
    assert node._field_updater.last_update == "field_reuse"
    assert node._publish_planner_data.call_count == 2
    node.current_goal.pose.position.x = 1.6
    node._recompute_and_publish()
    assert node._field_updater.last_update == "map_reuse"
    assert node._publish_planner_data.call_count == 3


@pytest.mark.parametrize("change", ["outside", "occupied", "unknown", "removed"])
def test_invalid_goal_does_not_leave_a_live_result(node, change):
    node._recompute_and_publish()
    assert node._field_result is not None
    if change == "outside":
        node.current_goal.pose.position.x = 3.0
    elif change in ("occupied", "unknown"):
        node.map_array[10, 15] = 100 if change == "occupied" else -1
    else:
        node._goal_cb(SimpleNamespace(header=SimpleNamespace(frame_id="")))
        assert node._field_updater.last_update == "empty"
    node._recompute_and_publish()
    assert node._field_result is None
    assert node._publish_planner_data.call_count == 1
