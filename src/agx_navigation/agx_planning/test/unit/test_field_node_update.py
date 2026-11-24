"""Check field callbacks with ROS message and node stubs."""

import ast
from dataclasses import fields, replace
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

_rclpy_mock = sys.modules.pop("rclpy", None)
try:
    from agx_planning.vector_field import FieldUpdater, VectorFieldConfig
    from agx_planning.vector_field import field as vf
finally:
    if _rclpy_mock is not None:
        sys.modules["rclpy"] = _rclpy_mock

class _NodeStub:
    def __init__(self, name):
        self.node_name = name
        self.get_logger = MagicMock()
        self.create_subscription = MagicMock()
        self.create_publisher = MagicMock()
        self.create_timer = MagicMock()


with patch.object(sys.modules["rclpy.node"], "Node", _NodeStub):
    from agx_planning.vector_field import node as node_module
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


def test_default_factory_keeps_cached_backend():
    cfg = NodeConfig()
    assert cfg.field_backend == "cached"
    assert cfg.field_checkpoint_interval == 10000
    assert cfg.field_checkpoint_growth == 1
    assert cfg.field_max_checkpoints == 4
    assert cfg.field_cache_mib == 256
    assert cfg.make_updater()._eikonal is None
    assert VectorFieldConfig().early_exit_enable is True


@pytest.mark.parametrize("count,budget", [(3, 8), (0, 8), (3, 0)])
def test_incremental_factory_passes_bounded_settings(count, budget):
    cfg = NodeConfig(field_backend="incremental", field_checkpoint_interval=7,
                     field_checkpoint_growth=4, field_max_checkpoints=count,
                     field_cache_mib=budget)
    controller = cfg.make_updater()._eikonal
    assert controller.interval == 7
    assert controller.growth == 4
    assert controller.max_checkpoints == count
    assert controller.cache_bytes == budget * 1024**2


@pytest.mark.parametrize("settings,name", [
    ({"field_backend": "full"}, "field_backend"),
    ({"field_checkpoint_interval": 0}, "field_checkpoint_interval"),
    ({"field_checkpoint_interval": 1.5}, "field_checkpoint_interval"),
    ({"field_checkpoint_growth": 0}, "field_checkpoint_growth"),
    ({"field_checkpoint_growth": "4"}, "field_checkpoint_growth"),
    ({"field_max_checkpoints": -1}, "field_max_checkpoints"),
    ({"field_max_checkpoints": True}, "field_max_checkpoints"),
    ({"field_cache_mib": -1}, "field_cache_mib"),
    ({"field_cache_mib": 2.5}, "field_cache_mib"),
])
def test_factory_validates_even_unused_cached_settings(settings, name):
    with pytest.raises(ValueError, match=name):
        NodeConfig(**settings).make_updater()


@pytest.mark.parametrize("backend", ["cached", "incremental"])
def test_startup_builds_configured_updater_and_logs_backend(monkeypatch, backend):
    cfg = NodeConfig(field_backend=backend, field_checkpoint_interval=7,
                     field_checkpoint_growth=4, field_max_checkpoints=2,
                     field_cache_mib=8)
    loader = MagicMock(side_effect=[cfg, VectorFieldConfig()])
    monkeypatch.setattr(node_module, "declare_and_load_dataclass", loader)
    actual = VectorFieldNode()
    assert actual.node_name == "vector_field"
    assert (actual._field_updater._eikonal is not None) == (backend == "incremental")
    if backend == "incremental":
        assert actual._field_updater._eikonal.growth == 4
    assert actual.create_subscription.call_count == 2
    assert actual.create_publisher.call_count == 4
    actual.create_timer.assert_called_once()
    message = actual.get_logger().info.call_args.args[0]
    assert f"field_backend={backend}" in message
    assert "early_exit_enable=True" in message
    assert "checkpoint_growth=4" in message


def test_invalid_startup_settings_fail_before_tf_setup(monkeypatch):
    loader = MagicMock(side_effect=[NodeConfig(field_checkpoint_growth=0), VectorFieldConfig()])
    buffer = MagicMock()
    monkeypatch.setattr(node_module, "declare_and_load_dataclass", loader)
    monkeypatch.setattr(node_module, "Buffer", buffer)
    with pytest.raises(ValueError, match="field_checkpoint_growth"):
        VectorFieldNode()
    buffer.assert_not_called()


def incremental_node(node, early_exit=False):
    node.node_cfg = replace(node.node_cfg, field_backend="incremental",
                            field_checkpoint_interval=10, field_checkpoint_growth=4,
                            field_cache_mib=4)
    node.field_cfg = VectorFieldConfig(early_exit_enable=early_exit, centring=True,
                                       centring_cap=0.2, speed_profile="exponential")
    node._field_updater = node.node_cfg.make_updater()
    node.map_array = np.zeros((41, 81), dtype=np.int8)
    node.map_array[[0, -1], :] = 100
    node.map_array[:, [0, -1]] = 100
    node.map_msg.info.width, node.map_msg.info.height = 81, 41
    node.current_goal.pose.position.x, node.current_goal.pose.position.y = 1.5, 2.0
    node._get_current_pose_2d = lambda _: (0.5, 2.0)
    return node


def assert_fresh_field(node):
    expected, _ = vf.compute_field(node.map_array, 15, 20, 0.1, 0.0, 0.0,
                                    node.field_cfg, 5, 20,
                                    node.node_cfg.occupancy_threshold,
                                    node.node_cfg.allow_unknown)
    for item in fields(vf.VectorFieldResult):
        np.testing.assert_equal(getattr(node._field_result, item.name),
                                getattr(expected, item.name), err_msg=item.name)


def test_incremental_map_callbacks_keep_exact_fields_and_publication_order(node):
    node = incremental_node(node)
    updater = node._field_updater
    published = MagicMock()
    published.attach_mock(node._publish_planner_data, "planner")
    published.attach_mock(node._publish_cost_to_go_grid, "cost")
    published.attach_mock(node._publish_arrows, "arrows")
    for index, value in enumerate((0, 100, 0)):
        node.map_array[5:8, 65] = value
        msg = SimpleNamespace(info=node.map_msg.info, data=node.map_array.ravel().tolist())
        node._map_cb(msg)
        assert node._field_updater is updater
        assert_fresh_field(node)
        if index:
            assert updater.last_fmm["status"] == "replay"
            assert updater.last_fmm["reused_batches"] > 0
            assert updater.last_speed["status"] == "local_update"
    assert published.mock_calls == [call.planner(), call.cost(), call.arrows()] * 3
    messages = [item.args[0] for item in node.get_logger().info.call_args_list]
    assert any("speed=local_update/changed_region" in message
               and "fmm=replay/checkpoint_replayed" in message
               and "reused_batches=" in message for message in messages)

    node.get_logger().info.reset_mock()
    node._goal_cb(SimpleNamespace(header=SimpleNamespace(frame_id="map"),
                                  pose=node.current_goal.pose))
    assert updater.last_update == "field_reuse"
    assert updater.last_fmm == updater.last_speed == {}
    message = next(item.args[0] for item in node.get_logger().info.call_args_list
                   if item.args[0].startswith("field_reuse:"))
    assert "speed=" not in message and "fmm=" not in message and "reused_batches=" not in message
    assert published.mock_calls == [call.planner(), call.cost(), call.arrows()] * 4


def test_incremental_early_exit_keeps_partial_field_fallback(node):
    node = incremental_node(node, early_exit=True)
    node._recompute_and_publish()
    assert_fresh_field(node)
    stats = node._field_updater.last_fmm
    assert stats["status"] == "full_recompute" and stats["reason"] == "early_exit"
    assert stats["reused_batches"] == 0 and stats["checkpoint_count"] == 0
    assert np.isnan(node._field_result.travel_time[node.map_array == 0]).any()
    messages = [item.args[0] for item in node.get_logger().info.call_args_list]
    assert any("fmm=full_recompute/early_exit, reused_batches=0" in message for message in messages)


@pytest.mark.parametrize("change", ["removed", "unknown"])
def test_existing_invalid_request_paths_clear_incremental_cache(node, change):
    node = incremental_node(node)
    node._recompute_and_publish()
    assert node._field_updater._eikonal._state is not None
    goal = node.current_goal
    if change == "removed":
        node._goal_cb(SimpleNamespace(header=SimpleNamespace(frame_id="")))
    else:
        node.map_array[20, 15] = -1
        node._recompute_and_publish()
    assert node._field_result is None
    assert node._field_updater._eikonal._state is None
    assert node._field_updater.last_fmm == node._field_updater.last_speed == {}
    assert node._publish_planner_data.call_count == 1
    node.current_goal = goal
    node.map_array[20, 15] = 0
    node._recompute_and_publish()
    assert_fresh_field(node)
    assert node._field_updater.last_fmm["status"] == "full_recompute"
    assert node._field_updater.last_fmm["reused_batches"] == 0


def test_launch_defaults_and_typed_field_parameter_wiring():
    path = Path(__file__).resolve().parents[3] / "agx_bringup/launch/vec_pmp.launch.py"
    source = path.read_text()
    compile(source, str(path), "exec")
    tree = ast.parse(source)
    defaults = {}
    for item in ast.walk(tree):
        if isinstance(item, ast.Call) and isinstance(item.func, ast.Name) and item.func.id == "DeclareLaunchArgument":
            default = next(k.value for k in item.keywords if k.arg == "default_value")
            defaults[item.args[0].value] = default.value
    assert {name: defaults[name] for name in (
        "field_backend", "field_checkpoint_interval", "field_checkpoint_growth",
        "field_max_checkpoints", "field_cache_mib", "field_early_exit",
    )} == dict(field_backend="cached", field_checkpoint_interval="10000",
              field_checkpoint_growth="1", field_max_checkpoints="4",
              field_cache_mib="256", field_early_exit="true")
    assert defaults["corrector"] == "identity"
    expected = {"field_backend": ("field_backend", "str"),
                "field_checkpoint_interval": ("field_checkpoint_interval", "int"),
                "field_checkpoint_growth": ("field_checkpoint_growth", "int"),
                "field_max_checkpoints": ("field_max_checkpoints", "int"),
                "field_cache_mib": ("field_cache_mib", "int"),
                "early_exit_enable": ("field_early_exit", "bool")}
    found = {}
    for item in ast.walk(tree):
        if isinstance(item, ast.Dict):
            for key, value in zip(item.keys, item.values):
                if isinstance(key, ast.Constant) and key.value in expected:
                    assert isinstance(value, ast.Call) and value.func.id == "ParameterValue"
                    assert value.args[0].func.id == "LaunchConfiguration"
                    kind = next(k.value.id for k in value.keywords if k.arg == "value_type")
                    found[key.value] = (value.args[0].args[0].value, kind)
    assert found == expected
