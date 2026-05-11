from dataclasses import fields, replace
from typing import Any
from rclpy.node import Node


def declare_and_load_dataclass(node: Node, instance: Any, prefix: str = "") -> Any:
    """Declare every dataclass field as a ROS2 parameter and return a new
    instance populated from the parameter values.

    The dataclass instance's current values become the parameter defaults.
    Field types are inferred from the runtime values (Python types of the
    defaults), so the dataclass should use concrete types -- str, float,
    int, bool, list -- rather than typing-module aliases.

    A new instance is returned via dataclasses.replace; the input is left
    unmodified. This is the same idea as Pydantic BaseSettings or hydra's
    config dataclasses, scoped to ROS2's flat parameter API.
    """
    updates: dict = {}
    for f in fields(instance):
        name = prefix + f.name
        default = getattr(instance, f.name)
        # ROS2 declare_parameter infers descriptor type from the default's
        # Python type. The dataclass default has the correct type, so this
        # round-trip is safe.
        node.declare_parameter(name, default)
        updates[f.name] = node.get_parameter(name).value
    return replace(instance, **updates)
