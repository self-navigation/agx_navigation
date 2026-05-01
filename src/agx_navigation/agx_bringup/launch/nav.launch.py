from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
)
from launch.substitutions import (
    LaunchConfiguration,
    EqualsSubstitution,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition
from launch_ros.actions import Node
from agx_bringup import launch_file

NAV2_NAV_MODE = "nav2"
CUSTOM_NAV_MODE = "vec-pmp"


def generate_launch_description():
    declared_args = [
        DeclareLaunchArgument(
            "nav_mode",
            default_value="nav2",
            description="Which nav stack to use. Allowed values: nav2, vec-pmp.",
        ),
    ]

    launch_nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(launch_file("nav2")),
        condition=IfCondition(
            EqualsSubstitution(LaunchConfiguration("nav_mode"), NAV2_NAV_MODE)
        ),
    )
    launch_custom = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(launch_file("vec_pmp")),
        condition=IfCondition(
            EqualsSubstitution(LaunchConfiguration("nav_mode"), CUSTOM_NAV_MODE)
        ),
    )
    frontier_explorer = Node(
        package="agx_planning",
        executable="frontier_explorer",
        name="frontier_explorer",
        parameters=[
            {
                "traversable_cost_threshold": 60,
                "replan_frequency": 2.0,
                "known_space_dilation": 4,
            }
        ],
        output="screen",
    )

    return LaunchDescription(
        declared_args
        + [
            launch_nav2,
            launch_custom,
            frontier_explorer,
        ]
    )
