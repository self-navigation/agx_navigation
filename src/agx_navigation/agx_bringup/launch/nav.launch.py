from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.substitutions import (
    LaunchConfiguration,
    EqualsSubstitution,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition
from launch_ros.actions import Node
from agx_bringup import RewrittenYaml, Topics, cfg_file, launch_file

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

    # Create our own temporary YAML files that include substitutions
    param_substitutions = {"autostart": "true"}

    yaml_substitutions = {
        "LASERSCAN_TOPIC": Topics.SCAN,
        "POINTCLOUD": Topics.POINTS,
        "ROBOT_CONTROL_TOPIC": Topics.CMD_VEL,
        "ODOM_TOPIC": Topics.ODOM_FILTERED,
        "ASSISTED_TELEOP_TOPIC": Topics.CMD_VEL_ASSISTED,
        "DEPTH_CAMERA_TOPIC": f"{Topics.CAMERA_DEPTH_POINTS}/downsampled",
    }

    # RewrittenYaml: Adds namespace to the parameters file as a root key
    # Note: Make sure that all frames are correctly namespaced in the parameters file
    # Do not add namespace to topics in the parameters file, as they will be remapped
    # by the root key only if they are not prefixed with a forward slash.
    # e.g. 'map' will be remapped to '/<namespace>/map', but '/map' will not be remapped.
    # IMPORTANT: to make your yaml file dynamic you can refer to humble branch under
    # nav2_bringup/launch/bringup_launch.py to see how the parameters file is configured
    # using ReplaceString <robot_namespace>
    configured_params = RewrittenYaml(
        source_file=cfg_file("nav2_params.yaml"),
        root_key="",
        param_rewrites=param_substitutions,
        value_rewrites=yaml_substitutions,
        convert_types=True,
    )

    stdout_linebuf_envvar = SetEnvironmentVariable(
        "RCUTILS_LOGGING_BUFFERED_STREAM", "1"
    )

    launch_nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(launch_file("nav2")),
        launch_arguments={
            "configured_params": configured_params,
        }.items(),
        condition=IfCondition(
            EqualsSubstitution(LaunchConfiguration("nav_mode"), NAV2_NAV_MODE)
        ),
    )
    launch_custom = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(launch_file("vec_pmp")),
        launch_arguments={
            "nav2_configured_params": configured_params,
        }.items(),
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
            stdout_linebuf_envvar,
            launch_nav2,
            launch_custom,
            frontier_explorer,
        ]
    )
