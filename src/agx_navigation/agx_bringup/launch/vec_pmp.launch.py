from launch import LaunchDescription
from launch.actions import (
    GroupAction,
    DeclareLaunchArgument,
)
from launch.substitutions import (
    LaunchConfiguration,
)
from launch_ros.actions import (
    Node,
    SetParameter,
)
from launch_ros.descriptions import ParameterFile


def generate_launch_description():
    declared_args = [
        DeclareLaunchArgument(
            "nav2_configured_params",
            description="Ready-to-use ros2 params file for nav2 nodes. "
            "Currently is only used for costmaps. "
            "You can use the same file as for the nav2 mode.",
        ),
    ]
    sim = LaunchConfiguration("sim")
    configured_params = ParameterFile(
        LaunchConfiguration("nav2_configured_params"),
        allow_substs=True,
    )

    remappings = [("tf", "/tf"), ("tf_static", "/tf_static"), ("map", "/map")]
    costmaps = GroupAction(
        actions=[
            SetParameter("use_sim_time", sim),
            Node(
                package="nav2_lifecycle_manager",
                executable="lifecycle_manager",
                name="lifecycle_manager_navigation",
                parameters=[
                    {
                        "autostart": True,
                        "node_names": [
                            "global_costmap/global_costmap",
                            "local_costmap/local_costmap",
                        ],
                        "bond_timeout": 0.0,
                    }
                ],
                output="screen",
            ),
            Node(
                package="nav2_costmap_2d",
                executable="nav2_costmap_2d",
                name="local_costmap",
                namespace="local_costmap",
                parameters=[configured_params],
                remappings=remappings,
                output="screen",
            ),
            Node(
                package="nav2_costmap_2d",
                executable="nav2_costmap_2d",
                name="global_costmap",
                namespace="global_costmap",
                parameters=[configured_params],
                remappings=remappings,
                output="screen",
            ),
        ],
    )

    vector_field = Node(
        package="agx_planning",
        executable="vector_field",
        output="screen",
        name="vector_field",
        parameters=[
            {
                "map_frame": "map",
                "robot_frame": "base_link",
                "allow_unknown": True,
                "viz_subsample": 4,
                "use_sim_time": sim,
            }
        ],
    )

    return LaunchDescription(
        declared_args
        + [
            costmaps,
            vector_field,
        ]
    )
