"""Drop-in replacement for slam.launch.py that serves a pre-baked map.

Selected with map_source:=static. Serves /map from the baked floor map
(rasterized from the Gazebo collision meshes by bake_floor_map.py) and
publishes map->odom as identity, so the robot runs on raw odometry. Nothing
here consumes the lidar or cameras, so the sim can run with SIM_SENSORS=false.
The map never updates; use the SLAM path to test navigation itself.
"""

from agx_bringup import launch_file
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    declared_args = [
        DeclareLaunchArgument(
            "floor_number",
            default_value="3",
            description="Which floor's baked map to serve. Kept in step with the "
                        "floor gz_sim.launch.py spawns.",
        ),
        DeclareLaunchArgument(
            "map_yaml",
            default_value="",
            description="Explicit map YAML path, overriding floor_number.",
        ),
    ]

    sim = LaunchConfiguration("sim")

    publish_map = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("rudn_ordjo_building"),
                    "launch",
                    "publish_map.launch.py",
                ]
            )
        ),
        launch_arguments={
            "floor_number": LaunchConfiguration("floor_number"),
            "map_yaml": LaunchConfiguration("map_yaml"),
            "use_sim_time": sim,
        }.items(),
    )

    # Stands in for the SLAM correction. Identity: see the module docstring for
    # what that costs. The EKF still supplies odom->base_link.
    map_to_odom = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_map_to_odom",
        output="screen",
        arguments=[
            "--x", "0", "--y", "0", "--z", "0",
            "--roll", "0", "--pitch", "0", "--yaw", "0",
            "--frame-id", "map",
            "--child-frame-id", "odom",
        ],
        parameters=[{"use_sim_time": sim}],
    )

    nav_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(launch_file("nav")),
    )

    # Small delay so /map and the transform are up before the vector field
    # starts asking for them. It subscribes with transient-local durability so a
    # late start is survivable, but this keeps the logs free of spurious waits.
    delayed_nav_launch = TimerAction(period=3.0, actions=[nav_launch])

    return LaunchDescription(
        declared_args
        + [
            publish_map,
            map_to_odom,
            delayed_nav_launch,
        ]
    )
