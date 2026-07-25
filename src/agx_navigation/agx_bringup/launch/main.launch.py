from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.substitutions import (
    LaunchConfiguration,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition, LaunchConfigurationEquals
from agx_bringup import launch_file


def generate_launch_description():
    declared_args = [
        DeclareLaunchArgument(
            "sim",
            default_value="false",
            description="Run in Gazebo sim (spawns the robot and uses sim time).",
        ),
        DeclareLaunchArgument(
            "map_source",
            default_value="slam",
            description=(
                "Where /map and map->odom come from. "
                "slam -- rtabmap builds the map online (the real behaviour); "
                "static -- serve a pre-baked map of the floor and pin map->odom "
                "to identity. The static path needs no sensors, so it pairs with "
                "sim_sensors:=false for a much faster sim; use it for controller "
                "work, where a deterministic map matters more than mapping."
            ),
        ),
    ]

    sim = LaunchConfiguration("sim")

    scout_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(launch_file("robot_control")),
        launch_arguments={}.items(),
    )

    gz_sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(launch_file("gz_sim")),
        launch_arguments={}.items(),
        condition=IfCondition(sim),
    )

    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(launch_file("slam")),
        condition=LaunchConfigurationEquals("map_source", "slam"),
    )

    static_map_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(launch_file("static_map")),
        condition=LaunchConfigurationEquals("map_source", "static"),
    )

    # The 10 s delay exists to let the sensor pipeline settle before rtabmap
    # starts consuming it. The static map has nothing to wait for, so it comes up
    # promptly -- which is most of why a plan can be requested immediately.
    delayed_slam_launch = TimerAction(period=10.0, actions=[slam_launch])

    return LaunchDescription(
        declared_args
        + [
            gz_sim_launch,
            scout_launch,
            delayed_slam_launch,
            static_map_launch,
        ]
    )
