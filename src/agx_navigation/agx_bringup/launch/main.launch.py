from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.substitutions import (
    LaunchConfiguration,
    NotEqualsSubstitution,
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
            "localization",
            default_value="slam",
            description=(
                "Where /map and map->odom come from. One knob, because the two "
                "are not independent -- an estimator needs a map to localize "
                "against. Four values, in decreasing order of realism:\n"
                "  slam  -- rtabmap builds the map online and localizes in it. "
                "The real deployment.\n"
                "  amcl  -- nav2_amcl localizes the lidar against the pre-baked "
                "map. Deployment after a good site survey; realistic AND "
                "repeatable, since the map is fixed.\n"
                "  truth -- the baked map, with map->odom taken from Gazebo. "
                "Sim only. The corrector's performance CEILING: use it to tell a "
                "corrector defect apart from a localization limit.\n"
                "  none  -- the baked map with map->odom pinned to identity, so "
                "the robot navigates on raw wheel odometry. Needs no sensors and "
                "so runs fastest, but odometry drift lands entirely in the "
                "measurement -- see static_map.launch.py before trusting a "
                "number from it."
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
        condition=LaunchConfigurationEquals("localization", "slam"),
    )

    # Every other mode serves the baked map; they differ only in what estimates
    # map->odom, which static_map.launch.py branches on.
    static_map_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(launch_file("static_map")),
        condition=IfCondition(
            NotEqualsSubstitution(LaunchConfiguration("localization"), "slam")),
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
