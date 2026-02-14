from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
)


def generate_launch_description():
    declared_args = []

    joint_state_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
        ],
        output="screen",
        parameters=[{"use_sim_time": LaunchConfiguration("sim")}],
    )

    diff_drive_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "diff_drive_controller",
            "--param-file",
            PathJoinSubstitution(
                [
                    FindPackageShare("scout_description"),
                    "config",
                    "diff_drive_controller.yaml",
                ]
            ),
            "--controller-ros-args",
            # Has to be specified with a list to be treated as one argument
            [
                "--remap diff_drive_controller/odom:=",
                LaunchConfiguration("odom_topic_name"),
                " ",
                "--remap diff_drive_controller/cmd_vel:=",
                LaunchConfiguration("motion_cmd_topic_name"),
            ],
        ],
        output="screen",
        parameters=[{"use_sim_time": LaunchConfiguration("sim")}],
    )

    robot_spawner = Node(
        package="ros_gz_sim",
        executable="create",
        name="scout_spawner",
        output="screen",
        arguments=[
            "-name",
            "scout_mini",
            "-topic",
            "/robot_description",
            "-allow_renaming",
            "true",
            "-x",
            "-23",
            "-y",
            "-5",
            "-z",
            "0.5",
        ],
    )

    return LaunchDescription(
        declared_args
        + [
            joint_state_spawner,
            diff_drive_spawner,
            robot_spawner,
        ]
    )
