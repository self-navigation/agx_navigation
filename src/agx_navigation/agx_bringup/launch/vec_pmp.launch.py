from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    declared_args = []

    sim = LaunchConfiguration("sim")

    vector_field = Node(
        package="agx_planning",
        executable="vector_field",
        output="screen",
        name="vector_field",
        parameters=[
            {
                "map_frame": "map",
                "robot_frame": "base_link",
                "obstacle_slope_factor": 400.0,
                "allow_unknown": True,
                "use_sim_time": sim,
            }
        ],
    )

    pmp_planner = Node(
        package="agx_planning",
        executable="pmp_planner",
        output="screen",
        name="pmp_planner",
        parameters=[
            {
                "enable_stamped_cmd_vel": True,
                "use_sim_time": sim,
            }
        ],
    )

    return LaunchDescription(
        declared_args
        + [
            vector_field,
            pmp_planner,
        ]
    )
