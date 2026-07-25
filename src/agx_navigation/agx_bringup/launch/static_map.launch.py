"""Drop-in replacement for slam.launch.py that serves a pre-baked map.

Selected with map_source:=static (see main.launch.py). Same contract as the SLAM
path -- provide /map and the map->odom transform, then bring up nav -- but with
the map known ahead of time instead of built online.

WHY
---
rtabmap is a bad fixture for testing the runtime corrector:

  * it needs the robot to drive before any map exists, so a plan cannot be
    requested from a standing start;
  * it intermittently fails to initialise ("Missing visual features...") on this
    world's untextured walls, and then /map is simply never published;
  * it produces a slightly different map every run, so two corrector runs are
    never strictly comparable.

The baked map removes all three. It is generated from the very meshes Gazebo
collides against, by rudn_ordjo_building's tools/bake_floor_map.py.

A second, larger benefit: nothing here consumes the lidar or the cameras. The
whole pointcloud-processing pipeline that slam.launch.py sets up is gone, so the
sim can run with SIM_SENSORS=false and a far better realtime factor. For
corrector work -- which only needs ground-truth pose and the planned trajectory
-- the sensors were pure overhead.

WHAT THIS GIVES UP
------------------
The map is static: obstacles that are not in the mesh (the surface patches are
flat, so they never appear anyway) will not show up, and the robot cannot
discover anything. That is the right trade for controller testing and the wrong
one for testing navigation itself -- use map_source:=slam for that.

Because there is no localisation, map->odom is published as identity. The robot
therefore trusts its own odometry completely: its believed pose drifts away from
ground truth exactly as odometry does. That is *deliberate* here -- the TVLQR
corrector is evaluated against Gazebo ground truth, not against /odom -- but it
does mean the planned path is laid out in a frame that slowly slides relative to
the world.
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
