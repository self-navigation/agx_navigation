"""Drop-in replacement for slam.launch.py that serves a pre-baked map.

Selected by localization:=amcl|truth|none (see main.launch.py). Same contract as
the SLAM path -- provide /map and the map->odom transform, then bring up nav --
but with the map known ahead of time instead of built online.

Three ways to estimate map->odom against that map, chosen by `localization`:

  amcl  -- nav2_amcl matching the lidar against the baked grid. The realistic
           mode: this is what a robot does after a site survey, and unlike SLAM
           it is repeatable, because the map does not change between runs.
           Needs sim_sensors:=true for the lidar.
  truth -- taken from Gazebo (truth_localization.py). Sim only, exact, and the
           corrector's performance ceiling: any error left is the corrector's.
  none  -- pinned to identity, so the robot navigates on raw wheel odometry.

`none` was the original and only behaviour here, and it is kept because it needs
no sensors and so runs fastest -- but it is not a neutral choice. Wheel odometry
cannot observe slip and over-reports distance travelled (0.6-0.7 m over one
fixture run on this world). Open-loop control ignores pose and is unaffected; a
pose-feedback corrector drives to the bias, so TVLQR measured 0.74 m short of
the goal while believing it had arrived within 5 cm. Under `none` this fixture
cannot evaluate a pose-feedback corrector at all. Use `truth` to measure the
corrector, `amcl` to measure the system, and `none` only for the things that
genuinely do not care -- planner geometry, throughput, smoke tests.

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
one for testing navigation itself -- use localization:=slam for that.
"""

import math

from agx_bringup import Topics, cfg_file, launch_file
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.conditions import LaunchConfigurationEquals
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
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
        # Only read by localization:=truth, which has to find the robot among
        # every entity Gazebo publishes a pose for.
        DeclareLaunchArgument("world_name", default_value="ordjo_world"),
        DeclareLaunchArgument("model_name", default_value="scout_mini"),
        # Where the robot was spawned (world frame == map frame, see
        # truth_localization.py). amcl's initial pose must equal it: the
        # historical hard-coded (0, 0, 0) is only correct for the historical
        # spawn point, and amcl seeded elsewhere starts with its particle
        # cloud around a pose the robot is not at -- on this world's
        # near-identical corridors that can converge to the wrong corridor
        # instead of to the robot. sim_control.launch.py owns the spawner; the
        # two arguments must be given the SAME values, which is why
        # tools/fixture_up.sh passes both from one --spawn flag.
        DeclareLaunchArgument(
            "amcl_params", default_value=cfg_file("nav2_params.yaml"),
            description="extra amcl params file, layered over nav2_params.yaml "
                        "(#34 sweep); the default re-applies the base file, a no-op"),
        DeclareLaunchArgument(
            "spawn_x", default_value="0.0",
            description="must match sim_control's spawn_x"),
        DeclareLaunchArgument(
            "spawn_y", default_value="0.0",
            description="must match sim_control's spawn_y"),
        DeclareLaunchArgument(
            "spawn_yaw", default_value="0.0",
            description="must match sim_control's spawn_yaw"),
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
            # No heartbeat: the map is latched. publish_map.launch.py defaults
            # to 2.0 s, and amcl rebuilds its particle filter on every map it
            # receives, so a heartbeat made the pose jump every 2 s and
            # invalidated the 2026-09-29 overnight Nav2 comparison (#10).
            "publish_period": "0.0",
        }.items(),
    )

    localization = LaunchConfiguration("localization")

    # --- localization:=none -------------------------------------------------
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
        condition=LaunchConfigurationEquals("localization", "none"),
    )

    # --- localization:=truth ------------------------------------------------
    truth_localization = Node(
        package="agx_bringup",
        executable="truth_localization",
        name="truth_localization",
        output="screen",
        parameters=[{
            "use_sim_time": sim,
            "world_name": LaunchConfiguration("world_name"),
            "model_name": LaunchConfiguration("model_name"),
        }],
        condition=LaunchConfigurationEquals("localization", "truth"),
    )

    # --- localization:=amcl -------------------------------------------------
    # AMCL wants a 2-D LaserScan; the robot carries a 3-D lidar. slam.launch.py
    # flattens the *aggregated* cloud (lidar + depth camera), but localizing
    # against a map baked from wall geometry only needs the lidar, so this skips
    # the aggregator and the camera with it. Band and resolution match
    # slam.launch.py so the two modes see comparable scans.
    scan_from_lidar = Node(
        package="pointcloud_to_laserscan",
        executable="pointcloud_to_laserscan_node",
        name="pointcloud_to_laserscan",
        output="screen",
        remappings=[
            ("cloud_in", Topics.LIDAR_POINTS),
            ("scan", Topics.SCAN),
        ],
        parameters=[{
            "target_frame": "base_link",
            "transform_tolerance": 0.1,
            "min_height": 0.05,
            "max_height": 0.7,
            "angle_min": -math.pi,
            "angle_max": math.pi,
            "angle_increment": (2 * math.pi) / 360 / 4,
            "scan_time": 0.1,
            "range_min": 0.2,
            "range_max": 150.0,
            "use_inf": True,
            "use_sim_time": sim,
        }],
        condition=LaunchConfigurationEquals("localization", "amcl"),
    )

    # The amcl block in nav2_params.yaml has been carried for a while but never
    # launched -- the nav2 stack relies on rtabmap for map->odom. This is its
    # first use. `set_initial_pose` matters: the robot spawns at the map origin
    # (see truth_localization.py on frames), and global localization from
    # scratch in a corridor of near-identical walls is not something a fixture
    # should be asked to do.
    amcl = Node(
        package="nav2_amcl",
        executable="amcl",
        name="amcl",
        output="screen",
        parameters=[
            cfg_file("nav2_params.yaml"),
            # Sweep hook (#34): a params file layered over nav2_params.yaml's
            # amcl block. The default is the base file itself, i.e. a no-op.
            LaunchConfiguration("amcl_params"),
            {
                "use_sim_time": sim,
                "scan_topic": Topics.SCAN,
                "set_initial_pose": True,
                # Belt and braces with publish_period above: the baked map
                # never changes, so any later /map would only reset the filter.
                "first_map_only": True,
                "initial_pose.x": ParameterValue(
                    LaunchConfiguration("spawn_x"), value_type=float),
                "initial_pose.y": ParameterValue(
                    LaunchConfiguration("spawn_y"), value_type=float),
                "initial_pose.yaw": ParameterValue(
                    LaunchConfiguration("spawn_yaw"), value_type=float),
            },
        ],
        condition=LaunchConfigurationEquals("localization", "amcl"),
    )

    # amcl is a lifecycle node and stays UNCONFIGURED unless something drives
    # it. The nav2 stack has its own lifecycle manager; this one exists because
    # under vec-pmp there is no nav2 stack to borrow one from.
    amcl_lifecycle = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_localization",
        output="screen",
        parameters=[{
            "use_sim_time": sim,
            "autostart": True,
            "node_names": ["amcl"],
        }],
        condition=LaunchConfigurationEquals("localization", "amcl"),
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
            truth_localization,
            scan_from_lidar,
            amcl,
            amcl_lifecycle,
            delayed_nav_launch,
        ]
    )
