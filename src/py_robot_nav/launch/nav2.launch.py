# Copyright (c) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    SetEnvironmentVariable,
)
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import (
    Node,
    LoadComposableNodes,
    SetParameter,
)
from launch_ros.descriptions import ComposableNode, ParameterFile
from launch_ros.substitutions import FindPackageShare
from py_robot_nav.launch import RewrittenYaml


def generate_launch_description():
    declared_args = [
        DeclareLaunchArgument(
            "autostart",
            default_value="true",
            description="Automatically startup the nav2 stack",
        ),
        DeclareLaunchArgument(
            "laserscan_topic",
            default_value="/lidar/laserscan",
            description="LaserScan topic for 2D SLAM",
        ),
        DeclareLaunchArgument(
            "pointcloud_topic",
            default_value="/lidar/points",
            description="PointCloud2 topic for 3D SLAM",
        ),
    ]

    autostart = LaunchConfiguration("autostart")
    laserscan_topic = LaunchConfiguration("laserscan_topic")
    pointcloud_topic = LaunchConfiguration("pointcloud_topic")
    robot_contro_topic = LaunchConfiguration("motion_cmd_topic_name")
    odom_topic = PathJoinSubstitution(
        [LaunchConfiguration("odom_topic_name"), "filtered"]
    )
    sim = LaunchConfiguration("sim")

    lifecycle_nodes = [
        "controller_server",
        "smoother_server",
        "planner_server",
        "route_server",
        "behavior_server",
        "velocity_smoother",
        "collision_monitor",
        "bt_navigator",
        "waypoint_follower",
        "docking_server",
    ]

    # Map fully qualified names to relative ones so the node's namespace can be prepended.
    remappings = [("/tf", "tf"), ("/tf_static", "tf_static")]

    # Create our own temporary YAML files that include substitutions
    param_substitutions = {"autostart": autostart}

    yaml_substitutions = {
        "LASERSCAN_TOPIC": laserscan_topic,
        "POINTCLOUD": pointcloud_topic,
        "ROBOT_CONTROL_TOPIC": robot_contro_topic,
        "ODOM_TOPIC": odom_topic,
    }

    params_file = PathJoinSubstitution(
        [FindPackageShare("py_robot_nav"), "config", "nav2_params.yaml"]
    )

    # RewrittenYaml: Adds namespace to the parameters file as a root key
    # Note: Make sure that all frames are correctly namespaced in the parameters file
    # Do not add namespace to topics in the parameters file, as they will be remapped
    # by the root key only if they are not prefixed with a forward slash.
    # e.g. 'map' will be remapped to '/<namespace>/map', but '/map' will not be remapped.
    # IMPORTANT: to make your yaml file dynamic you can refer to humble branch under
    # nav2_bringup/launch/bringup_launch.py to see how the parameters file is configured
    # using ReplaceString <robot_namespace>
    configured_params = ParameterFile(
        RewrittenYaml(
            source_file=params_file,
            root_key="",
            param_rewrites=param_substitutions,
            value_rewrites=yaml_substitutions,
            convert_types=True,
        ),
        allow_substs=True,
    )

    stdout_linebuf_envvar = SetEnvironmentVariable(
        "RCUTILS_LOGGING_BUFFERED_STREAM", "1"
    )

    # WARN: starting container node separately (not with ComposableNodeContainer)
    # so that parameter file values propagate correctly.
    # Specifically local and global costmaps are spawned by the controller_server,
    # and with ComposableNodeContainer their config doesn't get passed.
    nav2_container = Node(
        name="nav2_container",
        package="rclcpp_components",
        executable="component_container_isolated",
        parameters=[configured_params, {"autostart": autostart}],
        remappings=remappings,
        output="screen",
    )

    load_composable_nodes = GroupAction(
        actions=[
            SetParameter("use_sim_time", sim),
            LoadComposableNodes(
                target_container="nav2_container",
                composable_node_descriptions=[
                    ComposableNode(
                        package="nav2_controller",
                        plugin="nav2_controller::ControllerServer",
                        name="controller_server",
                        parameters=[configured_params],
                        remappings=remappings + [("cmd_vel", "cmd_vel_nav")],
                    ),
                    ComposableNode(
                        package="nav2_smoother",
                        plugin="nav2_smoother::SmootherServer",
                        name="smoother_server",
                        parameters=[configured_params],
                        remappings=remappings,
                    ),
                    ComposableNode(
                        package="nav2_planner",
                        plugin="nav2_planner::PlannerServer",
                        name="planner_server",
                        parameters=[configured_params],
                        remappings=remappings,
                    ),
                    ComposableNode(
                        package="nav2_route",
                        plugin="nav2_route::RouteServer",
                        name="route_server",
                        parameters=[configured_params],
                        remappings=remappings,
                    ),
                    ComposableNode(
                        package="nav2_behaviors",
                        plugin="behavior_server::BehaviorServer",
                        name="behavior_server",
                        parameters=[configured_params],
                        remappings=remappings + [("cmd_vel", "cmd_vel_nav")],
                    ),
                    ComposableNode(
                        package="nav2_bt_navigator",
                        plugin="nav2_bt_navigator::BtNavigator",
                        name="bt_navigator",
                        parameters=[configured_params],
                        remappings=remappings,
                    ),
                    ComposableNode(
                        package="nav2_waypoint_follower",
                        plugin="nav2_waypoint_follower::WaypointFollower",
                        name="waypoint_follower",
                        parameters=[configured_params],
                        remappings=remappings,
                    ),
                    ComposableNode(
                        package="nav2_velocity_smoother",
                        plugin="nav2_velocity_smoother::VelocitySmoother",
                        name="velocity_smoother",
                        parameters=[configured_params],
                        remappings=remappings + [("cmd_vel", "cmd_vel_nav")],
                    ),
                    ComposableNode(
                        package="nav2_collision_monitor",
                        plugin="nav2_collision_monitor::CollisionMonitor",
                        name="collision_monitor",
                        parameters=[configured_params],
                        remappings=remappings,
                    ),
                    ComposableNode(
                        package="opennav_docking",
                        plugin="opennav_docking::DockingServer",
                        name="docking_server",
                        parameters=[configured_params],
                        remappings=remappings,
                    ),
                    ComposableNode(
                        package="nav2_lifecycle_manager",
                        plugin="nav2_lifecycle_manager::LifecycleManager",
                        name="lifecycle_manager_navigation",
                        parameters=[
                            {"autostart": autostart, "node_names": lifecycle_nodes}
                        ],
                    ),
                ],
            ),
        ],
    )

    return LaunchDescription(
        declared_args
        + [
            stdout_linebuf_envvar,
            nav2_container,
            load_composable_nodes,
        ]
    )
