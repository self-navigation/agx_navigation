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

import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    OpaqueFunction,
    SetEnvironmentVariable,
)
from launch.substitutions import (
    LaunchConfiguration,
)
from launch_ros.actions import (
    Node,
    LoadComposableNodes,
    SetParameter,
)
from launch_ros.descriptions import ComposableNode, ParameterFile

from agx_bringup import RewrittenYaml, Topics, cfg_file

# Local controllers the comparison runs against (launch arg `nav2_controller`).
# mppi is the long-standing default here and keeps this file's behaviour
# byte-identical when the arg is left alone; dwb and rpp are the other two
# standard nav2 local planners, and the paper's baseline set. Each non-mppi
# value loads config/nav2_controller_<name>.yaml as an overlay AFTER the main
# nav2_params.yaml, so only the controller_server subtree is replaced -- the
# costmaps, the Smac2D global planner, the BT navigator and the velocity
# smoother are shared by every arm.
CONTROLLERS = ("mppi", "dwb", "rpp")


def generate_launch_description():
    declared_args = [
        DeclareLaunchArgument(
            "nav2_controller",
            default_value="mppi",
            description=(
                "Which local controller FollowPath runs: "
                + " | ".join(CONTROLLERS)
                + ". dwb and rpp are loaded as parameter overlays on top of "
                "nav2_params.yaml (config/nav2_controller_<name>.yaml), so the "
                "global planner and costmaps stay identical across arms."
            ),
        ),
        DeclareLaunchArgument(
            "nav2_profile",
            default_value="",
            description=(
                "Optional parameter profile loaded AFTER nav2_params.yaml for "
                "every nav2 node (config/nav2_profile_<name>.yaml). Empty keeps "
                "the default stack; compare_static is the stage-1 baseline."
            ),
        ),
    ]
    return LaunchDescription(declared_args + [OpaqueFunction(function=_launch_setup)])


def _launch_setup(context):
    # `sim` stays a substitution (resolved when the Node is visited); the
    # controller choice must be concrete NOW to pick the overlay file.
    sim = LaunchConfiguration("sim")
    controller = LaunchConfiguration("nav2_controller").perform(context).strip().lower()
    if controller not in CONTROLLERS:
        raise RuntimeError(
            f"nav2_controller:='{controller}' is not one of {'|'.join(CONTROLLERS)}")

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

    configured_params = ParameterFile(
        RewrittenYaml(
            source_file=cfg_file("nav2_params.yaml"),
            root_key="",
            param_rewrites=param_substitutions,
            value_rewrites=yaml_substitutions,
            convert_types=True,
        ),
        allow_substs=True,
    )

    # The profile is a later file handed to EVERY node, so it may retune the
    # costmaps (which read their params through controller_server /
    # planner_server) and the collision monitor alike.
    from ament_index_python.packages import get_package_share_directory

    profile = LaunchConfiguration("nav2_profile").perform(context).strip()
    base_params = [configured_params]
    if profile:
        profile_file = os.path.join(
            get_package_share_directory("agx_bringup"),
            "config", f"nav2_profile_{profile}.yaml")
        if not os.path.isfile(profile_file):
            raise RuntimeError(f"nav2 profile missing: {profile_file}")
        base_params.append(ParameterFile(profile_file, allow_substs=True))

    # The controller overlay is a SECOND, LATER parameter file, so its
    # controller_server subtree wins over nav2_params.yaml's. Only
    # controller_server reads FollowPath, so only it needs the overlay; the
    # other composables are handed the base file only, so an MPPI-only key can
    # never leak into a node that does not declare it.
    if controller != "mppi":
        overlay = os.path.join(
            get_package_share_directory("agx_bringup"),
            "config", f"nav2_controller_{controller}.yaml")
        if not os.path.isfile(overlay):
            raise RuntimeError(f"nav2 controller overlay missing: {overlay}")
        controller_params = base_params + [ParameterFile(overlay, allow_substs=True)]
    else:
        controller_params = base_params

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
        parameters=[
            *controller_params,
            {
                "autostart": True,
                "use_sim_time": sim,
            },
        ],
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
                        parameters=controller_params,
                        remappings=remappings
                        + [
                            ("cmd_vel", "cmd_vel_nav"),
                            ("/odom", Topics.ODOM_FILTERED),
                        ],
                    ),
                    ComposableNode(
                        package="nav2_smoother",
                        plugin="nav2_smoother::SmootherServer",
                        name="smoother_server",
                        parameters=base_params,
                        remappings=remappings,
                    ),
                    ComposableNode(
                        package="nav2_planner",
                        plugin="nav2_planner::PlannerServer",
                        name="planner_server",
                        parameters=base_params,
                        remappings=remappings,
                    ),
                    ComposableNode(
                        package="nav2_route",
                        plugin="nav2_route::RouteServer",
                        name="route_server",
                        parameters=base_params,
                        remappings=remappings,
                    ),
                    ComposableNode(
                        package="nav2_behaviors",
                        # Jazzy renamed the composable class (upstream
                        # navigation_launch.py: behavior_server::BehaviorServer;
                        # the nav2_behaviors:: prefix is Humble-era and fails
                        # to load with "Failed to find class with the
                        # requested plugin name").
                        plugin="behavior_server::BehaviorServer",
                        name="behavior_server",
                        parameters=base_params,
                        remappings=remappings + [("cmd_vel", "cmd_vel_nav")],
                    ),
                    ComposableNode(
                        package="nav2_bt_navigator",
                        plugin="nav2_bt_navigator::BtNavigator",
                        name="bt_navigator",
                        parameters=base_params,
                        remappings=remappings,
                    ),
                    ComposableNode(
                        package="nav2_waypoint_follower",
                        plugin="nav2_waypoint_follower::WaypointFollower",
                        name="waypoint_follower",
                        parameters=base_params,
                        remappings=remappings,
                    ),
                    ComposableNode(
                        package="nav2_velocity_smoother",
                        plugin="nav2_velocity_smoother::VelocitySmoother",
                        name="velocity_smoother",
                        parameters=base_params,
                        remappings=remappings + [("cmd_vel", "cmd_vel_nav")],
                    ),
                    ComposableNode(
                        package="nav2_collision_monitor",
                        plugin="nav2_collision_monitor::CollisionMonitor",
                        name="collision_monitor",
                        parameters=base_params,
                        remappings=remappings,
                    ),
                    ComposableNode(
                        package="opennav_docking",
                        plugin="opennav_docking::DockingServer",
                        name="docking_server",
                        parameters=base_params,
                        remappings=remappings,
                    ),
                    ComposableNode(
                        package="nav2_lifecycle_manager",
                        plugin="nav2_lifecycle_manager::LifecycleManager",
                        name="lifecycle_manager_navigation",
                        parameters=[
                            {
                                "autostart": True,
                                "node_names": lifecycle_nodes,
                            }
                        ],
                    ),
                ],
            ),
        ],
    )

    return [
        stdout_linebuf_envvar,
        nav2_container,
        load_composable_nodes,
    ]
