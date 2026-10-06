"""Launch the fixture with the GMPC tracker in place of a nav stack.

Includes agx_bringup's main.launch.py with `nav_mode` forced to `none`, so no
planner, corrector or Nav2 node comes up, and adds the GMPC follow_path server.
GMPC's Python dependencies (casadi, manifpy) live in a venv named by
`gmpc_venv` (default ~/.venvs/gmpc), whose site-packages is prepended to the
node's PYTHONPATH.

With `supervisor:=true` the stack runs under `nav_mode:=nav2` instead, the
server is renamed `follow_path_gmpc` and publishes to `/cmd_vel_nav_out` for
supervisor_node to gate. `gmpc_nav2:=true` gives the same stack without the
supervisor.
"""

import glob
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from agx_bringup import Topics, launch_file


SUPERVISED_CMD_TOPIC = "/cmd_vel_nav_out"  # == nav2.launch.py's
GMPC_ACTION_NAV2 = "follow_path_gmpc"


def _flags(context):
    supervised = LaunchConfiguration("supervisor").perform(context).lower() == "true"
    nav2 = supervised or LaunchConfiguration("gmpc_nav2").perform(context).lower() == "true"
    return supervised, nav2


def _base(context):
    supervised, nav2 = _flags(context)
    args = {"nav_mode": "nav2" if nav2 else "none", "frontier": "false",
            "supervisor": "true" if supervised else "false"}
    if nav2:
        args.update(supervisor_mode="follow_path",
                    supervisor_follow_path_action=GMPC_ACTION_NAV2)
    return [IncludeLaunchDescription(
        PythonLaunchDescriptionSource(launch_file("main")),
        launch_arguments=args.items())]


def _node(context):
    venv = os.path.expanduser(LaunchConfiguration("gmpc_venv").perform(context))
    site = glob.glob(os.path.join(venv, "lib", "python3*", "site-packages"))
    env = {}
    if site:
        env["PYTHONPATH"] = site[0] + os.pathsep + os.environ.get("PYTHONPATH", "")
    sim = LaunchConfiguration("sim").perform(context).lower() == "true"
    supervised, nav2 = _flags(context)
    return [Node(
        package="agx_baselines",
        executable="gmpc_controller",
        name="gmpc_controller",
        output="screen",
        additional_env=env,
        parameters=[{
            "use_sim_time": sim,
            "cmd_vel_topic": SUPERVISED_CMD_TOPIC if supervised else Topics.CMD_VEL,
            "action_name": GMPC_ACTION_NAV2 if nav2 else "follow_path",
            "odom_topic": Topics.ODOM_FILTERED,
            "control_rate": float(LaunchConfiguration("gmpc_rate").perform(context)),
        }],
    )]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("gmpc_venv", default_value="~/.venvs/gmpc",
                              description="venv holding casadi + manifpy"),
        DeclareLaunchArgument("gmpc_rate", default_value="20.0",
                              description="GMPC control rate, Hz (paper: 50 sim / 5 real)"),
        DeclareLaunchArgument("supervisor", default_value="false",
                              description="supervisory layer (#42); implies gmpc_nav2"),
        DeclareLaunchArgument("gmpc_nav2", default_value="false",
                              description="run under nav_mode:=nav2 (planner_server up), "
                                          "GMPC on follow_path_gmpc"),
        OpaqueFunction(function=_base),
        OpaqueFunction(function=_node),
    ])
