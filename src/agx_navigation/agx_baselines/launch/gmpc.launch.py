"""The fixture base plus the GMPC tracker instead of a nav stack.

Includes agx_bringup's main.launch.py unchanged (sim, robot control, the
static map / localization branch) and adds the GMPC follow_path server. All
of main's launch arguments pass straight through the shared launch context
(sim, localization, spawn_*, surface_patches, phantom_walls, ...); `nav_mode`
is forced to `none`, a value nav.launch.py's two branches both ignore, so no
planner, corrector or Nav2 node comes up. Our launch files gain nothing
baseline-specific: this file is the only place that knows GMPC exists.

Selected by `make run LAUNCH_PKG=agx_baselines LAUNCH_FILE=gmpc.launch.py`
(what tools/fixture_up.sh --nav-mode gmpc does).

GMPC's Python dependencies (casadi, manifpy) live in a venv, not the system
ROS python: `gmpc_venv` names it and its site-packages is prepended to the
node's PYTHONPATH. Default ~/.venvs/gmpc; see the package README.
"""

import glob
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from agx_bringup import Topics, launch_file


def _node(context):
    venv = os.path.expanduser(LaunchConfiguration("gmpc_venv").perform(context))
    site = glob.glob(os.path.join(venv, "lib", "python3*", "site-packages"))
    env = {}
    if site:
        env["PYTHONPATH"] = site[0] + os.pathsep + os.environ.get("PYTHONPATH", "")
    sim = LaunchConfiguration("sim").perform(context).lower() == "true"
    return [Node(
        package="agx_baselines",
        executable="gmpc_controller",
        name="gmpc_controller",
        output="screen",
        additional_env=env,
        parameters=[{
            "use_sim_time": sim,
            "cmd_vel_topic": Topics.CMD_VEL,
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
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(launch_file("main")),
            launch_arguments={"nav_mode": "none", "frontier": "false"}.items(),
        ),
        OpaqueFunction(function=_node),
    ])
