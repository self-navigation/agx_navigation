import json

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    ExecuteProcess,
    RegisterEventHandler,
    EmitEvent,
    OpaqueFunction,
)
from launch.substitutions import (
    FindExecutable,
    PathJoinSubstitution,
    LaunchConfiguration,
    EnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node
from agx_bringup import Topics


def spawn_phantom_floor(context):
    """The floor exactly as spawn_floor.launch.py builds it, minus every collision.

    gpu_lidar renders VISUAL geometry, so the walls stay observable. Collision-
    free walls cannot be expressed through spawn_floor's own args, and that
    file lives in the rudn-ordjo-building submodule, so its builder is reused
    here rather than patched there. Both the building mesh and the world ground
    plane use friction mu=1 (the mesh by ODE default), so dropping the slab's
    collision does not change the plant.
    """
    import importlib.util
    import os
    import xml.etree.ElementTree as ET
    from ament_index_python.packages import get_package_share_directory

    pkg = get_package_share_directory("rudn_ordjo_building")
    spec = importlib.util.spec_from_file_location(
        "spawn_floor", os.path.join(pkg, "launch", "spawn_floor.launch.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    floor = LaunchConfiguration("floor_number").perform(context)
    with open(os.path.join(pkg, "models", "model_template.sdf")) as f:
        sdf = f.read().replace("{floor_num}", floor)
    sdf = mod.exclude_part(sdf, floor, 4, "center")
    sdf = mod.exclude_part(sdf, floor, 6, "right")
    root = ET.fromstring(sdf)
    for link in root.iter("link"):
        for col in link.findall("collision"):
            link.remove(col)
    sdf = ET.tostring(root, encoding="unicode").replace(
        "package://rudn_ordjo_building/", f"file://{pkg}/")
    return [Node(package="ros_gz_sim", executable="create", output="screen",
                 arguments=["-name", f"rudn_ordjo_building_floor_{floor}",
                            "-string", sdf, "-x", "23", "-y", "5", "-z", "0"])]


def launch_gz_sim(context):
    headless = LaunchConfiguration("headless")
    is_headless = headless.perform(context).lower() in ["true", "1", "yes"]

    # Directly launch GZ Sim using ExecuteProcess, to hook into process exit
    world_path = PathJoinSubstitution(
        [FindPackageShare("rudn_ordjo_building"), "worlds", "ordjo_world.world"]
    )

    cmd = [FindExecutable(name="gz"), "sim", "-v", "6", "-r", world_path, "-s"]
    if is_headless:
        cmd.append("--headless-rendering")

    gz_process_server = ExecuteProcess(
        cmd=cmd,
        output="screen",
        name="gz_sim_server",
    )

    gz_process_gui = ExecuteProcess(
        cmd=[FindExecutable(name="gz"), "sim", "-g"],
        output="screen",
        name="gz_sim_gui",
        condition=UnlessCondition(headless),
    )

    shutdown_handler = RegisterEventHandler(
        OnProcessExit(
            target_action=gz_process_server, on_exit=[EmitEvent(event=Shutdown())]
        )
    )

    return [
        gz_process_server,
        gz_process_gui,
        shutdown_handler,
    ]


def generate_launch_description():
    declared_args = [
        DeclareLaunchArgument(
            "floor_number",
            default_value="3",
            description="On which floor of the RUDN building to perform the simulation.",
        ),
        DeclareLaunchArgument(
            "headless",
            default_value="false",
            description="Enable headless rendering for gz sim.",
        ),
        DeclareLaunchArgument(
            "surface_patches",
            default_value="true",
            description="Spawn the low-friction/rough ground patches. On by "
                        "default (they are what the corrector exists to handle); "
                        "set false to isolate planner geometry from slip.",
        ),
        DeclareLaunchArgument(
            "phantom_walls",
            default_value="false",
            description="Sim-only diagnostic (#34): spawn the building with its "
                        "visuals but NO collision, so the lidar (and amcl) see "
                        "the walls while the robot drives through them. The "
                        "world's ground plane carries the robot.",
        ),
    ]

    headless = LaunchConfiguration("headless")
    floor_number = LaunchConfiguration("floor_number")
    sim = LaunchConfiguration("sim")

    # Set GZ_SIM_RESOURCE_PATH to enable model:// resolution
    set_gz_resource_path = SetEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH",
        value=[
            PathJoinSubstitution([FindPackageShare("scout_description"), ".."]),
            ":",
            EnvironmentVariable("GZ_SIM_RESOURCE_PATH", default_value=""),
        ],
    )
    set_gz_system_plugin_path = SetEnvironmentVariable(
        name="GZ_SIM_SYSTEM_PLUGIN_PATH",
        value="/opt/ros/jazzy/lib/",
    )
    set_display = SetEnvironmentVariable(
        name="DISPLAY",
        value="",
        condition=IfCondition(headless),
    )

    gz_sim = OpaqueFunction(function=launch_gz_sim)

    spawn_floor = IncludeLaunchDescription(
        condition=UnlessCondition(LaunchConfiguration("phantom_walls")),
        launch_description_source=PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("rudn_ordjo_building"),
                    "launch",
                    "spawn_floor.launch.py",
                ]
            )
        ),
        launch_arguments={
            "floor_number": floor_number,
            "x": "23",
            "y": "5",
        }.items(),
    )

    # Edit the JSON array below to add custom physics surface patches.
    # Each object requires: x, y, z, width, length, profile.
    # Optional: yaw (radians), name.
    # Profiles: "slippery", "icy", "rough", "directional_x", "directional_y"
    #
    # The robot spawns at (0, 0) in world frame; the building floor mesh is
    # offset to (23, 5) so that (0, 0) falls inside it.  Place patches near
    # (0, 0) to have them close to the robot start position.
    surface_patches_json = json.dumps([
        {"x":  0.0, "y":  -2.0, "width": 3.0, "length": 2.0, "profile": "slippery"},
        {"x":  0.0, "y":  0.0, "width": 2.0, "length": 2.0, "profile": "icy"},
        {"x": -4.0, "y": -2.0, "width": 4.0, "length": 2.0, "profile": "rough"},
        {"x":  3.0, "y": -2.0, "width": 3.0, "length": 3.0, "profile": "directional_x"},
        {"x":  6.0, "y": -2.0, "width": 3.0, "length": 3.0, "profile": "directional_y"},
    ])

    # The patches sit on top of the robot's start: it spawns at (0, 0), which is
    # inside the "icy" patch, and a drive along y=0 crosses the directional ones
    # at x=3 and x=6. That is deliberate for corrector work -- slip is the whole
    # phenomenon being corrected -- but it confounds PLANNER debugging, where a
    # wall strike then has two candidate causes (a bad plan, or a slip excursion
    # off a good one) and no way to tell them apart. Turn them off to isolate
    # planner geometry, back on to test the corrector.
    spawn_surface_patches = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("rudn_ordjo_building"),
                    "launch",
                    "spawn_surface_patches.launch.py",
                ]
            )
        ),
        launch_arguments={"patches": surface_patches_json}.items(),
        condition=IfCondition(LaunchConfiguration("surface_patches")),
    )

    gz_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="gz_bridge",
        output="screen",
        arguments=[
            # GZ->ROS
            # Clock
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            # NOTE: ground-truth pose is deliberately NOT bridged here. The
            # Pose_V -> TFMessage conversion drops the entity names (every
            # frame_id and child_frame_id arrives empty), leaving no way to pick
            # the robot out of the hundreds of entities in pose/info. Consumers
            # that need it (agx_planning/run_recorder.py, the RL GazeboBridge)
            # subscribe over gz-transport directly and match Pose_V.pose[].name.
            # Camera
            "/d435_camera/color/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
            "/d435_camera/color/image_raw@sensor_msgs/msg/Image[gz.msgs.Image",
            "/d435_camera/depth/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
            "/d435_camera/depth/image_raw@sensor_msgs/msg/Image[gz.msgs.Image",
            "/d435_camera/depth/image_raw/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",
            # LiDAR Point Cloud
            "/lidar/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",
            # IMU and Magnetometer
            "/imu/data@sensor_msgs/msg/Imu[gz.msgs.IMU",
            "/imu/mag@sensor_msgs/msg/MagneticField[gz.msgs.Magnetometer",
        ],
        parameters=[
            {
                "lazy": True,
                "use_sim_time": sim,
            }
        ],
        # For sim-to-life unity remapping topics to a common name.
        # NOTE: remapping depth points to an intermediate topic,
        # because its transform is fixed later in sim_control.launch.py
        remappings=[
            ("/d435_camera/color/image_raw", Topics.CAMERA_COLOR_IMAGE),
            ("/d435_camera/color/camera_info", Topics.CAMERA_COLOR_INFO),
            ("/d435_camera/depth/image_raw", Topics.CAMERA_DEPTH_IMAGE),
            (
                "/d435_camera/depth/image_raw/points",
                Topics.CAMERA_DEPTH_POINTS_SIM_INTERMEDIATE,
            ),
            ("/d435_camera/depth/camera_info", Topics.CAMERA_DEPTH_INFO),
            ("/imu/data", Topics.IMU),
            ("/imu/mag", Topics.MAGNETIC_FIELD),
        ],
    )

    return LaunchDescription(
        declared_args
        + [
            set_gz_resource_path,
            set_gz_system_plugin_path,
            set_display,
            gz_sim,
            gz_bridge,
            spawn_floor,
            OpaqueFunction(function=spawn_phantom_floor,
                           condition=IfCondition(LaunchConfiguration("phantom_walls"))),
            spawn_surface_patches,
        ]
    )
