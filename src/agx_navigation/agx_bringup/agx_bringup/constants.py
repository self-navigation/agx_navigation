class Topics:
    # --------------
    #    SENSORS
    # --------------
    CAMERA_COLOR_IMAGE = "/camera/color/image_raw"
    CAMERA_COLOR_INFO = "/camera/color/camera_info"

    CAMERA_DEPTH_IMAGE = "/camera/depth/image_raw"
    CAMERA_DEPTH_POINTS_SIM_INTERMEDIATE = "/camera/depth/points/skewed"
    CAMERA_DEPTH_POINTS = "/camera/depth/points"
    CAMERA_DEPTH_DOWNSAMPLED = "/camera/depth/points/downsampled"
    CAMERA_DEPTH_INFO = "/camera/depth/camera_info"

    CAMERA_RGBD_IMAGE = "/camera/rgbd/image_raw"
    CAMERA_RGBD_POINTS = "/camera/rgbd/points"
    CAMERA_RGBD_INFO = "/camera/rgbd/camera_info"

    LIDAR_POINTS = "/lidar/points"
    LIDAR_DOWNSAMPLED = "/lidar/points/downsampled"

    POINTS = "/points"
    SCAN = "/scan"

    # --------------
    #  LOCALIZATION
    # --------------
    ODOM = "/odom"
    ODOM_FILTERED = "/odom/filtered"

    IMU = "/imu/data"
    IMU_FILTERED = f"/imu/data/filtered"

    MAGNETIC_FIELD = "/imu/mag"

    # --------------
    #    STATUS
    # --------------
    STATUS = "/scout_status"
    JOINT_STATES = "/joint_states"
    ROBOT_DESCRIPTION = "/robot_description"

    # --------------
    #    CONTROL
    # --------------
    # Base cmd_vel that goes directly to the scout node
    CMD_VEL_RAW = "/cmd_vel_raw"
    # cmd_vel that is listened by the repeater to circumwent autostop in the chassis
    CMD_VEL = "/cmd_vel"
    # cmd_vel that is listened by the nav2 behavior server for collision-safe movement
    CMD_VEL_ASSISTED = "/cmd_vel_assisted_teleop"
    CMD_LIGHT = "/light_control"
    # Raw 4-wheel velocity input to twist_to_wheels (future planner interface)
    CMD_WHEEL = "/cmd_wheel"
    # Output of twist_to_wheels → JointGroupVelocityController
    WHEEL_COMMANDS = "/wheel_velocity_controller/commands"
