class Topics:
    # --------------
    #    SENSORS
    # --------------
    CAMERA_COLOR_IMAGE = "/camera/color/image_raw"
    CAMERA_COLOR_INFO = "/camera/color/camera_info"

    CAMERA_DEPTH_IMAGE = "/camera/depth/image_raw"
    CAMERA_DEPTH_POINTS_SIM_INTERMEDIATE = "/camera/depth/points/skewed"
    CAMERA_DEPTH_POINTS = "/camera/depth/points"
    CAMERA_DEPTH_INFO = "/camera/depth/camera_info"

    CAMERA_RGBD_IMAGE = "/camera/rgbd/image_raw"
    CAMERA_RGBD_POINTS = "/camera/rgbd/points"
    CAMERA_RGBD_INFO = "/camera/rgbd/camera_info"

    LIDAR_POINTS = "/lidar/points"

    POINTS = "/points"
    SCAN = "/scan"

    # --------------
    #  LOCALIZATION
    # --------------
    ODOM = "/odom"
    ODOM_FILTERED = "/odom/filtered"

    IMU = "/imu"
    IMU_FILTERED = f"/imu/filtered"

    MAGNETIC_FIELD = "/magnetic_field"

    # --------------
    #    STATUS
    # --------------
    STATUS = "/scout_status"
    JOINT_STATES = "/joint_states"
    ROBOT_DESCRIPTION = "/robot_description"

    # --------------
    #    CONTROL
    # --------------
    CMD_VEL = "/cmd_vel"
    CMD_LIGHT = "/light_control"
