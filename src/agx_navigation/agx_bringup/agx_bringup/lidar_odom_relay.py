"""Re-stamp rf2o's laser odometry with a usable covariance for the EKF (#34).

rf2o_laser_odometry publishes all-zero covariances and hard-codes its twist's
lateral velocity to 0, so neither field can be fused as is. The EKF instead
takes rf2o's POSE in differential mode (ekf_params via robot_control's
lidar_odom flag): successive poses become a body-frame velocity, lateral
component included -- the skid term wheel odometry cannot observe, and the
reason this source exists. This node only fills the pose covariance.
"""
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node


class LidarOdomRelay(Node):
    def __init__(self):
        super().__init__("lidar_odom_relay")
        sxy = self.declare_parameter("sigma_xy", 0.02).value
        syaw = self.declare_parameter("sigma_yaw", 0.01).value
        cov = [0.0] * 36
        cov[0] = cov[7] = sxy ** 2
        cov[14] = cov[21] = cov[28] = 1e3   # z, roll, pitch: unused (two_d_mode)
        cov[35] = syaw ** 2
        self._cov = cov
        self._pub = self.create_publisher(Odometry, "odom_out", 10)
        self.create_subscription(Odometry, "odom_in", self._cb, 10)

    def _cb(self, msg: Odometry):
        msg.pose.covariance = self._cov
        self._pub.publish(msg)


def main():
    rclpy.init()
    rclpy.spin(LidarOdomRelay())
