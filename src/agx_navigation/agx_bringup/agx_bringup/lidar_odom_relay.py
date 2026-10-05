"""Fill in a pose covariance on rf2o laser odometry so the EKF can fuse it.

rf2o publishes all-zero covariances and a zero lateral twist. The EKF takes its
pose in differential mode instead, which yields a body-frame velocity including
the lateral component that wheel odometry cannot observe.
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
