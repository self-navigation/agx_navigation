#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import time
import math
import sys

class OdomEcho(Node):
    def __init__(self):
        super().__init__('odom_echo')
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
    
    def odom_callback(self, msg):
        print(msg)

if __name__ == '__main__':
    rclpy.init()
    node = OdomEcho()
    print("running")
    rate = node.create_rate(10)  # 10 Hz loop for processing
    start_time = time.time()
    while rclpy.ok() and (time.time() - start_time) < 30.0:
        rclpy.spin_once(node)  # Process callbacks once per iteration
        time.sleep(0.1)
    rclpy.shutdown()