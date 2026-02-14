import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import math
from scipy.interpolate import interp1d
import numpy as np


class Interpolator(Node):
    def __init__(self):
        super().__init__('laserscan_interpolator')
        
        # Declare parameters
        self.declare_parameter('input_topic', '/scan')
        self.declare_parameter('output_topic', '/interpolated_scan')
        self.declare_parameter('interpolation_factor', 2.0)
        
        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.interpolation_factor = self.get_parameter('interpolation_factor').value
        
        if self.interpolation_factor <= 0.0:
            self.get_logger().error('Interpolation factor must be a positive float > 0')
            raise ValueError('Invalid interpolation factor')
        
        self.subscription = self.create_subscription(
            LaserScan,
            input_topic,
            self.callback,
            10
        )
        
        self.publisher = self.create_publisher(
            LaserScan,
            output_topic,
            10
        )
    
    def resample(self, source_list):
        target_length = round(len(source_list) * self.interpolation_factor)
        x_original = np.linspace(0, 1, len(source_list))
        
        interpolator = interp1d(x_original, source_list, kind='linear')
        x_new = np.linspace(0, 1, target_length)
        resampled = interpolator(x_new)
        
        return resampled.tolist()

    def callback(self, msg: LaserScan):
        if not msg.ranges or len(msg.ranges) < 2:
            self.publisher.publish(msg)
            return

        new_ranges = self.resample(msg.ranges)
        new_intensities = None
        if msg.intensities:
            new_intensities = self.resample(msg.intensities)

        new_inc = (msg.angle_max - msg.angle_min) / (len(new_ranges) - 1)
        
        new_msg = LaserScan()
        new_msg.header = msg.header
        new_msg.angle_min = msg.angle_min
        new_msg.angle_max = msg.angle_max
        new_msg.angle_increment = new_inc
        new_msg.time_increment = msg.time_increment / self.interpolation_factor
        new_msg.scan_time = msg.scan_time
        new_msg.range_min = msg.range_min
        new_msg.range_max = msg.range_max
        new_msg.ranges = new_ranges
        if new_intensities:
            new_msg.intensities = new_intensities
        
        # print('received', len(msg.ranges), 'sent', len(new_msg.ranges))
        self.publisher.publish(new_msg)

def main(args=None):
    rclpy.init(args=args)
    node = Interpolator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()