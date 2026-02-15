import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import math
from scipy.interpolate import interp1d
import numpy as np


class Interpolator(Node):
    def __init__(self):
        super().__init__("laserscan_interpolator")
        
        # Declare parameters
        self.declare_parameter("input_topic", "/scan")
        self.declare_parameter("output_topic", "/interpolated_scan")
        self.declare_parameter("target_degree_inc", 0.5)
        
        input_topic = self.get_parameter("input_topic").value
        output_topic = self.get_parameter("output_topic").value
        self.target_angle_inc = math.radians(self.get_parameter("target_degree_inc").value)

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

    def resample(self, source_list, interpolation_factor):
        source_array = np.array(source_list)
        target_length = round(len(source_list) * interpolation_factor)
        x_original = np.linspace(0, 1, len(source_list))

        valid = ~np.isnan(source_array)
        
        if np.sum(valid) < 2:
            return [np.nan] * target_length  # Not enough valid points

        interpolator = interp1d(x_original[valid], source_array[valid], kind="linear", fill_value="extrapolate")
        x_new = np.linspace(0, 1, target_length)
        resampled = interpolator(x_new)
        
        return resampled.tolist()

    def callback(self, msg: LaserScan):
        if not msg.ranges or len(msg.ranges) < 2:
            self.publisher.publish(msg)
            return

        current_inc = msg.angle_increment
        target_inc = self.target_angle_inc
        interp_factor = current_inc / target_inc

        new_ranges = self.resample(msg.ranges, interp_factor)
        new_intensities = None
        if msg.intensities:
            new_intensities = self.resample(msg.intensities, interp_factor)

        new_inc = (msg.angle_max - msg.angle_min) / (len(new_ranges) - 1)
        # print(f"{current_inc=} {target_inc=} {interp_factor=} {new_inc=}")
        
        new_msg = LaserScan()
        new_msg.header = msg.header
        new_msg.angle_min = msg.angle_min
        new_msg.angle_max = msg.angle_max
        new_msg.angle_increment = new_inc
        new_msg.time_increment = msg.time_increment / interp_factor
        new_msg.scan_time = msg.scan_time
        new_msg.range_min = msg.range_min
        new_msg.range_max = msg.range_max
        new_msg.ranges = new_ranges
        if new_intensities:
            new_msg.intensities = new_intensities
        
        # print("received", len(msg.ranges), "sent", len(new_msg.ranges))
        # print(new_msg.ranges)
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

if __name__ == "__main__":
    main()
