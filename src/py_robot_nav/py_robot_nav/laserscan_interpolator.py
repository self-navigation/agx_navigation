import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from scipy.interpolate import interp1d
import numpy as np


class Interpolator(Node):
    def __init__(self):
        super().__init__("laserscan_interpolator")

        # Declare parameters
        self.declare_parameter("input_topic", "/scan")
        self.declare_parameter("output_topic", "/interpolated_scan")
        self.declare_parameter("target_angle_increment", 0.5)
        self.declare_parameter("distance_threshold", 0.5)

        input_topic = self.get_parameter("input_topic").value
        output_topic = self.get_parameter("output_topic").value
        self.target_angle_increment = self.get_parameter("target_angle_increment").value
        self.distance_threshold = self.get_parameter("distance_threshold").value

        self.subscription = self.create_subscription(
            LaserScan,
            input_topic,
            self.callback,
            qos_profile=qos_profile_sensor_data,
        )

        self.publisher = self.create_publisher(
            LaserScan,
            output_topic,
            qos_profile=qos_profile_sensor_data,
        )

    def resample(
        self, source_list, original_angle_step, target_angle_step, distance_threshold
    ):
        """
        Resample laser scan data with intelligent handling of inf/nan values.

        Args:
            source_list: Original laser scan ranges
            original_angle_step: Angular resolution of source data (degrees)
            target_angle_step: Desired angular resolution (degrees)
            distance_threshold: Max Euclidean distance to keep points in same subrange (meters)

        Returns:
            Resampled laser scan with uniform angular resolution
        """
        source_array = np.array(source_list)
        interpolation_factor = original_angle_step / target_angle_step
        target_length = int(round((2 * np.pi) / target_angle_step))

        # Step 1: Find valid points (not inf, not nan)
        valid_mask = np.isfinite(source_array)
        valid_indices = np.where(valid_mask)[0]
        valid_values = source_array[valid_mask]

        if len(valid_values) < 2:
            return [np.inf] * target_length

        # Step 2: Split into subranges based on:
        #   a) Angular continuity (consecutive indices)
        #   b) Spatial continuity (Euclidean distance threshold)
        subranges = []
        current_start_idx = valid_indices[0]
        current_indices = [valid_indices[0]]
        current_values = [valid_values[0]]

        for i in range(1, len(valid_values)):
            # Check if there's an angular gap (non-consecutive indices)
            angular_gap = (valid_indices[i] - valid_indices[i - 1]) > 1

            # Calculate actual Euclidean distance between points
            # Using law of cosines: d² = r₁² + r₂² - 2*r₁*r₂*cos(θ)
            r1 = valid_values[i - 1]
            r2 = valid_values[i]
            angle_diff_deg = (
                valid_indices[i] - valid_indices[i - 1]
            ) * original_angle_step
            angle_diff_rad = np.deg2rad(angle_diff_deg)

            euclidean_distance = np.sqrt(
                r1**2 + r2**2 - 2 * r1 * r2 * np.cos(angle_diff_rad)
            )

            spatial_discontinuity = euclidean_distance > distance_threshold

            if spatial_discontinuity:
                # Save current subrange
                subranges.append(
                    {
                        "start_idx": current_start_idx,
                        "end_idx": valid_indices[i - 1],  # Store end index too
                        "indices": current_indices.copy(),
                        "values": current_values.copy(),
                    }
                )
                # Start new subrange
                current_start_idx = valid_indices[i]
                current_indices = [valid_indices[i]]
                current_values = [valid_values[i]]
            else:
                # Continue current subrange
                current_indices.append(valid_indices[i])
                current_values.append(valid_values[i])

        # Add the last subrange
        subranges.append(
            {
                "start_idx": current_start_idx,
                "end_idx": valid_indices[-1],
                "indices": current_indices,
                "values": current_values,
            }
        )

        # Step 3: Interpolate each subrange independently
        result = np.full(target_length, np.inf)

        for subrange in subranges:
            values = np.array(subrange["values"])
            original_length = len(values)

            # Calculate target positions for this subrange
            # Start and end positions in the target array
            start_pos = int(round(subrange["start_idx"] * interpolation_factor))
            end_pos = int(round(subrange["end_idx"] * interpolation_factor))

            # Ensure we stay within bounds
            start_pos = max(0, min(start_pos, target_length - 1))
            end_pos = max(0, min(end_pos, target_length - 1))

            if start_pos >= end_pos:
                # Single point or degenerate case
                if 0 <= start_pos < target_length:
                    result[start_pos] = values[0] if len(values) > 0 else np.inf
                continue

            # Calculate exact number of points needed to densely fill this range
            new_length = end_pos - start_pos + 1

            if original_length < 2:
                # Single point - fill entire range with same value
                result[start_pos : end_pos + 1] = values[0]
            else:
                # Interpolate to fill exactly the required positions
                x_original = np.linspace(0, 1, original_length)
                x_new = np.linspace(0, 1, new_length)

                interpolator = interp1d(
                    x_original, values, kind="linear", fill_value="extrapolate"
                )
                interpolated_values = interpolator(x_new)

                # Fill the range completely (inclusive of both start and end)
                result[start_pos : end_pos + 1] = interpolated_values

        return result.tolist()

    def callback(self, msg: LaserScan):
        if not msg.ranges or len(msg.ranges) < 2:
            self.publisher.publish(msg)
            return

        current_inc = msg.angle_increment
        target_inc = self.target_angle_increment

        new_ranges = self.resample(
            msg.ranges,
            current_inc,
            target_inc,
            self.distance_threshold,
        )
        new_intensities = None
        if msg.intensities:
            new_intensities = self.resample(
                msg.intensities,
                current_inc,
                target_inc,
                self.distance_threshold,
            )

        new_inc = (msg.angle_max - msg.angle_min) / (len(new_ranges) - 1)
        # print(f"{current_inc=} {target_inc=} {interp_factor=} {new_inc=}")

        interp_factor = current_inc / target_inc

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
