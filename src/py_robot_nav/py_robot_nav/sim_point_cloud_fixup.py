import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import TransformStamped
import tf2_ros
import tf2_sensor_msgs  # For transforming point clouds using TF
from tf2_ros import TransformException
import numpy as np

class PointCloudRotator(Node):
    """
    ROS2 node to rotate a point cloud by a fixed angle around the Y-axis.
    Subscribes to raw /camera/points, applies +90° rotation (to counter -90° downward shift),
    and republishes to /camera/points_corrected.
    
    Assumptions:
    - Input cloud is in XYZ format (fields: x, y, z).
    - Rotation is around Y-axis (pitch); adjust matrix if mismatch is around another axis.
    - Uses TF buffer for frame consistency, but applies fixed transform relative to header.frame_id.
    """
    
    def __init__(self):
        super().__init__('point_cloud_rotator')
        
        # Parameters (adjustable via ROS params if needed)
        self.declare_parameter('input_topic', '/camera/points')
        self.declare_parameter('output_topic', '/camera/points_corrected')
        self.declare_parameter('rotation_angle_deg', 90.0)  # +90° around Y to fix downward point
    
        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        self.rotation_angle_deg = self.get_parameter('rotation_angle_deg').get_parameter_value().double_value
        
        # Compute rotation matrix (around Y-axis)
        angle_rad = np.deg2rad(self.rotation_angle_deg)
        self.rotation_matrix = np.array([
            [np.cos(angle_rad), 0, np.sin(angle_rad)],
            [0, 1, 0],
            [-np.sin(angle_rad), 0, np.cos(angle_rad)]
        ])
        
        # TF buffer for potential frame transforms (though we apply relative here)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Subscriber and publisher
        self.subscription = self.create_subscription(
            PointCloud2,
            input_topic,
            self.callback,
            10  # QoS depth
        )
        self.publisher = self.create_publisher(PointCloud2, output_topic, 10)
        
        self.get_logger().info(f'Rotating point clouds from {input_topic} to {output_topic} by {self.rotation_angle_deg}° around Y.')

    def get_parameter_value(self, name: str) -> str:
        return self.get_parameter(name).get_parameter_value().string_value

    def callback(self, msg: PointCloud2):
        try:
            # Extract points as NumPy array (structured dtype for fields)
            points = self.pointcloud2_to_array(msg)
            
            # Apply rotation: Multiply XYZ points by rotation matrix
            rotated_points = np.dot(points[:, :3], self.rotation_matrix.T)  # Transpose for matrix mult
            
            # Preserve other fields (e.g., intensity or RGB as columns 3+ if present)
            if points.shape[1] > 3:
                rotated_points = np.hstack((rotated_points, points[:, 3:]))
            
            # Convert back to PointCloud2
            rotated_msg = self.array_to_pointcloud2(rotated_points, msg.header, msg.fields)
            
            # Publish
            self.publisher.publish(rotated_msg)
        
        except TransformException as ex:
            self.get_logger().warn(f'Could not transform point cloud: {ex}')
    
    def pointcloud2_to_array(self, cloud_msg: PointCloud2) -> np.ndarray:
        """Convert PointCloud2 to NumPy array (structured dtype for fields)."""
        dtype_list = [(f.name, np.float32) for f in cloud_msg.fields]  # Assume float32; adjust if uint8/intensity
        arr = np.frombuffer(cloud_msg.data, dtype=np.dtype(dtype_list))
        return arr.view(np.float32).reshape(arr.shape + (-1,))  # Reshape to N x num_fields
    
    def array_to_pointcloud2(self, points: np.ndarray, header, original_fields) -> PointCloud2:
        """Convert NumPy array back to PointCloud2, reusing original fields for consistency."""
        num_points = points.shape[0]
        point_step = points.dtype.itemsize  # Bytes per point
        data = points.astype(np.float32).tobytes()
        
        # Reuse original fields to preserve structure (e.g., if intensity/RGB present)
        fields = original_fields  # This ensures compatibility with rgbd_camera output
        
        return PointCloud2(
            header=header,
            height=1,  # Unorganized (common for rgbd_camera points)
            width=num_points,
            is_dense=True,  # No NaNs assumed; set False if filtering NaNs
            is_bigendian=False,
            fields=fields,
            point_step=point_step,
            row_step=point_step * num_points,
            data=data
        )

def main(args=None):
    rclpy.init(args=args)
    node = PointCloudRotator()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()