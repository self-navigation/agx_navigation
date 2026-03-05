#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>

namespace pointcloud_utils {

class PointCloudFieldStripper : public rclcpp::Node {
public:
  explicit PointCloudFieldStripper(const rclcpp::NodeOptions &options);

private:
  void callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg);
  static sensor_msgs::msg::PointField make_field(const std::string &name,
                                                 uint32_t offset);

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
};

} // namespace pointcloud_utils
