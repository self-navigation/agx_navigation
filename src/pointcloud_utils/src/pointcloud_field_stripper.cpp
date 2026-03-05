// src/pointcloud_field_stripper.cpp
#include "pointcloud_utils/pointcloud_field_stripper.hpp"

#include <cstring>

#include <rclcpp_components/register_node_macro.hpp>

namespace pointcloud_utils {

PointCloudFieldStripper::PointCloudFieldStripper(
    const rclcpp::NodeOptions &options)
    : Node("pointcloud_field_stripper", options) {
  pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("output", 10);
  sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      "input", rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
        callback(msg);
      });
}

void PointCloudFieldStripper::callback(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg) {
  // Find source offsets for the fields we want to keep
  int src_x = -1, src_y = -1, src_z = -1, src_i = -1;
  for (const auto &f : msg->fields) {
    if (f.name == "x")
      src_x = f.offset;
    else if (f.name == "y")
      src_y = f.offset;
    else if (f.name == "z")
      src_z = f.offset;
    else if (f.name == "intensity")
      src_i = f.offset;
  }

  if (src_x < 0 || src_y < 0 || src_z < 0) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                         "Input cloud missing x/y/z fields, skipping");
    return;
  }

  const bool has_intensity = (src_i >= 0);
  const uint32_t dst_step = has_intensity ? 16 : 12;
  const uint32_t n_points = msg->width * msg->height;

  auto out = std::make_unique<sensor_msgs::msg::PointCloud2>();
  out->header = msg->header;
  out->height = 1;
  out->width = n_points;
  out->is_bigendian = false;
  out->point_step = dst_step;
  out->row_step = dst_step * n_points;
  out->is_dense = msg->is_dense;
  out->data.resize(out->row_step);

  // Define output fields with proper 4-byte alignment
  out->fields.push_back(make_field("x", 0));
  out->fields.push_back(make_field("y", 4));
  out->fields.push_back(make_field("z", 8));
  if (has_intensity) {
    out->fields.push_back(make_field("intensity", 12));
  }

  // Copy field by field using memcpy (safe regardless of source alignment)
  const uint8_t *src = msg->data.data();
  uint8_t *dst = out->data.data();
  const uint32_t src_step = msg->point_step;

  for (uint32_t i = 0; i < n_points; ++i) {
    const uint8_t *sp = src + i * src_step;
    uint8_t *dp = dst + i * dst_step;
    std::memcpy(dp + 0, sp + src_x, 4);
    std::memcpy(dp + 4, sp + src_y, 4);
    std::memcpy(dp + 8, sp + src_z, 4);
    if (has_intensity) {
      std::memcpy(dp + 12, sp + src_i, 4);
    }
  }

  pub_->publish(std::move(out));
}

sensor_msgs::msg::PointField
PointCloudFieldStripper::make_field(const std::string &name, uint32_t offset) {
  sensor_msgs::msg::PointField f;
  f.name = name;
  f.offset = offset;
  f.datatype = sensor_msgs::msg::PointField::FLOAT32;
  f.count = 1;
  return f;
}

} // namespace pointcloud_utils

RCLCPP_COMPONENTS_REGISTER_NODE(pointcloud_utils::PointCloudFieldStripper)
