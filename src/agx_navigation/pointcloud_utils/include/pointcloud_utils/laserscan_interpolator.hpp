#ifndef POINTCLOUD_UTILS__LASERSCAN_INTERPOLATOR_HPP_
#define POINTCLOUD_UTILS__LASERSCAN_INTERPOLATOR_HP

#include <cmath>
#include <limits>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <vector>

namespace pointcloud_utils {

class LaserScanInterpolator : public rclcpp::Node {
public:
  explicit LaserScanInterpolator(const rclcpp::NodeOptions &options);

private:
  struct Subrange {
    int start_idx; // first valid index in the original scan
    int end_idx;   // last  valid index in the original scan
    std::vector<int> indices;
    std::vector<float> values;
  };

  void scan_callback(sensor_msgs::msg::LaserScan::UniquePtr msg);

  std::vector<float> resample(const std::vector<float> &source,
                              float original_angle_step,
                              float target_angle_step,
                              float distance_threshold) const;

  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr subscription_;
  rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr publisher_;

  float target_angle_increment_; // radians
  float distance_threshold_;     // metres
};

} // namespace pointcloud_utils

#endif // POINTCLOUD_UTILS__LASERSCAN_INTERPOLATOR_HPP_
