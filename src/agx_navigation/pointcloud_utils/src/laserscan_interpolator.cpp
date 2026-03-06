#include "pointcloud_utils/laserscan_interpolator.hpp"
#include <algorithm>
#include <cmath>
#include <rclcpp_components/register_node_macro.hpp>

namespace pointcloud_utils {

LaserScanInterpolator::LaserScanInterpolator(const rclcpp::NodeOptions &options)
    : Node("laserscan_interpolator", options) {
  this->declare_parameter<std::string>("input_topic", "/scan");
  this->declare_parameter<std::string>("output_topic", "/interpolated_scan");
  this->declare_parameter<float>(
      "target_angle_increment",
      std::round((2.0f * static_cast<float>(M_PI)) / 360.0f)); // radians
  this->declare_parameter<float>("distance_threshold", 0.5f);  // metres

  const auto input_topic = this->get_parameter("input_topic").as_string();
  const auto output_topic = this->get_parameter("output_topic").as_string();
  target_angle_increment_ = static_cast<float>(
      this->get_parameter("target_angle_increment").as_double());
  distance_threshold_ =
      static_cast<float>(this->get_parameter("distance_threshold").as_double());

  // Use sensor-data QoS (best-effort, volatile) - matches typical lidar drivers
  const auto qos = rclcpp::SensorDataQoS();

  // Subscribe with a unique_ptr callback signature so the middleware can
  // hand us ownership of the message without copying.
  subscription_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
      input_topic, qos,
      std::bind(&LaserScanInterpolator::scan_callback, this,
                std::placeholders::_1));

  publisher_ =
      this->create_publisher<sensor_msgs::msg::LaserScan>(output_topic, qos);

  RCLCPP_INFO(this->get_logger(),
              "Interpolator ready  [in: %s → out: %s | target_inc: %.4f rad | "
              "dist_thresh: %.2f m]",
              input_topic.c_str(), output_topic.c_str(),
              target_angle_increment_, distance_threshold_);
}

void LaserScanInterpolator::scan_callback(
    sensor_msgs::msg::LaserScan::UniquePtr msg) {
  if (msg->ranges.size() < 2) {
    publisher_->publish(std::move(msg));
    return;
  }

  const float current_inc = msg->angle_increment;
  const float target_inc = target_angle_increment_;
  const float interp_factor = current_inc / target_inc;

  // Resample ranges
  auto new_ranges =
      resample(msg->ranges, current_inc, target_inc, distance_threshold_);

  // Resample intensities if present
  std::vector<float> new_intensities;
  if (!msg->intensities.empty()) {
    new_intensities = resample(msg->intensities, current_inc, target_inc,
                               distance_threshold_);
  }

  // Build output message via unique_ptr for zero-copy publish
  auto out = std::make_unique<sensor_msgs::msg::LaserScan>();

  out->header = msg->header;
  out->angle_min = msg->angle_min;
  out->angle_max = msg->angle_max;
  out->angle_increment = (new_ranges.size() > 1)
                             ? (msg->angle_max - msg->angle_min) /
                                   static_cast<float>(new_ranges.size() - 1)
                             : msg->angle_increment;
  out->time_increment = msg->time_increment / interp_factor;
  out->scan_time = msg->scan_time;
  out->range_min = msg->range_min;
  out->range_max = msg->range_max;
  out->ranges = std::move(new_ranges);

  if (!new_intensities.empty()) {
    out->intensities = std::move(new_intensities);
  }

  publisher_->publish(std::move(out));
}

std::vector<float> LaserScanInterpolator::resample(
    const std::vector<float> &source, float original_angle_step,
    float target_angle_step, float distance_threshold) const {
  const float interp_factor = original_angle_step / target_angle_step;
  const int target_length = static_cast<int>(
      std::round((2.0f * static_cast<float>(M_PI)) / target_angle_step));

  // Collect valid (finite) points
  std::vector<int> valid_indices;
  std::vector<float> valid_values;
  valid_indices.reserve(source.size());
  valid_values.reserve(source.size());

  for (size_t i = 0; i < source.size(); ++i) {
    if (std::isfinite(source[i])) {
      valid_indices.push_back(static_cast<int>(i));
      valid_values.push_back(source[i]);
    }
  }

  if (valid_values.size() < 2) {
    return std::vector<float>(target_length,
                              std::numeric_limits<float>::infinity());
  }

  // Split into subranges on spatial discontinuities
  std::vector<Subrange> subranges;
  Subrange current;
  current.start_idx = valid_indices[0];
  current.indices = {valid_indices[0]};
  current.values = {valid_values[0]};

  for (size_t i = 1; i < valid_values.size(); ++i) {
    const float r1 = valid_values[i - 1];
    const float r2 = valid_values[i];

    // Angle gap in radians between the two valid points
    const float angle_diff =
        static_cast<float>(valid_indices[i] - valid_indices[i - 1]) *
        original_angle_step;

    // Law of cosines:  d^2 = r1^2 + r2^2 − 2*r1*r2*cos(delta theta)
    const float euclidean_dist =
        std::sqrt(r1 * r1 + r2 * r2 - 2.0f * r1 * r2 * std::cos(angle_diff));

    if (euclidean_dist > distance_threshold) {
      // Close off current subrange, start a new one
      current.end_idx = valid_indices[i - 1];
      subranges.push_back(std::move(current));

      current = Subrange{};
      current.start_idx = valid_indices[i];
      current.indices = {valid_indices[i]};
      current.values = {valid_values[i]};
    } else {
      current.indices.push_back(valid_indices[i]);
      current.values.push_back(valid_values[i]);
    }
  }

  // Finalize the last subrange
  current.end_idx = valid_indices.back();
  subranges.push_back(std::move(current));

  // Interpolate each subrange independently
  std::vector<float> result(target_length,
                            std::numeric_limits<float>::infinity());

  for (const auto &sr : subranges) {
    const int original_len = static_cast<int>(sr.values.size());

    // Map original index range → target index range
    int start_pos = static_cast<int>(std::round(sr.start_idx * interp_factor));
    int end_pos = static_cast<int>(std::round(sr.end_idx * interp_factor));

    start_pos = std::clamp(start_pos, 0, target_length - 1);
    end_pos = std::clamp(end_pos, 0, target_length - 1);

    if (start_pos >= end_pos) {
      // Degenerate / single-point subrange
      if (start_pos >= 0 && start_pos < target_length && !sr.values.empty()) {
        result[start_pos] = sr.values.front();
      }
      continue;
    }

    const int new_len = end_pos - start_pos + 1;

    if (original_len < 2) {
      // Single source point -> fill entire target span with that value
      std::fill(result.begin() + start_pos, result.begin() + end_pos + 1,
                sr.values.front());
      continue;
    }

    // Linear interpolation (equivalent to scipy interp1d "linear")
    // Map both original and new indices onto [0, 1] and lerp.
    for (int j = 0; j < new_len; ++j) {
      // Position in [0, 1] within this subrange
      const float t = static_cast<float>(j) / static_cast<float>(new_len - 1);

      // Which segment of the original data does t fall into?
      const float src_pos = t * static_cast<float>(original_len - 1);
      const int lo = static_cast<int>(std::floor(src_pos));
      const int hi = std::min(lo + 1, original_len - 1);
      const float frac = src_pos - static_cast<float>(lo);

      result[start_pos + j] =
          sr.values[lo] * (1.0f - frac) + sr.values[hi] * frac;
    }
  }

  return result;
}

} // namespace pointcloud_utils

// Register as a composable node so it can be loaded into a component container
RCLCPP_COMPONENTS_REGISTER_NODE(pointcloud_utils::LaserScanInterpolator)
