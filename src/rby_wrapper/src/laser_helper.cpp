#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <chrono>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2/transform_datatypes.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"

using namespace std::chrono_literals;

class LaserHelper : public rclcpp::Node
{
public:
  LaserHelper() : Node("laser_helper")
  {
    // Create publisher for combined scan
    scan_publisher_ = this->create_publisher<sensor_msgs::msg::LaserScan>("/scan", 10);  // Increased queue size

    // Configure QoS profile with a larger history and reliability
    auto qos = rclcpp::QoS(rclcpp::KeepLast(10))
               .reliability(rclcpp::ReliabilityPolicy::BestEffort)
               .durability(rclcpp::DurabilityPolicy::Volatile);

    // Create subscribers for left and right lidar scans with better QoS settings
    left_scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
      "/scan_left", qos, std::bind(&LaserHelper::leftScanCallback, this, std::placeholders::_1));
    
    right_scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
      "/scan_right", qos, std::bind(&LaserHelper::rightScanCallback, this, std::placeholders::_1));

    // Timer for publishing combined scan - 20Hz (50ms)
    timer_ = this->create_wall_timer(
      50ms, std::bind(&LaserHelper::publishCombinedScan, this));
    
    // Initialize static transforms from URDF
    // Left LiDAR: <origin xyz="0.230 0.177 0.22282" rpy="0 0 0.9491100472345164"/>
    left_lidar_transform_.translation.x = 0.230;
    left_lidar_transform_.translation.y = 0.177;
    left_lidar_transform_.translation.z = 0.22282;
    
    tf2::Quaternion q_left;
    q_left.setRPY(0.0, 0.0, 0.9491100472345164);
    left_lidar_transform_.rotation.x = q_left.x();
    left_lidar_transform_.rotation.y = q_left.y();
    left_lidar_transform_.rotation.z = q_left.z();
    left_lidar_transform_.rotation.w = q_left.w();
    
    // Right LiDAR: <origin xyz="0.228 -0.1765 0.22282" rpy="0 0 -0.9491100472345164"/>
    right_lidar_transform_.translation.x = 0.228;
    right_lidar_transform_.translation.y = -0.1765;
    right_lidar_transform_.translation.z = 0.22282;
    
    tf2::Quaternion q_right;
    q_right.setRPY(0.0, 0.0, -0.9491100472345164);
    right_lidar_transform_.rotation.x = q_right.x();
    right_lidar_transform_.rotation.y = q_right.y();
    right_lidar_transform_.rotation.z = q_right.z();
    right_lidar_transform_.rotation.w = q_right.w();
    
    // Initialize counters and timestamps for diagnostics
    left_callbacks_ = 0;
    right_callbacks_ = 0;
    process_attempts_ = 0;
    successful_publishes_ = 0;
    
    last_diagnostics_time_ = this->now();
    
    RCLCPP_INFO(this->get_logger(), "LaserHelper node initialized (publish rate: 20Hz)");
  }

private:
  void leftScanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
  {
    auto start_time = this->now();
    RCLCPP_DEBUG(this->get_logger(), "Left scan received: %f.%f", msg->header.stamp.sec, msg->header.stamp.nanosec);
    
    // Set timestamp to current time
    auto stamped_msg = std::make_shared<sensor_msgs::msg::LaserScan>(*msg);
    stamped_msg->header.stamp = this->now();
    
    {
      std::lock_guard<std::mutex> lock(scan_mutex_);
      left_scan_ = stamped_msg;
      left_received_time_ = this->now();
      left_callbacks_++;
    }
    
    auto processing_time = this->now() - start_time;
    RCLCPP_DEBUG(this->get_logger(), "Left scan processing time: %f s", 
                processing_time.seconds());
  }

  void rightScanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
  {
    auto start_time = this->now();
    RCLCPP_DEBUG(this->get_logger(), "Right scan received: %f.%f", msg->header.stamp.sec, msg->header.stamp.nanosec);
    
    // Set timestamp to current time
    auto stamped_msg = std::make_shared<sensor_msgs::msg::LaserScan>(*msg);
    stamped_msg->header.stamp = this->now();
    
    {
      std::lock_guard<std::mutex> lock(scan_mutex_);
      right_scan_ = stamped_msg;
      right_received_time_ = this->now();
      right_callbacks_++;
    }
    
    auto processing_time = this->now() - start_time;
    RCLCPP_DEBUG(this->get_logger(), "Right scan processing time: %f s", 
                processing_time.seconds());
  }

  void publishCombinedScan()
  {
    auto start_time = this->now();
    process_attempts_++;
    
    // Output diagnostics every 5 seconds
    auto current_time = this->now();
    double elapsed = (current_time - last_diagnostics_time_).seconds();
    if (elapsed > 10.0) {
      RCLCPP_INFO(this->get_logger(), 
                 "STATS: Left callbacks: %d (%.1f Hz), Right callbacks: %d (%.1f Hz), "
                 "Process attempts: %d (%.1f Hz), Successful publishes: %d (%.1f Hz)",
                 left_callbacks_, left_callbacks_ / elapsed,
                 right_callbacks_, right_callbacks_ / elapsed,
                 process_attempts_, process_attempts_ / elapsed,
                 successful_publishes_, successful_publishes_ / elapsed);
      
      // Reset counters
      left_callbacks_ = 0;
      right_callbacks_ = 0;
      process_attempts_ = 0;
      successful_publishes_ = 0;
      last_diagnostics_time_ = current_time;
    }
    
    // Skip if data mutex can't be locked immediately
    if (!scan_mutex_.try_lock()) {
      RCLCPP_DEBUG(this->get_logger(), "Timer callback skipped due to locked mutex");
      return;
    }
    
    // Use RAII to ensure mutex is unlocked when leaving scope
    std::unique_lock<std::mutex> lock(scan_mutex_, std::adopt_lock);
    
    // Skip if either scan hasn't been received yet
    if (!left_scan_ || !right_scan_) {
      RCLCPP_DEBUG(this->get_logger(), "Timer callback skipped: missing data (left: %s, right: %s)",
                  left_scan_ ? "yes" : "no", right_scan_ ? "yes" : "no");
      return;
    }
    
    // Check for stale data (older than 1 second)
    auto now = this->now();
    const double MAX_AGE = 1.0;  // 1 second maximum age
    
    bool left_stale = !left_received_time_.has_value() || 
                     (now - left_received_time_.value()).seconds() > MAX_AGE;
    bool right_stale = !right_received_time_.has_value() || 
                      (now - right_received_time_.value()).seconds() > MAX_AGE;
    
    if (left_stale || right_stale) {
      RCLCPP_WARN(this->get_logger(), 
                 "Stale scan data detected: left: %.2f s, right: %.2f s. Skipping.",
                 left_received_time_.has_value() ? (now - left_received_time_.value()).seconds() : -1.0,
                 right_received_time_.has_value() ? (now - right_received_time_.value()).seconds() : -1.0);
      return;
    }
    
    // Make copies of the scan data to release mutex early
    auto left_scan_copy = left_scan_;
    auto right_scan_copy = right_scan_;
    
    // Release the mutex early to allow callbacks to proceed
    lock.unlock();
    
    // Transform both scans to base frame using static transforms
    auto transform_start = this->now();
    auto left_transformed = transformLaserScanWithStaticTransform(left_scan_copy, left_lidar_transform_);
    auto right_transformed = transformLaserScanWithStaticTransform(right_scan_copy, right_lidar_transform_);
    auto transform_duration = (this->now() - transform_start).seconds();
    
    // Combine scans and publish
    if (left_transformed && right_transformed) {
      auto combine_start = this->now();
      auto combined_scan = combineLaserScans(*left_transformed, *right_transformed);
      auto combine_duration = (this->now() - combine_start).seconds();
      
      if (combined_scan) {
        scan_publisher_->publish(*combined_scan);
        successful_publishes_++;
        
        auto total_processing_time = (this->now() - start_time).seconds();
        RCLCPP_DEBUG(this->get_logger(), 
                    "Combined scan published: transform time: %.3f s, combine time: %.3f s, total: %.3f s",
                    transform_duration, combine_duration, total_processing_time);
      }
    } else {
      RCLCPP_WARN(this->get_logger(), "Failed to transform one or both scans");
    }
  }

  std::unique_ptr<sensor_msgs::msg::LaserScan> transformLaserScanWithStaticTransform(
    const sensor_msgs::msg::LaserScan::SharedPtr& scan,
    const geometry_msgs::msg::Transform& transform)
  {
    if (!scan) {
      return nullptr;
    }

    // Create output scan
    auto output = std::make_unique<sensor_msgs::msg::LaserScan>();
    
    // Copy header and frame properties
    output->header = scan->header;
    output->header.frame_id = "base";  // Set frame to base
    output->angle_min = scan->angle_min;
    output->angle_max = scan->angle_max;
    output->angle_increment = scan->angle_increment;
    output->time_increment = scan->time_increment;
    output->scan_time = scan->scan_time;
    output->range_min = scan->range_min;
    output->range_max = scan->range_max;
    
    // Initialize ranges with max range value
    output->ranges.resize(scan->ranges.size(), output->range_max);
    
    // Get transform data
    double tx = transform.translation.x;
    double ty = transform.translation.y;
    
    // Get quaternion from transform
    tf2::Quaternion q(
      transform.rotation.x,
      transform.rotation.y,
      transform.rotation.z,
      transform.rotation.w);
    
    // Convert quaternion to yaw (rotation around z axis)
    double roll, pitch, yaw;
    tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
    
    // Pre-calculate sin and cos of yaw for efficiency
    double cos_yaw = std::cos(yaw);
    double sin_yaw = std::sin(yaw);
    
    // Create a 360-degree range buffer for the output scan
    const int num_angles = 360;
    std::vector<float> range_buffer(num_angles, output->range_max);
    
    // Transform each range in the scan
    for (size_t i = 0; i < scan->ranges.size(); ++i) {
      float range = scan->ranges[i];
      
      // Skip invalid measurements
      if (!std::isfinite(range) || range < scan->range_min || range > scan->range_max) {
        continue;
      }
      
      // Calculate angle of this range measurement
      double angle = scan->angle_min + i * scan->angle_increment;
      
      // Convert polar to cartesian coordinates in source frame
      double x = range * std::cos(angle);
      double y = range * std::sin(angle);
      
      // Apply transform
      double x_transformed = x * cos_yaw - y * sin_yaw + tx;
      double y_transformed = x * sin_yaw + y * cos_yaw + ty;
      
      // Convert back to polar coordinates
      double range_transformed = std::sqrt(x_transformed * x_transformed + y_transformed * y_transformed);
      double angle_transformed = std::atan2(y_transformed, x_transformed);
      
      // Normalize angle to [0, 2π]
      if (angle_transformed < 0) {
        angle_transformed += 2 * M_PI;
      }
      
      // Map to discrete angle index (0-359)
      int angle_idx = static_cast<int>(angle_transformed * 180.0 / M_PI) % 360;
      
      // Update range buffer if this range is shorter
      if (range_transformed < range_buffer[angle_idx]) {
        range_buffer[angle_idx] = range_transformed;
      }
    }
    
    // Map the 360 discrete angles back to the output scan's angle resolution
    for (size_t i = 0; i < output->ranges.size(); ++i) {
      double angle = output->angle_min + i * output->angle_increment;
      if (angle < 0) {
        angle += 2 * M_PI;
      }
      int angle_idx = static_cast<int>(angle * 180.0 / M_PI) % 360;
      output->ranges[i] = range_buffer[angle_idx];
    }
    
    return output;
  }

  std::unique_ptr<sensor_msgs::msg::LaserScan> combineLaserScans(
    const sensor_msgs::msg::LaserScan& scan1,
    const sensor_msgs::msg::LaserScan& scan2)
  {
    // Create combined scan with parameters from one of the input scans
    auto combined = std::make_unique<sensor_msgs::msg::LaserScan>();
    combined->header.stamp = this->now();
    combined->header.frame_id = "base";
    
    // Use parameters from scan1 for the combined scan
    combined->angle_min = scan1.angle_min;
    combined->angle_max = scan1.angle_max;
    combined->angle_increment = scan1.angle_increment;
    combined->time_increment = scan1.time_increment;
    combined->scan_time = 0.05;  // 20Hz
    combined->range_min = std::min(scan1.range_min, scan2.range_min);
    combined->range_max = std::max(scan1.range_max, scan2.range_max);
    
    // Initialize ranges with max range value
    int num_readings = std::round((scan1.angle_max - scan1.angle_min) / scan1.angle_increment) + 1;
    combined->ranges.resize(num_readings, combined->range_max);
    
    // Create a 360-degree range buffer for merging
    const int num_angles = 360;
    std::vector<float> range_buffer(num_angles, combined->range_max);
    
    // Helper function to map from angle to buffer index
    auto angleToIndex = [&](double angle) -> int {
        // Normalize angle to [0, 2π]
        while (angle < 0) angle += 2 * M_PI;
        while (angle >= 2 * M_PI) angle -= 2 * M_PI;
        
        // Map to discrete angle index (0-359)
        return static_cast<int>(angle * 180.0 / M_PI) % 360;
    };
    
    // Merge scan1 ranges into buffer
    for (size_t i = 0; i < scan1.ranges.size(); ++i) {
        float range = scan1.ranges[i];
        if (std::isfinite(range) && range >= scan1.range_min && range <= scan1.range_max) {
            double angle = scan1.angle_min + i * scan1.angle_increment;
            int idx = angleToIndex(angle);
            if (range < range_buffer[idx]) {
                range_buffer[idx] = range;
            }
        }
    }
    
    // Merge scan2 ranges into buffer
    for (size_t i = 0; i < scan2.ranges.size(); ++i) {
        float range = scan2.ranges[i];
        if (std::isfinite(range) && range >= scan2.range_min && range <= scan2.range_max) {
            double angle = scan2.angle_min + i * scan2.angle_increment;
            int idx = angleToIndex(angle);
            if (range < range_buffer[idx]) {
                range_buffer[idx] = range;
            }
        }
    }
    
    // Map the 360 discrete angles back to the output scan's angle resolution
    for (size_t i = 0; i < combined->ranges.size(); ++i) {
        double angle = combined->angle_min + i * combined->angle_increment;
        int idx = angleToIndex(angle);
        combined->ranges[i] = range_buffer[idx];
    }
    
    return combined;
  }
  
  // Static transforms for lidars to base
  geometry_msgs::msg::Transform left_lidar_transform_;
  geometry_msgs::msg::Transform right_lidar_transform_;
  
  // Subscribers
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr left_scan_sub_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr right_scan_sub_;
  
  // Publisher
  rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr scan_publisher_;
  
  // Timer
  rclcpp::TimerBase::SharedPtr timer_;
  
  // Scan buffers
  sensor_msgs::msg::LaserScan::SharedPtr left_scan_;
  sensor_msgs::msg::LaserScan::SharedPtr right_scan_;
  
  // Mutex for thread safety
  std::mutex scan_mutex_;
  
  // Timestamps for received data
  std::optional<rclcpp::Time> left_received_time_;
  std::optional<rclcpp::Time> right_received_time_;
  
  // Statistics counters
  int left_callbacks_;
  int right_callbacks_;
  int process_attempts_;
  int successful_publishes_;
  rclcpp::Time last_diagnostics_time_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  
  // Set logging level to INFO by default
  auto node = std::make_shared<LaserHelper>();
  
  // Increase thread count for parallel processing
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
  executor.add_node(node);
  executor.spin();
  
  rclcpp::shutdown();
  return 0;
} 