#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <memory>
#include <cmath>

// map_or_odom
// #define FRAME_ID "map"  
#define FRAME_ID "odom"  

#define BASE_FRAME "base"

class SE2Publisher : public rclcpp::Node
{
public:
    SE2Publisher() : Node("se2_publisher")
    {
        // Initialize TF buffer and listener
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        // Create publisher for SE2 information
        se2_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>(
            "/robot/se2", 10);

        // Create timer for publishing SE2 data
        publish_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(40),  // 25Hz
            std::bind(&SE2Publisher::publish_se2, this));

        // Initialize time tracking variables
        last_log_time_ = this->now();
        last_error_time_ = this->now();

        RCLCPP_INFO(this->get_logger(), "SE2 Publisher node initialized");
    }

private:
    void publish_se2()
    {
        try
        {
            // Look up the transform from map to base
            geometry_msgs::msg::TransformStamped transform = tf_buffer_->lookupTransform(
                FRAME_ID,      // target frame
                BASE_FRAME,    // source frame
                rclcpp::Time(0));  // get the latest available transform

            // Extract x, y from the transform
            float x = transform.transform.translation.x;
            float y = transform.transform.translation.y;
            
            // Calculate theta (yaw) from quaternion
            float qx = transform.transform.rotation.x;
            float qy = transform.transform.rotation.y;
            float qz = transform.transform.rotation.z;
            float qw = transform.transform.rotation.w;
            
            // Convert quaternion to Euler angles (specifically yaw)
            float theta = std::atan2(2.0f * (qw * qz + qx * qy), 1.0f - 2.0f * (qy * qy + qz * qz));

            // Create float array message
            auto msg = std::make_unique<std_msgs::msg::Float32MultiArray>();
            
            // Configure dimensions
            msg->layout.dim.push_back(std_msgs::msg::MultiArrayDimension());
            msg->layout.dim[0].size = 3;
            msg->layout.dim[0].stride = 1;
            msg->layout.dim[0].label = "se2";
            
            // Set data: [x, y, theta]
            msg->data = {x, y, theta};
            
            // Publish the message
            se2_pub_->publish(*msg);
            
            // Log periodically (every ~5 seconds) to avoid flooding
            auto current_time = this->now();
            double time_since_last_log = (current_time.seconds() - last_log_time_.seconds());
            // if (time_since_last_log >= 0.0) {
            //     RCLCPP_INFO(this->get_logger(), "Publishing SE2: x=%.2f, y=%.2f, theta=%.2f", x, y, theta);
            //     last_log_time_ = current_time;
            // }
        }
        catch (const std::exception& e)
        {
            // Log error periodically to avoid flooding
            auto current_time = this->now();
            double time_since_last_error = (current_time.seconds() - last_error_time_.seconds());
            if (time_since_last_error >= 5.0) {
                RCLCPP_ERROR(this->get_logger(), "Error looking up transform: %s", e.what());
                last_error_time_ = current_time;
            }
        }
    }

    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr se2_pub_;
    rclcpp::TimerBase::SharedPtr publish_timer_;
    rclcpp::Time last_log_time_;
    rclcpp::Time last_error_time_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SE2Publisher>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}