#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/empty.hpp>
#include <std_msgs/msg/int32.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <memory>
#include <vector>
#include <fstream>
#include <std_msgs/msg/bool.hpp>
#include <tf2/time.h>
#include "rby_wrapper/msg/robot_state.hpp"
#include <set>

// map_or_odom
// #define FRAME_ID "map"  
#define FRAME_ID "odom"  

#define IF_RECORD_ROBOT2 false

using namespace std::chrono_literals;
namespace fs = std::filesystem;

class ClientHelperStitch : public rclcpp::Node
{
public:
    ClientHelperStitch() : Node("client_helper_stitch")
    {
        // Initialize TF buffer and listener
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        // Create publishers and subscribers
        pointcloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/stitched_pointcloud", 1);
        
        // Add publisher for next_id_
        id_pub_ = this->create_publisher<std_msgs::msg::Int32>(
            "/exp/id", 1);
        
        collect_sub_ = this->create_subscription<std_msgs::msg::Empty>(
            "/manager/collect_pointcloud",
            1,
            std::bind(&ClientHelperStitch::collect_callback, this, std::placeholders::_1));
        
        pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/head_cam/point_cloud",
            1,
            std::bind(&ClientHelperStitch::pointcloud_callback, this, std::placeholders::_1));

        save_sub_ = this->create_subscription<std_msgs::msg::Bool>(
            "/manager/save_data",
            1,
            std::bind(&ClientHelperStitch::save_callback, this, std::placeholders::_1));

        clean_buffer_sub_ = this->create_subscription<std_msgs::msg::Empty>(
            "/manager/clean_stitched_buffer",
            1,
            std::bind(&ClientHelperStitch::clean_buffer_callback, this, std::placeholders::_1));

        robot_state_sub_ = this->create_subscription<rby_wrapper::msg::RobotState>(
            "/robot/state",
            10,  // Larger queue size to ensure we don't miss updates
            std::bind(&ClientHelperStitch::robot_state_callback, this, std::placeholders::_1));

        // Add subscriber for policy call
        policy_call_sub_ = this->create_subscription<std_msgs::msg::Bool>(
            "/manager/call_policy",
            1,
            std::bind(&ClientHelperStitch::policy_call_callback, this, std::placeholders::_1));

        // Create timer for publishing stitched point cloud
        publish_timer_ = this->create_wall_timer(
            100ms,
            std::bind(&ClientHelperStitch::publish_stitched_pointcloud, this));
            
        // Create timer for publishing next_id_
        id_timer_ = this->create_wall_timer(
            500ms,
            std::bind(&ClientHelperStitch::publish_id, this));
        
        // Initialize save paths
        save_dir_ = "dataset_rollout/lamp/pcl";
        if (IF_RECORD_ROBOT2){
            meta_file_ = "n2m_inference/tbd/meta_data.yaml";
            save_dir_ = "n2m_inference/tbd/pcl";
        }
        else{
            meta_file_ = "dataset_rollout/tbd/meta_data.yaml";
            save_dir_ = "dataset_rollout/tbd/pcl";
        }
        
        // Create directories if they don't exist
        fs::create_directories(save_dir_);
        fs::create_directories(fs::path(meta_file_).parent_path());

        // Initialize next_id_
        next_id_ = get_next_id();

        RCLCPP_INFO(this->get_logger(), "Client Helper Stitch node initialized");
    }

private:
    void policy_call_callback(const std_msgs::msg::Bool::SharedPtr /* msg */)
    {
        try
        {
            // Record transform from FRAME_ID to base
            policy_base_transform_ = tf_buffer_->lookupTransform(
                FRAME_ID,      // target frame
                "base",        // source frame
                rclcpp::Time(0));  // get the latest available transform
                
            // Record transform from FRAME_ID to head_cam
            policy_head_cam_transform_ = tf_buffer_->lookupTransform(
                FRAME_ID,      // target frame
                "head_cam",    // source frame
                rclcpp::Time(0));  // get the latest available transform
            
            if (IF_RECORD_ROBOT2)
                robot2_base_transform_ = tf_buffer_->lookupTransform(
                    FRAME_ID,      // target frame
                    "robot2_base",    // source frame
                    rclcpp::Time(0));  // get the latest available transform

            // Store the SE3 matrices
            policy_base_matrix_ = transform_to_matrix(policy_base_transform_);
            policy_head_cam_matrix_ = transform_to_matrix(policy_head_cam_transform_);
            robot2_base_matrix_ = transform_to_matrix(robot2_base_transform_);

            has_policy_transforms_ = true;
            
            RCLCPP_INFO(this->get_logger(), "Recorded transforms at policy call time");
        }
        catch (const std::exception& e)
        {
            RCLCPP_ERROR(this->get_logger(), "Error recording transforms at policy call: %s", e.what());
            has_policy_transforms_ = false;
        }
    }

    void collect_callback(const std_msgs::msg::Empty::SharedPtr /* msg */)
    {
        should_collect_ = true;
        RCLCPP_INFO(this->get_logger(), "Collection triggered");
    }

    void robot_state_callback(const rby_wrapper::msg::RobotState::SharedPtr msg)
    {
        // Update our buffer with the latest robot state
        latest_robot_state_ = *msg;
    }

    void save_callback(const std_msgs::msg::Bool::SharedPtr msg)
    {   
        if (pointclouds_.empty())
        {
            RCLCPP_WARN(this->get_logger(), "No point clouds to save");
            save_metadata(msg->data);
            next_id_++;
            return;
        }

        try
        {
            // Start timing
            auto start_time = std::chrono::high_resolution_clock::now();
            
            // Combined point cloud
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr combined_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
            
            RCLCPP_INFO(this->get_logger(), "Processing %zu point clouds with direct transform", pointclouds_.size());
            
            // Process all point clouds with direct transform only
            for (size_t i = 0; i < pointclouds_.size(); ++i)
            {
                // Convert to PCL format
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
                pcl::PCLPointCloud2 pcl_pc2;
                pcl_conversions::toPCL(pointclouds_[i], pcl_pc2);
                pcl::fromPCLPointCloud2(pcl_pc2, *cloud);
                
                // Skip empty clouds
                if (cloud->empty()) {
                    RCLCPP_WARN(this->get_logger(), "Point cloud %zu is empty", i);
                    continue;
                }
                
                // Remove NaN points
                std::vector<int> indices;
                pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);
                if (cloud->empty()) {
                    RCLCPP_WARN(this->get_logger(), "Point cloud %zu is empty after NaN removal", i);
                    continue;
                }
                
                RCLCPP_INFO(this->get_logger(), "Point cloud %zu has %zu valid points", 
                           i, cloud->size());
                
                // Apply transform from TF directly
                Eigen::Matrix4f transform_matrix = transform_to_matrix(transforms_[i]);
                pcl::transformPointCloud(*cloud, *cloud, transform_matrix);
                
                // Check again for NaNs
                indices.clear();
                pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);
                if (cloud->empty()) {
                    RCLCPP_WARN(this->get_logger(), "Point cloud %zu is empty after transform", i);
                    continue;
                }
                
                // Add to combined cloud
                *combined_cloud += *cloud;
                RCLCPP_INFO(this->get_logger(), "Added cloud %zu to combined cloud", i);
            }
            
            if (combined_cloud->empty()) {
                RCLCPP_ERROR(this->get_logger(), "No valid point clouds to save");
                return;
            }

            RCLCPP_INFO(this->get_logger(), "Combined cloud has %zu points", combined_cloud->size());

            // Set coordinate frame
            combined_cloud->header.frame_id = FRAME_ID;

            // Save point cloud as ASCII PCD with color info preserved
            std::string filename = save_dir_ + "/" + std::to_string(next_id_) + ".pcd";
            RCLCPP_INFO(this->get_logger(), "Saving PCD to %s with %zu points", 
                        filename.c_str(), combined_cloud->size());
                        
            if (pcl::io::savePCDFileASCII(filename, *combined_cloud) == -1)
            {
                RCLCPP_ERROR(this->get_logger(), "Failed to save point cloud to %s", filename.c_str());
                return;
            }
            
            // Calculate total processing time
            auto total_end_time = std::chrono::high_resolution_clock::now();
            auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end_time - start_time);
            RCLCPP_INFO(this->get_logger(), "Total processing completed in %ld ms", total_duration.count());
            
            RCLCPP_INFO(this->get_logger(), "Saved point cloud to %s", filename.c_str());

            // Save metadata
            save_metadata(msg->data);

            // Increment ID
            next_id_++;
        }
        catch (const std::exception& e)
        {
            RCLCPP_ERROR(this->get_logger(), "Error saving point cloud: %s", e.what());
        }
    }

    size_t get_next_id()
    {
        // Create a set of existing IDs from the meta_data.yaml file
        std::set<size_t> existing_ids;
        
        // Check if meta_data.yaml exists
        if (fs::exists(meta_file_))
        {
            try
            {
                // Load the YAML file
                YAML::Node node = YAML::LoadFile(meta_file_);
                
                // Extract all keys as IDs
                for (const auto& entry : node)
                {
                    try
                    {
                        size_t id = std::stoul(entry.first.as<std::string>());
                        existing_ids.insert(id);
                    }
                    catch (const std::exception& e)
                    {
                        RCLCPP_WARN(this->get_logger(), "Invalid ID format in metadata: %s", entry.first.as<std::string>().c_str());
                    }
                }
            }
            catch (const std::exception& e)
            {
                RCLCPP_WARN(this->get_logger(), "Error loading metadata file: %s", e.what());
                // If we can't read the file, return 0 as the starting ID
                return 0;
            }
        }
        
        // Find the smallest positive integer that's not in the set
        size_t next_id = 1;
        while (existing_ids.find(next_id) != existing_ids.end())
        {
            next_id++;
        }
        
        RCLCPP_INFO(this->get_logger(), "Next ID determined from metadata: %zu", next_id);
        return next_id;
    }

    void save_metadata(bool success)
    {
        YAML::Node node;
        
        // Load existing metadata if it exists
        if (fs::exists(meta_file_))
        {
            try
            {
                node = YAML::LoadFile(meta_file_);
            }
            catch (const std::exception& e)
            {
                RCLCPP_WARN(this->get_logger(), "Error loading metadata file: %s", e.what());
            }
        }

        // Add new entry
        YAML::Node new_entry;
        
        // Use the stored transform from policy call time instead of looking up at save time
        if (has_policy_transforms_)
        {
            // Extract x, y from stored transform
            new_entry["x"] = policy_base_transform_.transform.translation.x;
            new_entry["y"] = policy_base_transform_.transform.translation.y;
            
            // Calculate theta (yaw) from quaternion
            double qx = policy_base_transform_.transform.rotation.x;
            double qy = policy_base_transform_.transform.rotation.y;
            double qz = policy_base_transform_.transform.rotation.z;
            double qw = policy_base_transform_.transform.rotation.w;
            
            // Convert quaternion to Euler angles (specifically yaw)
            double theta = std::atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz));
            new_entry["theta"] = theta;
            
            // Store full SE3 transform matrices using row format
            YAML::Node base_transform_node;
            for (int i = 0; i < 4; i++) {
                std::vector<float> row;
                for (int j = 0; j < 4; j++) {
                    row.push_back(policy_base_matrix_(i, j));
                }
                base_transform_node["row" + std::to_string(i)] = row;
            }
            new_entry["base_transform"] = base_transform_node;
            
            YAML::Node head_cam_transform_node;
            for (int i = 0; i < 4; i++) {
                std::vector<float> row;
                for (int j = 0; j < 4; j++) {
                    row.push_back(policy_head_cam_matrix_(i, j));
                }
                head_cam_transform_node["row" + std::to_string(i)] = row;
            }
            new_entry["head_cam_transform"] = head_cam_transform_node;

            YAML::Node robot2_base_transform_node;
            if (IF_RECORD_ROBOT2)
            {
                for (int i = 0; i < 4; i++) {
                    std::vector<float> row;
                    for (int j = 0; j < 4; j++) {
                        row.push_back(robot2_base_matrix_(i, j));
                    }
                    robot2_base_transform_node["row" + std::to_string(i)] = row;
                }
                new_entry["robot2_base_transform"] = robot2_base_transform_node;
            }
            
            RCLCPP_INFO(this->get_logger(), "Using stored policy call transform: x=%.2f, y=%.2f, theta=%.2f",
                      policy_base_transform_.transform.translation.x, 
                      policy_base_transform_.transform.translation.y, 
                      theta);
        }
        else
        {
            // Fallback to current transform if policy transform wasn't recorded
            RCLCPP_WARN(this->get_logger(), "No policy transforms available at save time! Using current transforms instead.");
            
            try
            {
                // Look up the transforms
                geometry_msgs::msg::TransformStamped base_transform = tf_buffer_->lookupTransform(
                    FRAME_ID,    // target frame
                    "base",      // source frame
                    rclcpp::Time(0));  // get the latest available transform
                
                geometry_msgs::msg::TransformStamped head_cam_transform = tf_buffer_->lookupTransform(
                    FRAME_ID,    // target frame
                    "head_cam",  // source frame
                    rclcpp::Time(0));  // get the latest available transform
                
                geometry_msgs::msg::TransformStamped robot2_base_transform = tf_buffer_->lookupTransform(
                    FRAME_ID,      // target frame
                    "robot2_base", // source frame
                    rclcpp::Time(0));  // get the latest available transform
                
                // Convert to matrices
                Eigen::Matrix4f base_matrix = transform_to_matrix(base_transform);
                Eigen::Matrix4f head_cam_matrix = transform_to_matrix(head_cam_transform);
                Eigen::Matrix4f robot2_base_matrix = transform_to_matrix(robot2_base_transform);
                    
                // Extract x, y from translation
                new_entry["x"] = base_transform.transform.translation.x;
                new_entry["y"] = base_transform.transform.translation.y;
                
                // Calculate theta (yaw) from quaternion
                double qx = base_transform.transform.rotation.x;
                double qy = base_transform.transform.rotation.y;
                double qz = base_transform.transform.rotation.z;
                double qw = base_transform.transform.rotation.w;
                
                // Convert quaternion to Euler angles (specifically yaw)
                double theta = std::atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz));
                new_entry["theta"] = theta;
                
                // Store full SE3 transform matrices using row format
                YAML::Node base_transform_node;
                for (int i = 0; i < 4; i++) {
                    std::vector<float> row;
                    for (int j = 0; j < 4; j++) {
                        row.push_back(base_matrix(i, j));
                    }
                    base_transform_node["row" + std::to_string(i)] = row;
                }
                new_entry["base_transform"] = base_transform_node;
                
                YAML::Node head_cam_transform_node;
                for (int i = 0; i < 4; i++) {
                    std::vector<float> row;
                    for (int j = 0; j < 4; j++) {
                        row.push_back(head_cam_matrix(i, j));
                    }
                    head_cam_transform_node["row" + std::to_string(i)] = row;
                }
                new_entry["head_cam_transform"] = head_cam_transform_node;
                
                YAML::Node robot2_base_transform_node;
                for (int i = 0; i < 4; i++) {
                    std::vector<float> row;
                    for (int j = 0; j < 4; j++) {
                        row.push_back(robot2_base_matrix(i, j));
                    }
                    robot2_base_transform_node["row" + std::to_string(i)] = row;
                }
                new_entry["robot2_base_transform"] = robot2_base_transform_node;
                
                RCLCPP_WARN(this->get_logger(), "No stored policy call transform available. Using current transform: x=%.2f, y=%.2f, theta=%.2f",
                          base_transform.transform.translation.x, base_transform.transform.translation.y, theta);
            }
            catch (const std::exception& e)
            {
                // Fallback to dummy data if transform lookup fails
                RCLCPP_WARN(this->get_logger(), "Could not FRAME_ID->base transform: %s. Using dummy values", e.what());
                new_entry["x"] = 0.0;
                new_entry["y"] = 0.0;
                new_entry["theta"] = 0.0;
            }
        }
        
        // Add torso information if available
        if (!latest_robot_state_.torso_qpos.empty() && latest_robot_state_.torso_qpos.size() > 1) {
            new_entry["torso_1"] = latest_robot_state_.torso_qpos[1];
        }
        
        new_entry["success"] = success;
        node[std::to_string(next_id_)] = new_entry;

        // Save to file
        std::ofstream fout(meta_file_);
        fout << node;
        RCLCPP_INFO(this->get_logger(), "\033[1;32mUpdated metadata file\033[0m");
    }

    void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        if (!should_collect_)
            return;

        try
        {
            // Get the latest transform from head_camera to map without considering timestamp
            geometry_msgs::msg::TransformStamped transform = tf_buffer_->lookupTransform(
                FRAME_ID,                // target frame
                "head_cam",           // source frame
                rclcpp::Time(0)); // Use time 0 to get the latest available transform

            // Store point cloud and transform
            pointclouds_.push_back(*msg);
            transforms_.push_back(transform);

            RCLCPP_INFO(this->get_logger(), "Collected point cloud %zu", pointclouds_.size());
            should_collect_ = false;
        }
        catch (const std::exception& e)
        {
            RCLCPP_ERROR(this->get_logger(), "Error collecting point cloud: %s", e.what());
        }
    }

    void publish_stitched_pointcloud()
    {
        if (pointclouds_.empty()) {
            // No point clouds to publish
            return;
        }

        try
        {
            // Simple combined cloud - no alignment, just transform
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr combined_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
            
            for (size_t i = 0; i < pointclouds_.size(); ++i)
            {
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
                pcl::PCLPointCloud2 pcl_pc2;
                pcl_conversions::toPCL(pointclouds_[i], pcl_pc2);
                pcl::fromPCLPointCloud2(pcl_pc2, *cloud);

                // Remove NaN points
                std::vector<int> indices;
                pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);
                if (cloud->empty()) {
                    continue;
                }

                // Apply transform directly from TF
                Eigen::Matrix4f transform_matrix = transform_to_matrix(transforms_[i]);
                pcl::transformPointCloud(*cloud, *cloud, transform_matrix);
                
                // Check again for NaNs
                indices.clear();
                pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);
                if (cloud->empty()) {
                    continue;
                }

                // Simply add the cloud to the combined cloud
                *combined_cloud += *cloud;
            }

            // Convert back to PointCloud2 and publish
            sensor_msgs::msg::PointCloud2 stitched_msg;
            pcl::toROSMsg(*combined_cloud, stitched_msg);
            stitched_msg.header.frame_id = FRAME_ID;
            stitched_msg.header.stamp = this->now();
            pointcloud_pub_->publish(stitched_msg);
        }
        catch (const std::exception& e)
        {
            RCLCPP_ERROR(this->get_logger(), "Error publishing stitched point cloud: %s", e.what());
        }
    }

    void clean_buffer_callback(const std_msgs::msg::Empty::SharedPtr /* msg */)
    {
        // Remove the last pointcloud and transform if not empty
        size_t num_clouds = pointclouds_.size();
        if (!pointclouds_.empty() && !transforms_.empty()) {
            pointclouds_.pop_back();
            transforms_.pop_back();
            RCLCPP_INFO(this->get_logger(), 
                       "Clean buffer received - removed last pointcloud from queue. Remaining: %zu", 
                       pointclouds_.size());
        } else {
            RCLCPP_INFO(this->get_logger(), 
                       "Clean buffer received - but queue already empty.");
        }
        should_collect_ = false;
        // publish an empty point cloud to refresh visualization
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr empty_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        sensor_msgs::msg::PointCloud2 empty_msg;
        pcl::toROSMsg(*empty_cloud, empty_msg);
        empty_msg.header.frame_id = FRAME_ID;
        empty_msg.header.stamp = this->now();
        pointcloud_pub_->publish(empty_msg);
    }

    void publish_id()
    {
        auto msg = std::make_unique<std_msgs::msg::Int32>();
        msg->data = static_cast<int32_t>(next_id_);
        id_pub_->publish(*msg);
    }

    Eigen::Matrix4f transform_to_matrix(const geometry_msgs::msg::TransformStamped& transform)
    {
        Eigen::Matrix4f matrix = Eigen::Matrix4f::Identity();
        
        // Set translation
        matrix(0, 3) = transform.transform.translation.x;
        matrix(1, 3) = transform.transform.translation.y;
        matrix(2, 3) = transform.transform.translation.z;

        // Set rotation
        double qx = transform.transform.rotation.x;
        double qy = transform.transform.rotation.y;
        double qz = transform.transform.rotation.z;
        double qw = transform.transform.rotation.w;

        // Normalize quaternion to ensure orthogonal rotation matrix
        double norm = std::sqrt(qx*qx + qy*qy + qz*qz + qw*qw);
        if (norm > 0.0) {
            qx /= norm;
            qy /= norm;
            qz /= norm;
            qw /= norm;
        }

        matrix(0, 0) = 1 - 2*qy*qy - 2*qz*qz;
        matrix(0, 1) = 2*qx*qy - 2*qw*qz;
        matrix(0, 2) = 2*qx*qz + 2*qw*qy;

        matrix(1, 0) = 2*qx*qy + 2*qw*qz;
        matrix(1, 1) = 1 - 2*qx*qx - 2*qz*qz;
        matrix(1, 2) = 2*qy*qz - 2*qw*qx;

        matrix(2, 0) = 2*qx*qz - 2*qw*qy;
        matrix(2, 1) = 2*qy*qz + 2*qw*qx;
        matrix(2, 2) = 1 - 2*qx*qx - 2*qy*qy;

        return matrix;
    }

    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_pub_;
    rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr id_pub_;
    rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr collect_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr save_sub_;
    rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr clean_buffer_sub_;
    rclcpp::Subscription<rby_wrapper::msg::RobotState>::SharedPtr robot_state_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr policy_call_sub_;
    rclcpp::TimerBase::SharedPtr publish_timer_;
    rclcpp::TimerBase::SharedPtr id_timer_;

    std::vector<sensor_msgs::msg::PointCloud2> pointclouds_;
    std::vector<geometry_msgs::msg::TransformStamped> transforms_;
    bool should_collect_ = false;
    
    // Store the latest robot state
    rby_wrapper::msg::RobotState latest_robot_state_;

    // Transforms captured at policy call time
    geometry_msgs::msg::TransformStamped policy_base_transform_;
    geometry_msgs::msg::TransformStamped robot2_base_transform_;
    geometry_msgs::msg::TransformStamped policy_head_cam_transform_;
    Eigen::Matrix4f policy_base_matrix_ = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f policy_head_cam_matrix_ = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f robot2_base_matrix_ = Eigen::Matrix4f::Identity();
    bool has_policy_transforms_ = false;

    std::string save_dir_;
    std::string meta_file_;
    size_t next_id_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ClientHelperStitch>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
} 