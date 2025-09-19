#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <Eigen/Dense>
#include <chrono>
#include <vector>
#include <limits>
#include <cmath>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/utils.h>

// Macro for frame ID
#define FRAME_ID "base"

using namespace std::chrono_literals;

class CameraIntrinsics {
public:
    float fx, fy, cx, cy;
    int width, height;

    CameraIntrinsics(float fx, float fy, float cx, float cy, int width, int height)
        : fx(fx), fy(fy), cx(cx), cy(cy), width(width), height(height) {}
};

class CamRenderHelper : public rclcpp::Node
{
public:
    CamRenderHelper() : Node("cam_render_helper")
    {
        // Initialize TF buffer and listener - changed to match client_helper_stitch.cpp
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
        
        // Create publisher for rendered point cloud
        rendered_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/robot/pcl_rendered", 1);

        // Subscribe to point cloud
        pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/head_cam/point_cloud",
            1,
            std::bind(&CamRenderHelper::pointcloud_callback, this, std::placeholders::_1));

        // Initialize camera intrinsics with default values
        // These match the values used in fpv_render_for_Inference.cpp
        intrinsics_ = std::make_shared<CameraIntrinsics>(500, 500, 320, 260, 640, 520);

        RCLCPP_INFO(this->get_logger(), "CamRenderHelper node initialized");
    }

private:
    void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        try
        {
            // Convert to PCL format
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
            pcl::PCLPointCloud2 pcl_pc2;
            pcl_conversions::toPCL(*msg, pcl_pc2);
            pcl::fromPCLPointCloud2(pcl_pc2, *cloud);
            
            // Process the point cloud directly
            process_point_cloud(cloud);
        }
        catch (const std::exception& e)
        {
            RCLCPP_ERROR(this->get_logger(), "Error processing point cloud: %s", e.what());
        }
    }

    void process_point_cloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud)
    {
        try
        {
            // Fixed values for the camera pose
            // You can set these to appropriate constant values since we're not using TF anymore
            
            // Create camera pose - using identity transform since we're rendering from the camera's perspective
            Eigen::Matrix4f cameraPose = Eigen::Matrix4f::Identity();
            
            // Render the point cloud
            auto renderedCloud = renderPointCloudWithOcclusion<pcl::PointXYZRGB>(
                cloud,
                cameraPose,
                *intrinsics_,
                -0.1f,  // minDepth
                10.0f   // maxDepth
            );

            // listen to base to frame tf transform
            geometry_msgs::msg::TransformStamped base_transform_;
            geometry_msgs::msg::TransformStamped head_cam_transform_;
            try {
                base_transform_ = tf_buffer_->lookupTransform(
                    FRAME_ID,      // target frame
                    "base",        // source frame
                    rclcpp::Time(0));  // get the latest available transform
                
                // Record transform from FRAME_ID to head_cam
                head_cam_transform_ = tf_buffer_->lookupTransform(
                    FRAME_ID,      // target frame
                    "head_cam",    // source frame
                    rclcpp::Time(0));  // get the latest available transform
            }
            catch (const tf2::TransformException& ex) {
                RCLCPP_ERROR(this->get_logger(), "Transform error: %s", ex.what());
                return;
            }
            // translation from headcam transform
            Eigen::Vector3f translation(
                head_cam_transform_.transform.translation.x,
                head_cam_transform_.transform.translation.y,
                head_cam_transform_.transform.translation.z
            );
            // yaw(theta) from base transform
            float qx = base_transform_.transform.rotation.x;
            float qy = base_transform_.transform.rotation.y;
            float qz = base_transform_.transform.rotation.z;
            float qw = base_transform_.transform.rotation.w;
            float yaw = std::atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz));
            // pitch and roll are fixed
            float pitch = 0.50;
            float roll = 0.0;
            Eigen::Matrix3f rotation;
            rotation = Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()) *
                    Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY()) *
                    Eigen::AngleAxisf(roll, Eigen::Vector3f::UnitX());
            // Update camera pose
            cameraPose.block<3,3>(0,0) = rotation;
            cameraPose.block<3,1>(0,3) = translation;

            // transform rendered cloud to FRAME_ID
            transformPoints(*renderedCloud, *renderedCloud, cameraPose);
            // Convert back to ROS message and publish
            sensor_msgs::msg::PointCloud2 rendered_msg;
            pcl::toROSMsg(*renderedCloud, rendered_msg);
            // RCLCPP_INFO(this->get_logger(), "before transform frame_id: %s", 
            //           rendered_msg.header.frame_id.c_str());
            rendered_msg.header.frame_id = FRAME_ID;
            rendered_msg.header.stamp = this->now();
            rendered_cloud_pub_->publish(rendered_msg);
            // RCLCPP_INFO(this->get_logger(), "after transform frame_id: %s", 
            //           rendered_msg.header.frame_id.c_str());
            
            RCLCPP_DEBUG(this->get_logger(), "Published rendered point cloud with %zu points", 
                      renderedCloud->size());
        }
        catch (const std::exception& e)
        {
            RCLCPP_ERROR(this->get_logger(), "Error in point cloud processing: %s", e.what());
        }
    }

    template<typename PointT>
    void transformPoints(const pcl::PointCloud<PointT>& source, 
                        pcl::PointCloud<PointT>& target,
                        const Eigen::Matrix4f& transform) {
        pcl::transformPointCloud(source, target, transform);
    }

    template<typename PointT>
    typename pcl::PointCloud<PointT>::Ptr renderPointCloudWithOcclusion(
        const typename pcl::PointCloud<PointT>::Ptr& pointCloud,
        const Eigen::Matrix4f& cameraPose,
        const CameraIntrinsics& intrinsics,
        float minDepth = 0.1,
        float maxDepth = 10.0
    ) {
        // Transform points to camera frame
        Eigen::Matrix4f cameraToWorld = cameraPose.inverse();
        typename pcl::PointCloud<PointT>::Ptr cloudCamera(new pcl::PointCloud<PointT>());
        transformPoints(*pointCloud, *cloudCamera, cameraToWorld);
        
        // Initialize depth buffer
        std::vector<std::vector<float>> depthBuffer(intrinsics.width, 
                                                std::vector<float>(intrinsics.height, 
                                                std::numeric_limits<float>::infinity()));
        std::vector<std::vector<int>> pointIndices(intrinsics.width, 
                                                std::vector<int>(intrinsics.height, -1));
        
        // Process each point
        for (size_t i = 0; i < cloudCamera->points.size(); ++i) {
            const auto& point = cloudCamera->points[i];
            
            // In our camera model, x-axis is forward, y-axis is right, z-axis is down
            // So we use x as the depth, project y and z onto the image plane
            float depth = point.x;  // x is the depth in the camera frame
            float y = point.y;      // y is right
            float z = point.z;      // z is down
            
            // Skip if behind camera or invalid
            if (depth <= 0 || !std::isfinite(depth) || !std::isfinite(y) || !std::isfinite(z)) {
                continue;
            }
            
            // Project point to image plane
            // Since x is depth, we use y/depth and z/depth for projection
            float u = (-y * intrinsics.fx / depth) + intrinsics.cx;  // Negative y because image x increases to the right
            float v = (-z * intrinsics.fy / depth) + intrinsics.cy;  // Negative z because image y increases downward
            
            // Skip if outside image bounds
            if (u < 0 || u >= intrinsics.width || v < 0 || v >= intrinsics.height) {
                continue;
            }
            
            // Get pixel coordinates
            int pixelX = static_cast<int>(u);
            int pixelY = static_cast<int>(v);
            
            // Skip if outside depth range
            if (depth < minDepth || depth > maxDepth) {
                continue;
            }
            
            // Update depth buffer if point is closer
            if (depth < depthBuffer[pixelX][pixelY]) {
                depthBuffer[pixelX][pixelY] = depth;
                pointIndices[pixelX][pixelY] = i;
            }
        }
        
        // Create output point cloud
        std::vector<int> validIndices;
        for (const auto& row : pointIndices) {
            for (int idx : row) {
                if (idx >= 0) {
                    validIndices.push_back(idx);
                }
            }
        }
        
        if (validIndices.empty()) {
            return typename pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>());
        }
        
        typename pcl::PointCloud<PointT>::Ptr renderedCloud(new pcl::PointCloud<PointT>());
        renderedCloud->reserve(validIndices.size());
        
        // Extract visible points
        for (int idx : validIndices) {
            renderedCloud->points.push_back(pointCloud->points[idx]);
        }
        
        renderedCloud->width = renderedCloud->points.size();
        renderedCloud->height = 1;
        renderedCloud->is_dense = false;
        
        return renderedCloud;
    }

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr rendered_cloud_pub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    
    std::shared_ptr<CameraIntrinsics> intrinsics_;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr latest_cloud_;
    
    // TF2 objects - changed to match client_helper_stitch.cpp
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CamRenderHelper>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}