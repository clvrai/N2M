#include <iostream>
#include <chrono>
#include <vector>
#include <limits>
#include <cmath>
#include <librealsense2/rs.hpp>
#include <filesystem>
#include <fstream>
#include <json.hpp>

// PCL includes
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <Eigen/Dense>

using json = nlohmann::json;
namespace fs = std::filesystem;

class CameraIntrinsics {
public:
    float fx, fy, cx, cy;
    int width, height;

    CameraIntrinsics(float fx, float fy, float cx, float cy, int width, int height)
        : fx(fx), fy(fy), cx(cx), cy(cy), width(width), height(height) {}

    static CameraIntrinsics fromRealsense(const rs2_intrinsics& intrinsics) {
        return CameraIntrinsics(
            intrinsics.fx,
            intrinsics.fy,
            intrinsics.ppx,
            intrinsics.ppy,
            intrinsics.width,
            intrinsics.height
        );
    }
};

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
    
    auto timeStart = std::chrono::high_resolution_clock::now();
    double tTotal = 0.0;
    
    // Process each point
    for (size_t i = 0; i < cloudCamera->points.size(); ++i) {
        auto pointStart = std::chrono::high_resolution_clock::now();
        
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
        
        auto pointEnd = std::chrono::high_resolution_clock::now();
        tTotal += std::chrono::duration<double>(pointEnd - pointStart).count();
    }
    
    std::cout << "Time taken for point processing: " << tTotal * 1000 << " ms" << std::endl;
    
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

CameraIntrinsics getRealsenseIntrinsics() {
    // Initialize RealSense pipeline
    rs2::pipeline pipe;
    rs2::config cfg;
    
    // Configure and start the pipeline
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    pipe.start(cfg);
    
    // Get depth sensor intrinsics
    rs2::pipeline_profile profile = pipe.get_active_profile();
    rs2::video_stream_profile depthProfile = profile.get_stream(RS2_STREAM_DEPTH)
                                                   .as<rs2::video_stream_profile>();
    rs2_intrinsics intrinsics = depthProfile.get_intrinsics();
    
    // Stop the pipeline
    pipe.stop();
    
    return CameraIntrinsics::fromRealsense(intrinsics);
}

Eigen::Matrix4f createCameraPoseFromTranslationQuaternion(
    float tx, float ty, float tz,
    float qw, float qx, float qy, float qz
) {
    // Normalize quaternion
    float norm = std::sqrt(qw*qw + qx*qx + qy*qy + qz*qz);
    qw /= norm;
    qx /= norm;
    qy /= norm;
    qz /= norm;
    
    // Quaternion to rotation matrix conversion
    Eigen::Matrix3f R;
    R << 1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qw*qz, 2*qx*qz + 2*qw*qy,
         2*qx*qy + 2*qw*qz, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qw*qx,
         2*qx*qz - 2*qw*qy, 2*qy*qz + 2*qw*qx, 1 - 2*qx*qx - 2*qy*qy;
    
    // Create 4x4 transformation matrix
    Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
    pose.block<3,3>(0,0) = R;
    pose(0,3) = tx;
    pose(1,3) = ty;
    pose(2,3) = tz;
    
    return pose;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <dataset_path>" << std::endl;
        return 1;
    }
    // Load point cloud
    std::string pcl_path = argv[1];
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::io::loadPCDFile<pcl::PointXYZRGB>(pcl_path, *pointCloud);

    CameraIntrinsics intrinsics(500, 500, 320, 260, 640, 520);  // Use these camera intrinsics
    
    Eigen::Vector3f translation = Eigen::Vector3f(-0.0632835105, 0.0298347212, 1.26912034); // headcam translation read from tf
    // Next, I know this camera rotated yaw around its own z-axis, pitch around its own y-axis, roll around its own x-axis (roll=0)
    float yaw = -0.0066029798700795546; // This needs to be read from base's yaw in tf later
    float pitch = 0.50; // This pitch is constant
    float roll = 0.0; // roll is also constant

    Eigen::Matrix3f rotation;
    rotation = Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()) *
               Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY()) *
               Eigen::AngleAxisf(roll, Eigen::Vector3f::UnitX());
    
    Eigen::Matrix4f cameraPose = Eigen::Matrix4f::Identity();
    cameraPose.block<3,3>(0,0) = rotation;
    cameraPose.block<3,1>(0,3) = translation;
    
    auto renderedCloud = renderPointCloudWithOcclusion<pcl::PointXYZRGB>(
        pointCloud,
        cameraPose,
        intrinsics,
        0.1f,
        10.0f
    );

    // Save rendered point cloud
    pcl::io::savePCDFile("rendered_cloud.pcd", *renderedCloud);

    return 0;
} 