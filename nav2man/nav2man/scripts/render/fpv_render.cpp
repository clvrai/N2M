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
        
        // Transform coordinates to match Python implementation
        // In Python, camera looks along Y axis, with Z up and X right
        float x = point.x;  // X stays the same
        float y = point.y; // flip y
        float z = point.z;  // flip z
        
        // Skip if behind camera or invalid
        if (z <= 0 || !std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
            continue;
        }
        
        // Project point to image plane
        float u = (x * intrinsics.fx / z) + intrinsics.cx;
        float v = (y * intrinsics.fy / z) + intrinsics.cy;
        
        // Skip if outside image bounds
        if (u < 0 || u >= intrinsics.width || v < 0 || v >= intrinsics.height) {
            continue;
        }
        
        // Get pixel coordinates
        int pixelX = static_cast<int>(u);
        int pixelY = static_cast<int>(v);
        
        // Skip if outside depth range
        float depth = z;
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

    std::string dataset_path = argv[1];
    std::string pcl_dir = dataset_path + "/rollout/pcl";
    std::string meta_path = dataset_path + "/rollout/meta.json";
    std::string output_pcl_dir = dataset_path + "/rollout/pcl_aug";
    std::string output_meta_path = dataset_path + "/rollout/meta_aug.json";
    std::string camera_poses_path = dataset_path + "/rollout/camera_poses/camera_poses.json";

    // Create output directory if it doesn't exist
    fs::create_directories(output_pcl_dir);

    std::ifstream camera_poses_file(camera_poses_path);
    json camera_poses;
    camera_poses_file >> camera_poses;

    std::ifstream meta_file(meta_path);
    json meta;
    meta_file >> meta;

    CameraIntrinsics intrinsics(
        100.6919557412736, 100.6919557412736, 160.0, 120.0, 320, 240
    );

    json meta_aug;
    meta_aug["meta"] = meta["meta"];
    meta_aug["episodes"] = json::array();

    auto timeStart = std::chrono::high_resolution_clock::now();
    for (const auto& episode : meta["episodes"]) {
        // Load point cloud
        std::string pcl_path = fs::path(dataset_path) / "rollout" / episode["file_path"];
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(pcl_path, *pointCloud) == -1) {
            std::cerr << "Failed to load point cloud file: " << pcl_path << std::endl;
            continue;
        }

        int episode_id = episode["id"];
        for (int i = 0; i < camera_poses[episode_id].size(); i++) {
            json episode_aug = episode;
            
            std::string file_name = std::to_string(episode_id) + "_" + std::to_string(i) + ".pcd";
            std::string output_path = (fs::path(output_pcl_dir) / file_name).string();
            if (fs::exists(output_path)) {
                std::cout << "Skipping: " << output_path << std::endl;
                continue;
            }

            // define camera pose
            Eigen::Matrix4f cameraPose = Eigen::Matrix4f::Identity();
            for (int j = 0; j < 4; j++) {
                for (int k = 0; k < 4; k++) {
                    cameraPose(j, k) = camera_poses[episode_id][i][j][k];
                }
            };

            // Render point cloud
            auto renderedCloud = renderPointCloudWithOcclusion<pcl::PointXYZRGB>(
                pointCloud,
                cameraPose,
                intrinsics,
                0.1f,
                10.0f
            );

            // Save rendered point cloud
            pcl::io::savePCDFile(output_path, *renderedCloud);

            episode_aug["file_path"] = "pcl_aug/" + file_name;
            meta_aug["episodes"].push_back(episode_aug);

            auto timeEnd = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration<double>(timeEnd - timeStart).count();
            
            std::cout << "Rendered point cloud saved to: " << output_path << std::endl;
            std::cout << "Number of points: " << renderedCloud->size() << std::endl;
            std::cout << "Total time: " << duration << " s" << std::endl;
            std::cout << "----------------------------------------" << std::endl;
        }
    }

    // Save augmented meta file
    std::ofstream output_meta_file(output_meta_path);
    output_meta_file << meta_aug.dump(4);
    std::cout << "Augmented meta file saved to: " << output_meta_path << std::endl;
    
    return 0;
} 