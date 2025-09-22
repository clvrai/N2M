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

namespace nlohmann {
    // Convert Eigen::Vector3f to JSON
    void to_json(json& j, const Eigen::Vector3f& v) {
        j = json::array({v(0), v(1), v(2)});
    }

    // Convert JSON to Eigen::Vector3f
    void from_json(const json& j, Eigen::Vector3f& v) {
        v(0) = j[0].get<float>();
        v(1) = j[1].get<float>();
        v(2) = j[2].get<float>();
    }

    // Convert Eigen::Vector4f to JSON
    void to_json(json& j, const Eigen::Vector4f& v) {
        j = json::array({v(0), v(1), v(2), v(3)});
    }

    // Convert JSON to Eigen::Vector4f
    void from_json(const json& j, Eigen::Vector4f& v) {
        v(0) = j[0].get<float>();
        v(1) = j[1].get<float>();
        v(2) = j[2].get<float>();
        v(3) = j[3].get<float>();
    }
}

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
typename pcl::PointCloud<PointT>::Ptr translatePointCloud(
    const typename pcl::PointCloud<PointT>::Ptr& pointCloud,
    const Eigen::Matrix4f& basePose
) {
    // Create output point cloud
    typename pcl::PointCloud<PointT>::Ptr transformedCloud(new pcl::PointCloud<PointT>());
    
    // Create inverse transformation to move base pose to origin
    Eigen::Matrix4f inverseTransform = basePose.inverse();
    
    // Transform the point cloud
    pcl::transformPointCloud(*pointCloud, *transformedCloud, inverseTransform);
    
    return transformedCloud;
}

template<typename PointT>
struct TranslatedTarget {
    Eigen::Vector3f position;
    Eigen::Vector4f quaternion;
    Eigen::Vector4f se2_pose;
};

template<typename PointT>
TranslatedTarget<PointT> translateTarget(
    const Eigen::Vector4f& target_se2,
    const Eigen::Matrix4f& basePose
) {
    // Transform SE(2) pose
    // SE(2) pose is [x, y, theta]
    float x = target_se2(0);
    float y = target_se2(1);
    float theta = target_se2(2);
    float z = target_se2(3);
    
    // Create SE(2) transformation matrix
    Eigen::Matrix3f se2_transform;
    se2_transform << cos(theta), -sin(theta), x,
                     sin(theta), cos(theta), y,
                     0, 0, 1;
    
    // Get base SE(2) transformation
    float base_theta = atan2(basePose(1,0), basePose(0,0));
    Eigen::Matrix3f base_se2;
    base_se2 << cos(base_theta), -sin(base_theta), basePose(0,3),
                sin(base_theta), cos(base_theta), basePose(1,3),
                0, 0, 1;
    
    // Transform SE(2) pose
    Eigen::Matrix3f transformed_se2 = base_se2.inverse() * se2_transform;
    
    // Extract transformed SE(2) parameters
    Eigen::Vector4f translated_se2;
    translated_se2(0) = transformed_se2(0,2);
    translated_se2(1) = transformed_se2(1,2);
    translated_se2(2) = atan2(transformed_se2(1,0), transformed_se2(0,0));
    translated_se2(3) = z;
    
    // Return transformed parameters
    TranslatedTarget<PointT> result;
    result.se2_pose = translated_se2;
    
    return result;
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
    std::string meta_path = dataset_path + "/rollout/meta_positive.json";
    std::string output_pcl_dir = dataset_path + "/rollout/pcl_aug_positive_robot_centric_origin";
    std::string output_meta_path = dataset_path + "/rollout/meta_aug_positive_robot_centric_origin.json";
    std::string camera_poses_path = dataset_path + "/rollout/camera_poses/camera_poses.json";
    std::string base_poses_path = dataset_path + "/rollout/camera_poses/base_poses.json";

    // Create output directory if it doesn't exist
    fs::create_directories(output_pcl_dir);

    std::ifstream camera_poses_file(camera_poses_path);
    json camera_poses;
    camera_poses_file >> camera_poses;

    std::ifstream meta_file(meta_path);
    json meta;
    meta_file >> meta;

    std::ifstream base_poses_file(base_poses_path);
    json base_poses;
    base_poses_file >> base_poses;

    CameraIntrinsics intrinsics(
        500, 500, 320, 260, 640, 520
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

            // translate point cloud
            Eigen::Matrix4f basePose = Eigen::Matrix4f::Identity();
            for (int j = 0; j < 4; j++) {
                for (int k = 0; k < 4; k++) {
                    basePose(j, k) = base_poses[episode_id][i][j][k];
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

            // translate point cloud
            auto translatedCloud = translatePointCloud<pcl::PointXYZRGB>(renderedCloud, basePose);

            // translate target
            auto pose = episode_aug["pose"];
            auto translatedTarget = translateTarget<pcl::PointXYZRGB>(
                pose["se2"],
                basePose
            );
            episode_aug["pose"]["se2"] = translatedTarget.se2_pose;

            // Save rendered point cloud
            pcl::io::savePCDFile(output_path, *translatedCloud);

            episode_aug["file_path"] = "pcl_aug_positive_robot_centric_origin/" + file_name;
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