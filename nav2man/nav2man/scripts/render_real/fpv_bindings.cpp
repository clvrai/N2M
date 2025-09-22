#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>

#include "fpv_render.cpp"  // Reuse implementation

namespace py = pybind11;

// Wrapper function that loads a point cloud and applies rendering
int render_from_file(
    const std::string& pcd_path,
    const std::string& output_path,
    const Eigen::Matrix4f& camera_pose,
    const CameraIntrinsics& intrinsics,
    float min_depth = 0.1f,
    float max_depth = 10.0f
) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(pcd_path, *pointCloud) == -1) {
        std::cerr << "Failed to load point cloud file." << std::endl;
        return 1;
    }
    
    std::cout << "Rendering point cloud from " << pcd_path << " to " << output_path << std::endl;

    auto visible_cloud = renderPointCloudWithOcclusion<pcl::PointXYZRGB>(
        pointCloud, camera_pose, intrinsics, min_depth, max_depth
    );

    if (pointCloud->empty()) {
        std::cerr << "Error: Point cloud is empty. No data to save." << std::endl;
        return 2;
    }

    if (visible_cloud->empty()) {
        std::cerr << "Error: Rendered point cloud is empty. No data to save." << std::endl;
        return 2;
    }
    
    // Save rendered point cloud
    pcl::io::savePCDFile(output_path, *visible_cloud);

    return 0;
}

PYBIND11_MODULE(fpv_render, m) {
    py::class_<CameraIntrinsics>(m, "CameraIntrinsics")
        .def(py::init<float, float, float, float, int, int>())
        .def_readwrite("fx", &CameraIntrinsics::fx)
        .def_readwrite("fy", &CameraIntrinsics::fy)
        .def_readwrite("cx", &CameraIntrinsics::cx)
        .def_readwrite("cy", &CameraIntrinsics::cy)
        .def_readwrite("width", &CameraIntrinsics::width)
        .def_readwrite("height", &CameraIntrinsics::height);

    m.def("render_from_file", &render_from_file,
          "Render visible points from point cloud file to numpy array",
          py::arg("pcd_path"),
          py::arg("output_path"),
          py::arg("camera_pose"),
          py::arg("intrinsics"),
          py::arg("min_depth") = 0.1f,
          py::arg("max_depth") = 10.0f);
}