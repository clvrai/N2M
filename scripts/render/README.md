# Point Cloud Renderer

This is a C++ implementation of a point cloud renderer with occlusion handling, converted from a Python implementation for improved performance. It uses the Point Cloud Library (PCL) for point cloud processing and visualization.

## Performance Comparison

The C++ implementation is expected to be significantly faster than the Python version due to:

1. Native code execution without Python interpreter overhead
2. More efficient memory management 
3. Better optimization with compiler flags (`-O3 -march=native`)
4. Efficient point cloud operations using PCL's optimized algorithms
5. Reduced memory allocation overhead in the point processing loop

## Requirements

- C++14 compatible compiler (GCC or Clang recommended)
- CMake (version 3.10 or higher)
- Point Cloud Library (PCL) version 1.8 or higher
- librealsense2 SDK
- Eigen3 library

## Installation

### Installing Dependencies

#### Ubuntu/Debian:

```bash
# Install build tools
sudo apt-get update
sudo apt-get install build-essential cmake

# Install PCL and dependencies
sudo apt-get install libpcl-dev

# Install Eigen3
sudo apt-get install libeigen3-dev

# Install librealsense (following Intel's guide)
# See: https://github.com/IntelRealSense/librealsense/blob/master/doc/installation.md
sudo apt-key adv --keyserver keys.gnupg.net --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" -u
sudo apt-get install librealsense2-dkms librealsense2-utils librealsense2-dev
```

## Building the Project

```bash
# Clone this repository
git clone https://github.com/yourusername/point-cloud-renderer
cd point-cloud-renderer

# Create build directory
mkdir build && cd build

# Configure and build
cmake ..
make -j$(nproc)
```

## Usage

Make sure you have a point cloud file named `7.pcd` in the current directory, or modify the source code to point to your PCD file.

```bash
# Run the renderer
./fpv_render
```

The program will:
1. Load the specified point cloud file
2. Render it from the specified camera perspective
3. Save the result as `rendered_view.pcd`
4. Display the point cloud with coordinate frames in a visualization window

## Comparing with Python Version

To compare performance:

1. Run the Python version and note the reported timing
2. Run the C++ version and note the reported timing
3. Compare the quality of the rendered point clouds

The C++ implementation using PCL is expected to show significant performance improvements, especially for large point clouds, compared to the Python implementation using Open3D. PCL's native C++ implementation avoids Python's interpreter overhead and benefits from highly optimized algorithms specifically designed for point cloud processing. 