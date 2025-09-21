cmake_minimum_required(VERSION 3.10)
project(PointCloudRenderer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# Find packages
find_package(PCL 1.8 REQUIRED)
find_package(Eigen3 REQUIRED)

# Include PCL definitions and directories
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Add executable
add_executable(fpv_render fpv_render.cpp)

# Link libraries
target_link_libraries(fpv_render 
    ${PCL_LIBRARIES}
    Eigen3::Eigen
    stdc++fs  # Add filesystem library
)

# Include directories
target_include_directories(fpv_render PRIVATE
    ${PCL_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}/include
)

# Compiler flags
if(CMAKE_COMPILER_IS_GNUCXX)
    target_compile_options(fpv_render PRIVATE -O3 -march=native)
endif() 