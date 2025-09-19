cmake_minimum_required(VERSION 3.8)
project(rby_wrapper)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rclpy REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(message_filters REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# Generate messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/RobotState.msg"
  "msg/RobotControl.msg"
  DEPENDENCIES builtin_interfaces
)

# Add laser_helper executable
add_executable(laser_helper src/laser_helper.cpp)
ament_target_dependencies(laser_helper 
  rclcpp
  sensor_msgs
  geometry_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
)

# Add client_helper_stitch executable
add_executable(client_helper_stitch src/client_helper_stitch.cpp)
ament_target_dependencies(client_helper_stitch
  rclcpp
  sensor_msgs
  std_msgs
  geometry_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  pcl_conversions
)
target_include_directories(client_helper_stitch PRIVATE ${PCL_INCLUDE_DIRS})
target_link_libraries(client_helper_stitch ${PCL_LIBRARIES} yaml-cpp)
target_compile_definitions(client_helper_stitch PRIVATE ${PCL_DEFINITIONS})

# Link with the generated message interfaces
rosidl_target_interfaces(client_helper_stitch
  ${PROJECT_NAME} "rosidl_typesupport_cpp")

# Add colored point cloud generator executable
# add_executable(campcl_helper src/campcl_helper.cpp)
# ament_target_dependencies(campcl_helper
#   rclcpp
#   sensor_msgs
#   geometry_msgs
#   pcl_conversions
#   cv_bridge
#   image_transport
#   message_filters
# )

# Add camRender_helper executable
add_executable(camRender_helper src/camRender_helper.cpp)
ament_target_dependencies(camRender_helper
  rclcpp
  sensor_msgs
  geometry_msgs
  pcl_conversions
  tf2
  tf2_ros
  tf2_geometry_msgs
)
target_include_directories(camRender_helper PRIVATE ${PCL_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS})
target_link_libraries(camRender_helper ${PCL_LIBRARIES} Eigen3::Eigen)
target_compile_definitions(camRender_helper PRIVATE ${PCL_DEFINITIONS})


# Add base_se2_helper executable
add_executable(base_se2_helper src/base_se2_helper.cpp)
ament_target_dependencies(base_se2_helper
  rclcpp
  sensor_msgs
  geometry_msgs
  pcl_conversions
  tf2
  tf2_ros
  tf2_geometry_msgs
)
target_include_directories(base_se2_helper PRIVATE ${PCL_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS})
target_link_libraries(base_se2_helper ${PCL_LIBRARIES} Eigen3::Eigen)
target_compile_definitions(base_se2_helper PRIVATE ${PCL_DEFINITIONS})

# Add rosidl dependencies
rosidl_get_typesupport_target(cpp_typesupport_target "${PROJECT_NAME}" "rosidl_typesupport_cpp")
target_link_libraries(laser_helper "${cpp_typesupport_target}")

# Install Python scripts
install(PROGRAMS
  scripts/client_data_collection.py
  scripts/server_teleoperation.py
  scripts/server_rby1_robot.py
  scripts/client_manipulation_dp.py
  scripts/client_manipulation_pi0.py
  scripts/client_pcl_frameID_converter.py
  scripts/client_rollout_manager.py
  scripts/client_naive_navigation.py
  scripts/client_SIR_inference.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install C++ executables
install(TARGETS
  base_se2_helper
  laser_helper
  client_helper_stitch
  camRender_helper
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

# Install config files
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
)

# Create maps directory
install(DIRECTORY
  DESTINATION share/${PROJECT_NAME}/maps
)

# Install models directory
install(DIRECTORY
  models
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
