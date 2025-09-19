#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

import os

def generate_launch_description():
    package_dir = get_package_share_directory('rby_wrapper')
    
    # Declare launch parameters
    use_sim_time = LaunchConfiguration('use_sim_time')
    slam_params_file = LaunchConfiguration('slam_params_file')
    
    declare_use_sim_time_argument = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation/Gazebo clock'
    )
    
    declare_slam_params_file_cmd = DeclareLaunchArgument(
        'slam_params_file',
        default_value=os.path.join(package_dir, 'config', 'mapping.yaml'),
        description='Full path to the ROS2 parameters file to use for the slam_toolbox node'
    )
    
    # Create slam_toolbox node in online mapping mode
    start_async_slam_toolbox_node = Node(
        parameters=[
            slam_params_file,
            {'use_sim_time': use_sim_time}
        ],
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        arguments=['--ros-args', '--log-level', 'slam_toolbox:=warn'],
        remappings=[
            ('scan', '/scan'),
            ('odom', '/robot/odom'),
        ]
    )
    
    # Create the launch description and populate it
    ld = LaunchDescription()
    
    # Add the declarations
    ld.add_action(declare_use_sim_time_argument)
    ld.add_action(declare_slam_params_file_cmd)
    
    # Add the nodes
    ld.add_action(start_async_slam_toolbox_node)
    
    return ld 