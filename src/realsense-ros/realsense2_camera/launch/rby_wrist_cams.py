# Copyright 2023 Intel Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# DESCRIPTION #
# ----------- #
# Use this launch file to launch 2 devices.
# The Parameters available for definition in the command line for each camera are described in rs_launch.configurable_parameters
# For each device, the parameter name was changed to include an index.
# For example: to set camera_name for device1 set parameter camera_name1.
# command line example:
# ros2 launch realsense2_camera rs_multi_camera_launch.py camera_name1:=D400 device_type2:=l5. device_type1:=d4..

"""Launch realsense2_camera node."""
import copy
from launch import LaunchDescription, LaunchContext
import launch_ros.actions
from launch.actions import IncludeLaunchDescription, OpaqueFunction
from launch.substitutions import LaunchConfiguration, ThisLaunchFileDir
from launch.launch_description_sources import PythonLaunchDescriptionSource
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.absolute()))
import rs_launch

# Define serial numbers for left and right cameras
LEFT_CAMERA_SN = "409122274689"  # Left camera serial number
RIGHT_CAMERA_SN = "427622271135"  # Right camera serial number

# Default parameter configuration
default_parameters = {
    'camera_name1': 'left_camera',
    'camera_name2': 'right_camera',
    'camera_namespace1': 'left_camera',
    'camera_namespace2': 'right_camera',
    'serial_no1': f'"{LEFT_CAMERA_SN}"',  # Add quotes to ensure string type
    'serial_no2': f'"{RIGHT_CAMERA_SN}"',  # Add quotes to ensure string type
    'device_type1': "'d4..'",  # Force D4xx device type for left camera
    'device_type2': "'d4..'",  # Force D4xx device type for right camera
    'initial_reset1': 'true',  # Reset device before starting
    'initial_reset2': 'true',  # Reset device before starting
    'rgb_camera.color_profile1': '424x240x30',  # Fixed parameter name
    'rgb_camera.color_profile2': '424x240x30',  # Fixed parameter name
    # 'depth_module.depth_profile1': '640x480x30',
    # 'depth_module.depth_profile2': '640x480x30',
    'enable_color1': 'true',
    'enable_color2': 'true',
    'enable_depth1': 'false',
    'enable_depth2': 'false',
    'enable_sync1': 'false',
    'enable_sync2': 'false',
    'enable_rgbd1': 'false',
    'enable_rgbd2': 'false',
    'publish_tf1': 'false',
    'publish_tf2': 'false',
    'align_depth.enable1': 'false',  # Enable depth alignment
    'align_depth.enable2': 'false',  # Enable depth alignment
    'depth_module.enable_auto_exposure1': 'true',  # Enable auto exposure
    'depth_module.enable_auto_exposure2': 'true',  # Enable auto exposure
    'depth_module.exposure1': '8500',  # Set exposure value
    'depth_module.exposure2': '8500',  # Set exposure value
    'depth_module.gain1': '16',  # Set gain value
    'depth_module.gain2': '16',  # Set gain value
    'depth_module.power_line_frequency1': '1',  
    'depth_module.power_line_frequency2': '1',  
    'depth_module.power_line_frequency': '1',
    'rgb_camera.power_line_frequency1': '1',
    'rgb_camera.power_line_frequency2': '1',
    'rgb_camera.power_line_frequency': '1',
    'wait_for_device_timeout1': '10.0',  # Timeout for device connection
    'wait_for_device_timeout2': '10.0',  # Timeout for device connection
}

local_parameters = [{'name': 'camera_name1', 'default': default_parameters['camera_name1'], 'description': 'left camera unique name'},
                    {'name': 'camera_name2', 'default': default_parameters['camera_name2'], 'description': 'right camera unique name'},
                    {'name': 'camera_namespace1', 'default': default_parameters['camera_namespace1'], 'description': 'left camera namespace'},
                    {'name': 'camera_namespace2', 'default': default_parameters['camera_namespace2'], 'description': 'right camera namespace'},
                    {'name': 'serial_no1', 'default': default_parameters['serial_no1'], 'description': 'left camera serial number'},
                    {'name': 'serial_no2', 'default': default_parameters['serial_no2'], 'description': 'right camera serial number'},
                    ]

def set_configurable_parameters(local_params):
    return dict([(param['original_name'], LaunchConfiguration(param['name'])) for param in local_params])

def duplicate_params(general_params, posix):
    local_params = copy.deepcopy(general_params)
    for param in local_params:
        param['original_name'] = param['name']
        param['name'] += posix
        # Set default values
        if param['name'] in default_parameters:
            param['default'] = default_parameters[param['name']]
    return local_params

def launch_static_transform_publisher_node(context : LaunchContext):
    # Static transformation from left camera to right camera
    node = launch_ros.actions.Node(
            package = "tf2_ros",
            executable = "static_transform_publisher",
            arguments = ["0", "0", "0", "0", "0", "0",
                          context.launch_configurations['camera_name1'] + "_link",
                          context.launch_configurations['camera_name2'] + "_link"]
    )
    return [node]

def generate_launch_description():
    params1 = duplicate_params(rs_launch.configurable_parameters, '1')
    params2 = duplicate_params(rs_launch.configurable_parameters, '2')
    return LaunchDescription(
        rs_launch.declare_configurable_parameters(local_parameters) +
        rs_launch.declare_configurable_parameters(params1) +
        rs_launch.declare_configurable_parameters(params2) +
        [
        OpaqueFunction(function=rs_launch.launch_setup,
                       kwargs = {'params'           : set_configurable_parameters(params1),
                                 'param_name_suffix': '1'}),
        OpaqueFunction(function=rs_launch.launch_setup,
                       kwargs = {'params'           : set_configurable_parameters(params2),
                                 'param_name_suffix': '2'}),
        OpaqueFunction(function=launch_static_transform_publisher_node)
    ])
