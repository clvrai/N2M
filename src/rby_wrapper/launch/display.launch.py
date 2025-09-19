import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch.conditions import IfCondition


def generate_launch_description():
    package_name = 'rby_wrapper'
    package_path = get_package_share_directory(package_name)
    
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    urdf_file = LaunchConfiguration('urdf_file', default=os.path.join(
        package_path, 'models', 'rby1a', 'urdf_ros', 'model.urdf'))
    
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time if true')
    
    declare_urdf_file = DeclareLaunchArgument(
        'urdf_file',
        default_value=os.path.join(package_path, 'models', 'rby1a', 'urdf_ros', 'model.urdf'),
        description='Path to the URDF file')
    
    rviz_config_file = os.path.join(package_path, 'config', 'rby_display.rviz')
    
    robot_description = Command(['cat ', urdf_file])
    
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_description': robot_description
        }]
    )
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file],
        condition=IfCondition(LaunchConfiguration('rviz', default='true'))
    )
    
    # joint_state_publisher_gui_node = Node(
    #     package='joint_state_publisher_gui',
    #     executable='joint_state_publisher_gui',
    #     name='joint_state_publisher_gui',
    #     output='screen',
    #     condition=IfCondition(LaunchConfiguration('gui', default='true'))
    # )
    
    return LaunchDescription([
        declare_use_sim_time,
        declare_urdf_file,
        DeclareLaunchArgument('rviz', default_value='true', description='Start RViz'),
        DeclareLaunchArgument('gui', default_value='true', description='Start Joint State Publisher GUI'),
        robot_state_publisher_node,
        rviz_node,
        # joint_state_publisher_gui_node
    ]) 