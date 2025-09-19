import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node, PushRosNamespace
from launch.conditions import IfCondition


def generate_launch_description():
    package_name = 'rby_wrapper'
    package_path = get_package_share_directory(package_name)
    
    # Common launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    rviz_config_file = os.path.join(package_path, 'config', 'rby_display_multi.rviz')
    
    # XACRO file path instead of URDF
    xacro_file = os.path.join(package_path, 'models', 'rby1a', 'urdf_ros', 'model.xacro')
    
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time if true')
    
    declare_rviz = DeclareLaunchArgument(
        'rviz', 
        default_value='true',
        description='Start RViz')
    
    # Launch arguments for the number of robots
    declare_num_robots = DeclareLaunchArgument(
        'num_robots',
        default_value='2',
        description='Number of robots to spawn')
    
    # Launch rviz only once for all robots
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file],
        condition=IfCondition(LaunchConfiguration('rviz', default='true'))
    )
    
    # Configuration for robot 1 (default robot)
    robot1_group = GroupAction([
        PushRosNamespace('robot1'),
        
        # Robot description for robot1
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'robot_description': Command(['xacro ', xacro_file])
            }]
        ),
        
        # # Optional: Joint State Publisher GUI for robot1
        # Node(
        #     package='joint_state_publisher_gui',
        #     executable='joint_state_publisher_gui',
        #     name='joint_state_publisher_gui',
        #     output='screen',
        #     condition=IfCondition(LaunchConfiguration('gui', default='true'))
        # )
    ])
    
    # Configuration for robot 2
    robot2_group = GroupAction([
        PushRosNamespace('robot2'),
        
        # Robot description for robot2 - note the different robot_name prefix
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'robot_description': Command(['xacro ', xacro_file, ' robot_name:=robot2_'])
            }]
        ),
        
        # # Optional: Joint State Publisher GUI for robot2
        # Node(
        #     package='joint_state_publisher_gui',
        #     executable='joint_state_publisher_gui',
        #     name='joint_state_publisher_gui',
        #     output='screen',
        #     condition=IfCondition(LaunchConfiguration('gui', default='true'))
        # )
    ])
    
    # Create Launch Description
    ld = LaunchDescription([
        declare_use_sim_time,
        declare_rviz,
        DeclareLaunchArgument('gui', default_value='true', description='Start Joint State Publisher GUI'),
        declare_num_robots,
        rviz_node,
        robot1_group
    ])
    
    # Conditionally add robot2 if num_robots > 1
    ld.add_action(robot2_group)
    
    return ld 