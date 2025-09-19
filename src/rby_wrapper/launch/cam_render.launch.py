import launch
import launch.actions
import launch.substitutions
import launch.launch_description_sources
import launch_ros.actions
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    """Launch the camRender_helper node."""
    
    # Create the camRender_helper node
    cam_render_node = launch_ros.actions.Node(
        package='rby_wrapper',
        executable='camRender_helper',
        name='cam_render_helper',
        output='screen',
        emulate_tty=True,
        parameters=[
            # Add any necessary parameters here
        ]
    )

    # Return the launch description
    return launch.LaunchDescription([
        cam_render_node,
    ]) 