import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Locate your config directory and waypoints file
    config_dir = os.path.join(
        get_package_share_directory('shell_simulation'),
        'config'
    )
    waypoints_yaml_file = os.path.join(config_dir, 'waypoints.yaml')

    # Allow overriding via CLI: --ros-args -p waypoints_yaml:=/some/other.yaml
    declare_waypoints_yaml_cmd = DeclareLaunchArgument(
        'waypoints_yaml',
        default_value=waypoints_yaml_file,
        description='Full path to the waypoints YAML file'
    )

    # Planning node reads the YAML, connects to CARLA
    planning_node_cmd = Node(
        package='shell_simulation',
        executable='planning_node',
        name='planning_node',
        output='screen',
        parameters=[{
            'waypoints_yaml': LaunchConfiguration('waypoints_yaml'),
            'carla_host': 'localhost',
            'carla_port': 2000,
        }]
    )

    # Perception node—e.g. lidar
    perception_node_cmd = Node(
        package='shell_simulation',
        executable='perception_node',
        name='perception_node',
        output='screen',
        parameters=[{
            'use_lidar': True,
        }]
    )

    # Control node—publishes vehicle commands
    control_node_cmd = Node(
        package='shell_simulation',
        executable='control_node',
        name='control_node',
        output='screen',
        parameters=[{
            'wheel_base': 2.8,
            'target_speed_mps': 8.0,
        }]
    )

    return LaunchDescription([
        declare_waypoints_yaml_cmd,
        planning_node_cmd,
        perception_node_cmd,
        control_node_cmd,
    ])
