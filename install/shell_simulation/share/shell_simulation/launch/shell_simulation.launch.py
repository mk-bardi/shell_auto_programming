#!/usr/bin/env python3
import os
import glob
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import math # For math.radians

def locate_carla_python() -> str:
    roots = [
        os.getenv("CARLA_ROOT", ""),
        os.getenv("CARLA_HOME", ""),
        "/opt/carla-simulator",
        "/opt/carla",
        os.path.expanduser("~/CARLA_0.9.15"),
    ]
    for root in filter(None, roots):
        if not os.path.isdir(root):
            continue
        python_api_dir = os.path.join(root, "PythonAPI")
        if os.path.isdir(python_api_dir):
            eggs = glob.glob(os.path.join(python_api_dir, "carla", "dist", "carla-*py3*.egg"))
            if eggs:
                return f"{python_api_dir}:{eggs[0]}"
        eggs = glob.glob(os.path.join(root, "carla", "dist", "carla-*py3*.egg"))
        if eggs:
            if os.path.basename(root) == "PythonAPI":
                 return f"{root}:{eggs[0]}"
            elif os.path.exists(os.path.join(root, "../PythonAPI")):
                 return f"{os.path.join(root, '../PythonAPI')}:{eggs[0]}"
    print("WARNING: CARLA PythonAPI .egg file not found.")
    return ""

def generate_launch_description() -> LaunchDescription:
    package_name = 'shell_simulation' # REPLACE with your actual package name

    carla_python_path = locate_carla_python()
    env_patch = SetEnvironmentVariable(
        name="PYTHONPATH",
        value=f"{carla_python_path}:{os.environ.get('PYTHONPATH', '')}"
    )

    waypoints_yaml_path = os.path.join(
        get_package_share_directory(package_name),
        "config", "waypoints.yaml",
    )

    declare_debug_perception_arg = DeclareLaunchArgument(
        'debug_perception', default_value='False',
        description='Enable debug logging for perception_node.'
    )
    # Add other debug launch arguments if needed for other nodes

    planning_node = Node(
        package=package_name,
        executable="planning_node.py",
        name="planning_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            {"waypoints_yaml": waypoints_yaml_path},
            {"num_goals_to_pass": 15},
            {"cruise_speed": 8.0}, # Tune for energy
            {"a_lat_max": 1.5},
            # {"debug_log": False}, # If planning_node has a debug_log param
        ],
    )

    perception_node = Node(
        package=package_name,
        executable="perception_node.py",
        name="perception_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            {"cone_width_m": 4.0},
            {"cone_length_m": 25.0},
            {"cluster_eps_m": 0.6},
            {"min_cluster_points": 5},
            {"alert_distance_m": 5.0},
            {"kf_association_gate_m": 2.0},
            {"track_timeout_s": 1.0},
            {"kf_dt_s": 0.1},
            {"kf_process_noise_std": 1.0},
            {"kf_measurement_noise_std": 0.5},
            {"debug_log": LaunchConfiguration('debug_perception')}
        ],
    )

    control_node = Node(
        package=package_name,
        executable="control_node.py",
        name="control_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            {"kp": 0.4}, # PID: Proportional gain
            {"ki": 0.1}, # PID: Integral gain
            {"kd": 0.05},# PID: Derivative gain
            {"i_clamp": 0.3},

            {"cruise_speed_fallback": 7.0},
            {"max_speed_mph": 30.0},

            {"coast_trigger_ttc_s": 6.0},
            {"hard_brake_trigger_ttc_s": 2.0},

            {"lookahead_min_m": 3.0},
            {"lookahead_time_s": 0.4},

            {"goal_reached_threshold_m": 3.0},

            {"vehicle_wheelbase_m": 2.8}, # Tune this!
            {"max_steer_angle_rad": math.radians(35.0)}, # Tune this!
            # {"debug_log": False}, # If control_node has a debug_log param
        ],
    )

    return LaunchDescription([
        env_patch,
        declare_debug_perception_arg,
        # Add other declare_..._arg actions here
        planning_node,
        perception_node,
        control_node,
    ])