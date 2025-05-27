#!/usr/bin/env python3
"""
Launches the full SEM autonomy stack and makes CARLA's PythonAPI importable.

• Adds <CARLA_ROOT>/PythonAPI and its *.egg to PYTHONPATH
• Resolves waypoints.yaml from the installed share directory
• Starts planning, perception, and control nodes
"""

import os
import glob
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable
from launch_ros.actions import Node

def locate_carla_python() -> str:
    """Return 'PythonAPI:egg_path' or '' if CARLA cannot be found."""
    roots = [
        os.getenv("CARLA_ROOT", ""),      # honour user env
        "/opt/carla-simulator",           # default deb location
        "/opt/carla",                     # common container path
    ]
    for root in filter(os.path.isdir, roots):
        eggs = glob.glob(f"{root}/PythonAPI/carla/dist/carla-*py3*.egg")
        if eggs:
            return f"{root}/PythonAPI:{eggs[0]}"
    return ""

def generate_launch_description() -> LaunchDescription:
    # 1) Patch PYTHONPATH for every node
    env_patch = SetEnvironmentVariable(
        name="PYTHONPATH",
        value=f"{locate_carla_python()}:${{PYTHONPATH}}"
    )

    # 2) Absolute path to waypoints.yaml installed with the package
    waypoints_yaml = os.path.join(
        get_package_share_directory("shell_simulation"),
        "config", "waypoints.yaml",
    )

    # 3) Launch the three functional nodes
    return LaunchDescription([
        env_patch,

        Node(
            package="shell_simulation",
            executable="planning_node",
            name="planning_node",
            output="screen",
            parameters=[{"waypoints_yaml": waypoints_yaml}],
        ),
        Node(
            package="shell_simulation",
            executable="perception_node",
            name="perception_node",
            output="screen",
        ),
        Node(
            package="shell_simulation",
            executable="control_node",
            name="control_node",
            output="screen",
        ),
    ])
