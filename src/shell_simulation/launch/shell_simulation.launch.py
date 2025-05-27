# !/usr/bin/env python3
"""
Launch file that:
  • patches PYTHONPATH so the CARLA Python API (and agents) are importable
  • starts planning, perception and control nodes
"""
import os, glob
from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable
from launch_ros.actions import Node

def locate_carla_python():
    # 1. honour any CARLA_ROOT already set
    roots = [os.getenv("CARLA_ROOT", ""),
             "/opt/carla-simulator",         # deb-install default
             "/opt/carla"]                   # fallback image path
    for root in roots:
        if root and os.path.isdir(root):
            # add <root>/PythonAPI and the *.egg in PythonAPI/carla/dist
            egg_glob = glob.glob(f"{root}/PythonAPI/carla/dist/carla-*py3*.egg")
            if egg_glob:
                return f"{root}/PythonAPI:{egg_glob[0]}"
    # nothing found – leave empty, node imports will raise their own error
    return ""

def generate_launch_description():
    carla_python = locate_carla_python()
    env_patch = SetEnvironmentVariable(
        name="PYTHONPATH",
        value=f"{carla_python}:${{PYTHONPATH}}"
    )

    return LaunchDescription([
        env_patch,
        Node(package="shell_simulation", executable="planning_node",
             name="planning_node", output="screen"),
        Node(package="shell_simulation", executable="perception_node",
             name="perception_node", output="screen"),
        Node(package="shell_simulation", executable="control_node",
             name="control_node", output="screen"),
    ])
