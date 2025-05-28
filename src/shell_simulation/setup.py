#!/usr/bin/env python3
"""
ament_python setup script for the Shell Eco-marathon autonomous stack.
"""

from setuptools import setup
import os
from glob import glob

package_name = "shell_simulation" # The name of the Python package directory

setup(
    name=package_name, # In ROS 2, often name and package_name are the same for simplicity
    version="0.0.1",
    packages=[package_name],

    data_files=[
        ("share/ament_index/resource_index/packages",
            ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"),
            glob(os.path.join("launch", "*.launch.py"))), # More specific for .launch.py
        (os.path.join("share", package_name, "config"),
            glob(os.path.join("config", "*.yaml"))),
    ],

    install_requires=[
        "setuptools",
        "numpy",         # Core numerical library
        "scipy",         # For KDTree in perception
        "filterpy",      # For KalmanFilter in perception
        "PyYAML",        # For loading waypoints.yaml in planning
        "opencv-python", # If perception (camera part) or other nodes use it directly via Python import
                         # If only system-level OpenCV is needed, python3-opencv in package.xml is sufficient
    ],
    zip_safe=True,

    author="mkbardi", # From your package.xml
    author_email="muktarbardi@gmail.com", # From your package.xml
    maintainer="mkbardi",
    maintainer_email="muktarbardi@gmail.com",
    description="Shell Eco-marathon autonomous driving stack (planning, perception, control).",
    license="Apache-2.0", # Match package.xml
    tests_require=["pytest"],

    entry_points={
        "console_scripts": [
            "planning_node = shell_simulation.planning_node:main",
            "perception_node = shell_simulation.perception_node:main",
            "control_node = shell_simulation.control_node:main",
        ],
    },
)