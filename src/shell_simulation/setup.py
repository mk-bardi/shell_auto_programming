#!/usr/bin/env python3
"""
ament_python setup script for the Shell Eco-marathon autonomous stack.

• Installs the importable package  →  shell_simulation
• Publishes the distribution name  →  shell-simulation   (hyphen!)
• Exposes three console-script entry points:
      planning_node, perception_node, control_node
• Drops launch files and YAMLs under share/<pkg>/…
"""

from setuptools import setup
import os
from glob import glob

pkg_import_name = "shell_simulation"   # the Python package
dist_name       = "shell-simulation"   # what pip/rosdep will look for

setup(
    name=dist_name,
    version="0.0.1",                   # ← bump when you change metadata
    packages=[pkg_import_name],

    # ----- non-Python assets: launch + config --------------------------------
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + pkg_import_name]),
        ("share/" + pkg_import_name, ["package.xml"]),

        # install all *.py launch files
        (os.path.join("share", pkg_import_name, "launch"),
         glob(os.path.join("launch", "*.py"))),

        # install any YAMLs (e.g. waypoints.yaml)
        (os.path.join("share", pkg_import_name, "config"),
         glob(os.path.join("config", "*.yaml"))),
    ],

    # ----- runtime deps -------------------------------------------------------
    install_requires=[
        "setuptools",
        "numpy",
        "opencv-python",        # used by perception_node
    ],
    zip_safe=True,

    maintainer="YOUR NAME",
    maintainer_email="your@email.com",
    description="Shell Eco-marathon autonomous driving stack (planning, perception, control).",
    license="Apache License 2.0",
    tests_require=["pytest"],

    # ----- map console scripts to the node entry functions -------------------
    entry_points={
        "console_scripts": [
            "planning_node   = shell_simulation.planning_node:main",
            "perception_node = shell_simulation.perception_node:main",
            "control_node    = shell_simulation.control_node:main",
        ],
    },
)
