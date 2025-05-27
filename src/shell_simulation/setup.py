from setuptools import setup
import os
from glob import glob

package_name = 'shell_simulation'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
                (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mkbardi',
    maintainer_email='muktarbardi@gmail.com',
    description='Shell Eco-marathon Autonomous Programming Competition Package',
    license='Apache License 2.0', 
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'control_node = shell_simulation.control_node:main',
            'perception_node = shell_simulation.perception_node:main',
            'planning_node = shell_simulation.planning_node:main',
        ],
    },
)