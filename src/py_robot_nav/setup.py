from setuptools import find_packages, setup

# https://robotics.stackexchange.com/a/25014
import os
from glob import glob

package_name = 'py_robot_nav'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='agilex',
    maintainer_email='you@example.com',
    description='TODO: Package description',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'agx_nav = py_robot_nav.agx_nav:main',
            'agx_odometry_publisher = py_robot_nav.agx_odometry_publisher:main',
        ],
    },
)
