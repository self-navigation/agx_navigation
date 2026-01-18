#!/bin/bash
set -e

make
source install/setup.bash  # to get packages in environment
ros2 launch py_robot_nav gazebo.launch.py  # is_scout_mini:=true  # to launch robot chassis module

