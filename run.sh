#!/bin/bash
set -e

make
sudo ip link set can0 up type can bitrate 500000 || true  # init CANbus interface
source install/setup.bash  # to get packages in environment
#ros2 launch scout_base scout_base.launch.py is_scout_mini:=true  # to launch robot chassis module
ros2 launch py_robot_nav main.launch.py  # is_scout_mini:=true  # to launch robot chassis module

