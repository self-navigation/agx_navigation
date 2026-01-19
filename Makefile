SHELL := /bin/bash

.PHONY: all build clean install-deps run run-sim

all: build

build:
	source /opt/ros/jazzy/setup.bash && \
	colcon build

clean:
	rm -rf install build log

install-deps:
	sudo mkdir -p /etc/apt/keyrings && \
	curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | sudo tee /etc/apt/keyrings/librealsense.pgp > /dev/null && \
	echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo `lsb_release -cs` main" | \
	sudo tee /etc/apt/sources.list.d/librealsense.list && \
	sudo apt-get update && \
	sudo apt install libasio-dev \
	libpcap-dev \
	librealsense2-dev \
	librealsense2-utils \
	ros-jazzy-gz-ros2-control \
	ros-jazzy-diff-drive-controller

run: build
	source /opt/ros/jazzy/setup.bash && \
	sudo ip link set can0 up type can bitrate 500000 || true && \
	source install/setup.bash && \
	ros2 launch py_robot_nav main.launch.py

sim: build
	source /opt/ros/jazzy/setup.bash && \
	source install/setup.bash && \
	ros2 launch py_robot_nav gazebo.launch.py
