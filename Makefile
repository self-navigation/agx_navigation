SHELL := /bin/bash

.PHONY: all build clean install-deps run run-sim

WORLD ?= empty.sdf
ODOM_FRAME ?= odom
BASE_FRAME ?= base_link
ODOM_TOPIC_NAME ?= odom
FLOOR_NUMBER ?= 3
PORT_NAME ?= can0
SIMULATED_ROBOT ?= false
CONTROL_RATE ?= 50

USE_GPU_RENDER_ACCELERATION ?= true

ifeq ($(USE_GPU_RENDER_ACCELERATION),true)
GPU_PREFIX := __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia
else
GPU_PREFIX :=
endif

all: build

build:
	source /opt/ros/jazzy/setup.bash && \
		colcon build

clean:
	rm -rf install build log

install-deps:
	git submodule update --init --recursive
	sudo mkdir -p /etc/apt/keyrings
	curl -sSf https://librealsense.realsenseai.com/Debian/librealsense.pgp | sudo tee /etc/apt/keyrings/librealsense.pgp > /dev/null
	echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.realsenseai.com/Debian/apt-repo `lsb_release -cs` main" | \
		sudo tee /etc/apt/sources.list.d/librealsense.list && \
	sudo apt-get update
	sudo apt install -y \
		libasio-dev \
		libpcap-dev \
		librealsense2-dev=2.56.5-0~realsense.17055 \
		librealsense2-utils \
		ros-jazzy-ros-gz \
		ros-jazzy-gz-ros2-control \
		ros-jazzy-diff-drive-controller

run: build
	sudo ip link set $(PORT_NAME) up type can bitrate 500000 || true && \
		source install/setup.bash && \
		ros2 launch py_robot_nav main.launch.py \
		use_sim_time:=$(or $(USE_SIM_TIME),false) \
		port_name:=$(PORT_NAME) \
		odom_frame:=$(ODOM_FRAME) \
		base_frame:=$(BASE_FRAME) \
		odom_topic_name:=$(ODOM_TOPIC_NAME) \
		simulated_robot:=$(SIMULATED_ROBOT) \
		control_rate:=$(CONTROL_RATE)

sim: build
	source install/setup.bash && \
		$(GPU_PREFIX) ros2 launch py_robot_nav gazebo.launch.py

teleop:
	source /opt/ros/jazzy/setup.bash && \
		ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args --remap cmd_vel:=/cmd_vel

rviz: build
	source /opt/ros/jazzy/setup.bash && \
		source install/setup.bash && \
		rviz2

# vim: tabstop=2 softtabstop=2 shiftwidth=2
