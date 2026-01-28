SHELL := /bin/bash

.PHONY: all clean install-ros install-gazebo install-deps submodules run sim teleop rviz

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

# List all source files as dependencies for the build stamp
SOURCES := $(shell find src -type f \( -name '*.cpp' -o -name '*.h' -o -name '*.py' -o -name '*.yaml' -o -name '*.launch.py' -o -name 'package.xml' -o -name 'CMakeLists.txt' \))

all: build

build: .build.stamp

.build.stamp: $(SOURCES)
	source /opt/ros/jazzy/setup.bash && \
		colcon build --symlink-install
	touch $@

clean:
	rm -rf install build log *.stamp

setup: install-ros install-gazebo install-deps

install-ros:
	sudo apt update
	sudo apt install software-properties-common
	sudo add-apt-repository -y universe
	sudo apt install -y curl
	export ROS_APT_SOURCE_VERSION=$$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $$4}') && \
		curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/$${ROS_APT_SOURCE_VERSION}/ros2-apt-source_$${ROS_APT_SOURCE_VERSION}.$$(. /etc/os-release && echo $${UBUNTU_CODENAME:-$${VERSION_CODENAME}})_all.deb" && \
		sudo dpkg -i /tmp/ros2-apt-source.deb
	sudo apt update
	sudo apt install -y \
		ros-dev-tools \
		ros-jazzy-desktop

install-gazebo:
	sudo apt update
	sudo apt install -y curl lsb-release gnupg
	sudo curl https://packages.osrfoundation.org/gazebo.gpg --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
	echo "deb [arch=$$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] https://packages.osrfoundation.org/gazebo/ubuntu-stable $$(lsb_release -cs) main" | \
		sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
	sudo apt update
	sudo apt install gz-harmonic

submodules:
	git submodule update --init --recursive

install-deps: submodules
	sudo mkdir -p /etc/apt/keyrings
	curl -sSf https://librealsense.realsenseai.com/Debian/librealsense.pgp | sudo tee /etc/apt/keyrings/librealsense.pgp > /dev/null
	echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.realsenseai.com/Debian/apt-repo `lsb_release -cs` main" | \
		sudo tee /etc/apt/sources.list.d/librealsense.list && \
	sudo apt update
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
	source /opt/ros/jazzy/setup.bash && \
		source install/setup.bash && \
		$(GPU_PREFIX) ros2 launch py_robot_nav gazebo.launch.py

teleop:
	source /opt/ros/jazzy/setup.bash && \
		source install/setup.bash && \
		ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args --remap cmd_vel:=/cmd_vel

rviz: build
	source /opt/ros/jazzy/setup.bash && \
		source install/setup.bash && \
		$(GPU_PREFIX) rviz2 --display-config robot.rviz

# vim: tabstop=2 softtabstop=2 shiftwidth=2
