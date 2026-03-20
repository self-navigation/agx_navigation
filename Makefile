SHELL := /bin/bash

.PHONY: all clean update-caches install-ros install-gazebo install-deps can-bus run teleop rviz

SIM ?= true
HEADLESS ?= false
PORT_NAME ?= can0
TELEOP_RAW ?= false

DEBUG ?= false
USE_GPU_RENDER_ACCELERATION ?= true

PARAM_VARS := SIM \
							FLOOR_NUMBER \
							HEADLESS \
							NAV_MODE \
							PORT_NAME

define lc
$(shell echo '$(1)' | tr '[:upper:]' '[:lower:]')
endef

PARAMS := $(strip \
	$(foreach var,$(PARAM_VARS), \
		$(if $($(var)), \
			$(call lc,$(var)):=$($(var)) \
		) \
	) \
)

ifeq ($(USE_GPU_RENDER_ACCELERATION),true)
GPU_PREFIX := __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia
else
GPU_PREFIX :=
endif

ifeq ($(TELEOP_RAW),false)
ASSISTED_TELEOP_START := ros2 action send_goal /assisted_teleop nav2_msgs/action/AssistedTeleop "{time_allowance: {sec: 0, nanosec: 0}}"
ASSISTED_TELEOP_END := ros2 service call /assisted_teleop/_action/cancel_goal action_msgs/srv/CancelGoal
TELEOP_TOPIC := /cmd_vel_assisted_teleop
else
ASSISTED_TELEOP_START := 
ASSISTED_TELEOP_END :=
TELEOP_TOPIC := $(or $(MOTION_CMD_TOPIC_NAME), /cmd_vel)
endif

ifeq ($(DEBUG),true)
DEBUG_INFIX := --debug
else
DEBUG_INFIX :=
endif

SOURCES := $(shell find src -type f | sed 's/ /\\ /g')

all: build

build: update-caches ros-deps .build.stamp

update-caches:
	if [ "$(shell cat .last_build_user 2>/dev/null)" != "$$USER" ]; then \
		grep -E -rl '/home/[^/]+' ./build | xargs -I {} sh -c ' \
			file="$$1"; \
			mtime=$$(stat -c %Y "$$file" 2>/dev/null || echo ""); \
			sed -E -i "s|/home/[^/]+|/home/$$USER|g" "$$file"; \
			if [ -n "$$mtime" ] && [ -f "$$file" ]; then \
				touch -d "@$$mtime" "$$file" 2>/dev/null || true; \
			fi \
		' _ {}; \
		echo "$$USER" > .last_build_user; \
	fi

.build.stamp: $(SOURCES)
	source /opt/ros/jazzy/setup.bash && \
		colcon build
	touch $@

clean:
	rm -rf install build log .*.stamp .last_build_user

setup: install-ros install-gazebo system-deps ros-deps

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

system-deps:
	sudo mkdir -p /etc/apt/keyrings
	curl -sSf https://librealsense.realsenseai.com/Debian/librealsense.pgp | sudo tee /etc/apt/keyrings/librealsense.pgp > /dev/null
	echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.realsenseai.com/Debian/apt-repo `lsb_release -cs` main" | \
		sudo tee /etc/apt/sources.list.d/librealsense.list && \
	sudo apt update
	sudo apt install -y \
		libasio-dev \
		libpcap-dev \
		librealsense2-dev \
		librealsense2-utils \
		python3-pip

ros-deps:
	rosdep install --from-paths src --ignore-src -r -y
	@find src -name "setup.py" -printf '%h\n' | while read pkg_dir; do \
		pip install --break-system-packages "$$pkg_dir"; \
	done

can-bus:
	if [ "$(SIM)" != true ] ; then \
		if ! ip link show $(PORT_NAME) up | grep -q "$(PORT_NAME)"; then \
			sudo ip link set $(PORT_NAME) up type can bitrate 500000; \
		else \
			echo "CAN interface $(PORT_NAME) is already up, skipping."; \
		fi \
	fi

run: build can-bus
	source /opt/ros/jazzy/setup.bash && \
		source install/setup.bash && \
		$(GPU_PREFIX) \
		ros2 launch $(DEBUG_INFIX) \
		agx_bringup main.launch.py \
		$(PARAMS)

teleop:
	source /opt/ros/jazzy/setup.bash && \
		$(ASSISTED_TELEOP_START) &> /dev/null &
	source /opt/ros/jazzy/setup.bash && \
		source install/setup.bash && \
		ros2 run teleop_twist_keyboard teleop_twist_keyboard \
		--ros-args --remap \
		cmd_vel:=$(TELEOP_TOPIC) \
		--param stamped:=true
	source /opt/ros/jazzy/setup.bash && \
		$(ASSISTED_TELEOP_END) &> /dev/null &

# Using raw rviz command during development
# TODO: switch to rviz.launch.py
# once all visualisations are configured
rviz: build
	source /opt/ros/jazzy/setup.bash && \
		source install/setup.bash && \
		$(GPU_PREFIX) rviz2 \
		--display-config ./src/agx_navigation/agx_bringup/rviz/main.rviz \
		--ros-args --param use_sim_time:=$(SIM)

# vim: tabstop=2 softtabstop=2 shiftwidth=2
