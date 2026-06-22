SHELL := /bin/bash

.PHONY: all clean install-ros install-gazebo install-deps can-bus run teleop rviz test online offline nav2 rl-deps rl-sim rl-train rl-kill

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
							PMP_MODE \
							USE_SERVER \
							DO_CORRECTIONS \
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

PYTHON ?= /usr/bin/python3

SOURCES := $(shell find src -type f | sed 's/ /\\ /g')
PYTHON_SETUP_FILES := $(shell find src -name "setup.py")
PYTHON_PACKAGES := $(dir $(PYTHON_SETUP_FILES))

all: build

build: deps .build.stamp

.build.stamp: $(SOURCES)
	source /opt/ros/jazzy/setup.bash && \
		colcon build --base-paths src
	touch $@

clean:
	rm -rf install build log .*.stamp

setup: install-ros install-gazebo system-deps deps

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

.ros_python_deps.stamp: $(PYTHON_SETUP_FILES)
	pip install --break-system-packages $(PYTHON_PACKAGES)
	touch $@

deps: ros-deps .ros_python_deps.stamp

# RL runtime-corrector training stack (SAC). torch is also needed on-robot for
# policy inference; stable-baselines3[extra] pulls tensorboard + the progress bar
# train.py uses. Kept out of `deps` so the normal build doesn't drag in torch.
rl-deps:
	pip install --break-system-packages 'stable-baselines3[extra]' torch gymnasium

$(ACADOS_BIN)/t_renderer:
	cd $(TERA_RENDERER_ROOT) && \
		cargo build --release && \
		mkdir -p $(ACADOS_BIN) && \
		cp $(TERA_RENDERER_ROOT)/target/release/t_renderer $(ACADOS_BIN)

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
		LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:$(ACADOS_LIB) \
		ros2 launch $(DEBUG_INFIX) \
		agx_bringup main.launch.py \
		$(PARAMS)

# Convenience entry points wrapping `run` with the right nav/planner mode.
# All accept the usual overrides (e.g. `make online SIM=false`).
#   online  -- vec-pmp stack, planner runs its own control loop (live BVP).
#   offline -- vec-pmp stack, planner rolls out a full plan; corrector plays it back.
#   nav2    -- the nav2 navigation stack instead of vec-pmp.
online:
	$(MAKE) run NAV_MODE=vec-pmp PMP_MODE=online

offline:
	$(MAKE) run NAV_MODE=vec-pmp PMP_MODE=offline

nav2:
	$(MAKE) run NAV_MODE=nav2

# ---- RL runtime corrector: training ----------------------------------------
# Two terminals. First bring up the MINIMAL sim (physics + wheel controller +
# odom only -- no nav/planner/corrector, since the trainer IS the command
# source); then run the trainer against it. Override the usual way, e.g.
#   make rl-sim HEADLESS=true
#   make rl-train TIMESTEPS=500000 TERRAIN=false POLICY_OUT=~/my_policy
TIMESTEPS   ?= 200000
TERRAIN     ?= true
POLICY_OUT  ?= $(HOME)/rl_corrector_policy
LOAD        ?=
TB          ?=
TRAIN_ARGS  ?=
# Rendering sensors (GPU lidar + RGB/depth cameras) off by default: the trainer
# consumes only ground-truth pose + the /odom twist + the IMU, so the rendering is
# pure per-tick overhead. Set SIM_SENSORS=true to inspect them. (The IMU and
# magnetometer stay on regardless -- they have no rendering cost.)
SIM_SENSORS ?= false
TERRAIN_FLAG := $(if $(filter false,$(call lc,$(TERRAIN))),--no-terrain,--terrain)
LOAD_FLAG := $(if $(LOAD),--load $(LOAD),)
TB_FLAG := $(if $(TB),--tensorboard $(TB),)

rl-sim: build
	source /opt/ros/jazzy/setup.bash && \
		source install/setup.bash && \
		$(GPU_PREFIX) \
		ros2 launch agx_bringup rl_corrector_sim.launch.py \
		headless:=$(call lc,$(HEADLESS)) \
		sim_sensors:=$(call lc,$(SIM_SENSORS))

rl-train: build
	source /opt/ros/jazzy/setup.bash && \
		source install/setup.bash && \
		ros2 run agx_planning rl_corrector_train \
		--timesteps $(TIMESTEPS) \
		$(TERRAIN_FLAG) \
		--out $(POLICY_OUT) \
		$(LOAD_FLAG) \
		$(TB_FLAG) \
		$(TRAIN_ARGS)

# Kill any leftover sim/training processes. Run BEFORE a fresh rl-sim if a
# previous run was Ctrl-C'd or orphaned -- two sims share gz's default transport
# partition and break set_pose. Each pkill is prefixed with `-` so a "no process
# matched" (exit 1) doesn't fail the target. Order: trainer, then the launch and
# its gz server + ros_gz bridges (the launch can respawn children if killed last).
rl-kill:
	-pkill -f "rl_corrector_train"
	-pkill -f "rl_corrector_sim"
	-pkill -f "ros_gz_bridge/parameter_bridge"
	-pkill -f "ros_gz_sim"
	-pkill -f "gz sim"
	@sleep 1
	@echo "rl-kill: remaining gz/sim procs:"; \
		pgrep -af "gz sim|rl_corrector_sim|rl_corrector_train|parameter_bridge" \
		| grep -v pgrep || echo "  (none)"

server:
	source /opt/ros/jazzy/setup.bash && \
		source install/setup.bash && \
		ros2 launch $(DEBUG_INFIX) \
		agx_bringup planner.launch.py \
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

test:
	PYTHONPATH=src/agx_navigation/agx_planning $(PYTHON) -m pytest src/agx_navigation/agx_planning/test/unit/ -v

# vim: tabstop=2 softtabstop=2 shiftwidth=2
