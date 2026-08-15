SHELL := /bin/bash

.PHONY: all clean install-ros install-gazebo install-deps can-bus run teleop rviz test online offline nav2 fixture rl-deps rl-sim rl-train rl-kill p0 p1 p2 p3 curriculum

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
							CORRECTOR \
							PLAYBACK_INDEX \
							LOCALIZATION \
							SURFACE_PATCHES \
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

# ---- parallel sims: WORKER -------------------------------------------------
# WORKER=n runs the whole stack in its own Gazebo transport partition (agxN) AND
# its own DDS domain (40+n), so N sims coexist on one box. Both are needed and
# why is documented in tools/with-worker, which is the same mapping for anything
# invoked outside make (`tools/with-worker 1 python3 -m agx_planning.tuning.soak`).
#
# UNSET MUST STAY BYTE-IDENTICAL to the single-sim behaviour: every number in
# CLAUDE.md was measured in the default partition, so the plain `make rl-sim` /
# `make fixture` path may not gain an env var it did not have before. Hence the
# empty WORKER_ENV rather than a "worker 0" that exports defaults explicitly.
#
#   make rl-sim WORKER=1 HEADLESS=true     # sim on partition agx1, domain 41
#   make rl-train WORKER=1 ...             # trainer that talks to THAT sim
WORKER ?=
ifneq ($(strip $(WORKER)),)
ifeq ($(filter 1 2 3 4 5 6 7 8 9,$(strip $(WORKER))),)
$(error WORKER must be an integer 1-9 (got '$(WORKER)'); leave it unset for the default sim)
endif
WORKER_ENV := GZ_PARTITION=agx$(strip $(WORKER)) ROS_DOMAIN_ID=$(shell expr 40 + $(strip $(WORKER))) AGX_WORKER=$(strip $(WORKER))
else
WORKER_ENV :=
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
		$(WORKER_ENV) \
		$(GPU_PREFIX) \
		LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:$(ACADOS_LIB) \
		ros2 launch $(DEBUG_INFIX) \
		agx_bringup main.launch.py \
		$(PARAMS) $(EXTRA_PARAMS)

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

# Controller test rig: the vec-pmp stack on a PRE-BAKED map instead of live SLAM.
# Skips rtabmap entirely, so a plan can be requested from a standing start and
# the map is identical every run -- which is what makes two corrector runs
# comparable. Nothing but the amcl mode consumes
# the lidar/cameras, hence sim_sensors:=false and a much better realtime factor.
#   make fixture CORRECTOR=tvlqr     # the corrector under test
#   make fixture CORRECTOR=identity  # the do-nothing baseline to compare against
# Rebake the map with rudn_ordjo_building's tools/bake_floor_map.py.
#
# SURFACE_PATCHES defaults to true here, matching `make run`. Set it false to
# debug PLANNER geometry: with slip patches on, a wall strike is ambiguous
# between a bad plan and a slip excursion off a good one.
#   make fixture SURFACE_PATCHES=false
#
# PLAYBACK_INDEX selects how the offline playback cursor advances:
#   time     -- (default) one plan sample per control tick. Every correction
#               costs forward speed and the plan runs out short of the goal.
#   progress -- project the measured pose onto the plan, so a slow robot takes
#               longer rather than stopping early.
#   make fixture CORRECTOR=tvlqr PLAYBACK_INDEX=progress
#
# LOCALIZATION picks what estimates map->odom (see main.launch.py). It defaults
# to `truth` HERE -- unlike `make run`, which defaults to slam -- because this is
# the corrector rig: under `none` the robot navigates on wheel odometry, which
# over-reports distance by 0.6-0.7 m per run, and a pose-feedback corrector
# drives to that bias. `none` measures odometry, not the corrector.
#   make fixture LOCALIZATION=truth   # (default) corrector performance ceiling
#   make fixture LOCALIZATION=amcl    # localize the lidar against the baked map
#   make fixture LOCALIZATION=none    # identity map->odom; fastest, least honest
#
# Only amcl consumes the lidar, so only amcl pays for the rendering sensors.
FIXTURE_LOCALIZATION := $(or $(LOCALIZATION),truth)
FIXTURE_SENSORS := $(if $(filter amcl,$(FIXTURE_LOCALIZATION)),true,false)

fixture:
	$(MAKE) run NAV_MODE=vec-pmp PMP_MODE=offline \
		WORKER=$(strip $(WORKER)) \
		LOCALIZATION=$(FIXTURE_LOCALIZATION) \
		EXTRA_PARAMS="sim_sensors:=$(FIXTURE_SENSORS)"

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
# Training backend: gazebo (real physics, needs rl-sim up) or kinematic (fast,
# Gazebo-free analytic bridge for pretraining a baseline). See `p0`.
BRIDGE      ?= gazebo
# CPU thread cap for torch. The SAC nets are tiny MLPs, so 1 thread is fastest
# and steadiest on a contended desktop; raise only on an idle many-core box.
TORCH_THREADS ?= 1
BRIDGE_FLAG := --bridge $(call lc,$(BRIDGE))
# Rendering sensors (GPU lidar + RGB/depth cameras) off by default: the trainer
# consumes only ground-truth pose + the /odom twist + the IMU, so the rendering is
# pure per-tick overhead. Set SIM_SENSORS=true to inspect them. (The IMU and
# magnetometer stay on regardless -- they have no rendering cost.)
SIM_SENSORS ?= false
# rl_corrector.world runs uncapped (~33x), which is what training wants but puts
# /imu/data at ~3 kHz -- far faster than a normal subscriber drains its queue, so
# any drive-and-measure tool silently loses most samples. Use
# WORLD=rl_corrector_rt.world (1x realtime) for slip_ident and friends.
WORLD ?= rl_corrector.world
TERRAIN_FLAG := $(if $(filter false,$(call lc,$(TERRAIN))),--no-terrain,--terrain)
LOAD_FLAG := $(if $(LOAD),--load $(LOAD),)
TB_FLAG := $(if $(TB),--tensorboard $(TB),)

rl-sim: build
	source /opt/ros/jazzy/setup.bash && \
		source install/setup.bash && \
		$(WORKER_ENV) \
		$(GPU_PREFIX) \
		ros2 launch agx_bringup rl_corrector_sim.launch.py \
		headless:=$(call lc,$(HEADLESS)) \
		sim_sensors:=$(call lc,$(SIM_SENSORS)) \
		world:=$(WORLD)

rl-train: build
	source /opt/ros/jazzy/setup.bash && \
		source install/setup.bash && \
		$(WORKER_ENV) \
		ros2 run agx_planning rl_corrector_train \
		--timesteps $(TIMESTEPS) \
		$(TERRAIN_FLAG) \
		$(BRIDGE_FLAG) \
		--torch-threads $(TORCH_THREADS) \
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

# ---- RL runtime corrector: 3-phase curriculum -----------------------------
# Each phase is a thin wrapper over `rl-train` (so it inherits the build/source,
# sensors-off, GPU prefix, etc.), chained with --load: easy straights+gentle
# arcs (flat) -> widen to S-bends + harder turns (flat) -> full mix + terrain.
# Bring the sim up first (`make rl-sim` in another terminal; GUI is fine -- the
# bottleneck is per-step gz round-trips, not rendering) then run a phase. The
# obs/action layout is held fixed across phases (config defaults), so --load
# restores cleanly. Override per-phase steps/outputs the usual way, and pass
# TB=~/rl_tb to log all three phases under one TensorBoard tree:
#   make p1 TB=~/rl_tb && make p2 TB=~/rl_tb && make p3 TB=~/rl_tb
# or just `make curriculum TB=~/rl_tb` to run them back-to-back.
P0_OUT   ?= $(HOME)/rl_corrector_p0
P1_OUT   ?= $(HOME)/rl_corrector_p1
P2_OUT   ?= $(HOME)/rl_corrector_p2
P3_OUT   ?= $(HOME)/rl_corrector_p3
P0_STEPS ?= 30000
P1_STEPS ?= 40000
P2_STEPS ?= 60000
P3_STEPS ?= 100000

# Phase 0 -- fast Gazebo-FREE kinematic pretrain (no rl-sim needed). Single
# entrypoint for the throttling-debug baseline: straights + gentle arcs, slow,
# flat, slip-randomized analytic bridge. Logs to TensorBoard if TB is set, e.g.
#   make p0 TB=$(HOME)/rl_tb
# Free the CPU first (stop any sim/heavy desktop apps) -- throughput is CPU-gated.
p0:
	$(MAKE) rl-train BRIDGE=kinematic TERRAIN=false TIMESTEPS=$(P0_STEPS) \
		POLICY_OUT=$(P0_OUT) TB=$(TB) \
		TRAIN_ARGS="--nominal-kinds straight arc --omega-max 0.4 --v-max 0.30 $(TRAIN_ARGS)"

# Phase 1 -- easy: straights + gentle arcs, low speed, flat ground.
p1:
	$(MAKE) rl-train TERRAIN=false TIMESTEPS=$(P1_STEPS) POLICY_OUT=$(P1_OUT) TB=$(TB) \
		TRAIN_ARGS="--nominal-kinds straight arc --omega-max 0.4 --v-max 0.35 $(TRAIN_ARGS)"

# Phase 2 -- widen: add S-bends and harder turns, full speed, still flat.
# Continues from phase 1's policy.
p2:
	$(MAKE) rl-train TERRAIN=false TIMESTEPS=$(P2_STEPS) POLICY_OUT=$(P2_OUT) TB=$(TB) \
		LOAD=$(P1_OUT).zip \
		TRAIN_ARGS="--nominal-kinds straight arc scurve --omega-max 0.8 $(TRAIN_ARGS)"

# Phase 3 -- full difficulty + slip terrain. Continues from phase 2's policy.
p3:
	$(MAKE) rl-train TERRAIN=true TIMESTEPS=$(P3_STEPS) POLICY_OUT=$(P3_OUT) TB=$(TB) \
		LOAD=$(P2_OUT).zip \
		TRAIN_ARGS="--omega-max 1.0 $(TRAIN_ARGS)"

# Run all three phases back-to-back (each loads the previous). Needs the sim up.
curriculum:
	$(MAKE) p1 && $(MAKE) p2 && $(MAKE) p3

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
