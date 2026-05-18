# Scout Mini self driving

## Installation

Recursively clone this repository
to ensure all submodule dependencies are cloned as well:

```bash
git clone https://github.com/self-navigation/agx_navigation.git --recursive
```

## Dependencies

This ROS2 workspace is built for ROS2 Jazzy and Gazebo Harmonic.
You can prepare the workspace with:

```bash
make setup
```

This will install the latest versions of ROS2 Jazzy and Gazebo Harmonic
before installing the dependenceis for this project.

If you only want to install the dependencies use:

```bash
make deps
```

## Usage

Run a Gazebo simulation:

```bash
make run SIM=true
```

Run on an Agilex Scout Mini R&D Kit:

```bash
make run SIM=false
```

More options are available with variables exposed in the Makefile,
which map directly to launch parameters.

If you want to build without running:

```bash
make build # or simply make
```
