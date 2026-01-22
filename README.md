# Scout Mini self driving

## Installation

This ROS2 workspace is built for ROS2 Jazzy and Gazebo Harmonic.
Ensure they are installed on the system
and that the main ROS2 environment is sourced
prior to attempting to build and run this project.

Install the required dependencies with:

```bash
make install-deps
```

## Usage

Run a Gazebo simulation:

```bash
make sim
```

Run on an Agilex Scout Mini R&D Kit:

```bash
make run
```

Build without running:

```bash
make build # or simply make
```
