"""Reinforcement-learning runtime corrector.

A residual policy that adds a per-wheel velocity residual (rad/s) so the robot
holds the PMP planner's frozen trajectory under slip. The pure modules (config,
coeff, obs, reward, nominal, geometry, bridge, kinematic_bridge, terrain)
import neither ROS nor torch and are shared verbatim by training and
deployment. gazebo_bridge, env, policy and train must be imported explicitly;
nothing heavy loads at package import time.
"""
