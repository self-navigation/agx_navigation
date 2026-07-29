"""Reinforcement-learning runtime corrector.

A residual policy that adds a per-wheel velocity residual (rad/s) so the robot
stays on the PMP planner's frozen trajectory under terrain-induced slip.

This package is split so the math is shared verbatim between training and
deployment, and so the pure logic imports neither ROS nor torch:

  config   -- RLCorrectorConfig dataclass (also the kinematics single source)
  coeff    -- action -> additive per-wheel residual -> clamped wheel commands (pure)
  obs      -- observation construction in the path-relative frame (pure)
  reward   -- reward + termination/success predicates (pure)
  nominal  -- frozen reference-trajectory generation (pure)
  geometry -- arclength / projection helpers (pure)
  bridge          -- StateReading + the reset/step/close contract the env calls
  kinematic_bridge -- pure no-slip backend (tests + the env correctness anchor)
  gazebo_bridge   -- live Gazebo backend (gz.transport ground-truth + rclpy I/O)
  terrain         -- domain-randomization patches (reuses surface_patches builder)
  env / policy / train -- ROS/torch-dependent, imported explicitly

Nothing heavy is imported at package import time on purpose.
"""
