"""Offline parameter tuning for the runtime correctors.

`simplex` is pure (numpy only, no ROS, no Gazebo) and unit-tested; `tune_tvlqr`
is the driver that wraps a Gazebo replay as its objective.
"""
