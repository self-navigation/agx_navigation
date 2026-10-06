"""Supervisory layer: progress check, per-segment re-timing and wall safety.

The ROS-free core is core.py and walls.py; the ROS adapters are the runtime
corrector (our stack) and node.py (Nav2 and GMPC).
"""

from .core import Decision, SupervisorConfig, SupervisorCore  # noqa: F401
