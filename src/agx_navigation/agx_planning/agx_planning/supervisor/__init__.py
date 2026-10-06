"""Supervisory layer (#40): progress check, per-segment re-timing, wall safety.

ROS-free core (``core.py``, ``walls.py``); the ROS adapters are the runtime
corrector (our stack) and ``node.py`` (Nav2 / GMPC). Design and decisions:
docs/supervisor-plan.md.
"""

from .core import Decision, SupervisorConfig, SupervisorCore  # noqa: F401
