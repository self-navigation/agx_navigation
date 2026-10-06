"""GMPC comparison arm (Tang et al., RA-L 2024): a wrapper, not a port.

The controller itself is the upstream code, vendored as the git submodule
``third_party/GMPC-Tracking-Control`` (pinned; see the package README for the
licence situation). ``tracker.py`` is the ROS-free adapter that feeds it our
PMP plan as the reference trajectory, ``node.py`` is the ROS 2 node that
exposes it as a Nav2-style ``follow_path`` action so ``tools/compare_run.py``
can drive it exactly like the ``pmp-mppi`` / ``pmp-rpp`` arms.
"""
