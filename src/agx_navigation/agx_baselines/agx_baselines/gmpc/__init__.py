"""GMPC tracking controller (Tang et al., RA-L 2024) as a comparison baseline.

The controller is the upstream code, vendored as the submodule
``third_party/GMPC-Tracking-Control``. ``tracker.py`` is a ROS-free adapter
that feeds it a PMP plan as the reference trajectory; ``node.py`` exposes it as
a Nav2-style ``follow_path`` action.
"""
