from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

THIS_PACKAGE = "agx_bringup"


def cfg_file(name: str) -> PathJoinSubstitution:
    return _find_file("config", name)


def launch_file(name: str) -> PathJoinSubstitution:
    return _find_file("launch", f"{name}.launch.py")


def rviz_file(name: str) -> PathJoinSubstitution:
    return _find_file("rviz", f"{name}.rviz")


def _find_file(dir: str, name: str) -> PathJoinSubstitution:
    return PathJoinSubstitution(
        [
            FindPackageShare(THIS_PACKAGE),
            dir,
            name,
        ]
    )


def supervisor_overrides(context, prefix: str) -> dict:
    """``supervisor_params:="k=v,k=v"`` as ``{prefix + k: v}``, values parsed as
    YAML scalars (``false`` -> bool, ``1000`` -> int). The keys are
    ``agx_planning.supervisor.core.SupervisorConfig`` fields; the prefix is
    where each node loads that dataclass (``supervisor.`` on runtime_corrector,
    ``core.`` on supervisor_node). One launch arg for every layer variant, so a
    new field needs no launch/fixture/harness plumbing."""
    import yaml
    from launch.substitutions import LaunchConfiguration
    spec = LaunchConfiguration("supervisor_params", default="").perform(context).strip()
    out = {}
    for item in filter(None, (t.strip() for t in spec.split(","))):
        k, _, v = item.partition("=")
        if not _:
            raise ValueError(f"supervisor_params item {item!r} is not key=value")
        out[prefix + k.strip()] = yaml.safe_load(v)
    return out
