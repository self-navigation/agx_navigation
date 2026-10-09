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


def config_overrides(context, arg: str, defaults, prefix: str, sub=None) -> dict:
    """Launch arg ``arg:="k=v,k=v"`` as ``{prefix + k: v}``, values parsed as
    YAML scalars (``false`` -> bool, ``1000`` -> int) and cast to the type of
    the field in the dataclass instance ``defaults``; an unknown key raises.
    One launch arg per config dataclass, so a new field needs no
    launch/fixture/harness plumbing. ``sub`` maps a key prefix to another
    dataclass instance whose fields keep that prefix as their parameter
    prefix (``tvlqg.enabled=true`` inside ``tvlqr_params``)."""
    import yaml
    from launch.substitutions import LaunchConfiguration
    spec = LaunchConfiguration(arg, default="").perform(context).strip()
    out = {}
    for item in filter(None, (t.strip() for t in spec.split(","))):
        k, _, v = item.partition("=")
        k = k.strip()
        d, p = defaults, prefix
        for sp, sd in (sub or {}).items():
            if k.startswith(sp):
                d, p, k = sd, sp, k[len(sp):]
        if not _ or not hasattr(d, k):
            raise ValueError(f"{arg} item {item!r}: not key=value with a "
                             f"{type(d).__name__} field as the key")
        # Cast to the default's type: ROS rejects '1000' (INTEGER) for a
        # DOUBLE parameter (job 239 smoke).
        out[p + k] = type(getattr(d, k))(yaml.safe_load(v))
    return out


def supervisor_overrides(context, prefix: str) -> dict:
    """``supervisor_params:="k=v,k=v"``: ``SupervisorConfig`` fields; the prefix
    is where each node loads that dataclass (``supervisor.`` on
    runtime_corrector, ``core.`` on supervisor_node)."""
    from agx_planning.supervisor.core import SupervisorConfig
    return config_overrides(context, "supervisor_params", SupervisorConfig(), prefix)


def tvlqr_overrides(context) -> dict:
    """``tvlqr_params:="k=v,k=v"``: ``TVLQRConfig`` fields on runtime_corrector
    (e.g. ``sideslip_k=-0.33``, track D 2026-10-08); ``tvlqg.``-prefixed keys
    are ``TVLQGConfig`` fields (``tvlqg.enabled=true``, 2026-10-09)."""
    from agx_planning.runtime_corrector.tvlqg import TVLQGConfig
    from agx_planning.runtime_corrector.tvlqr import TVLQRConfig
    return config_overrides(context, "tvlqr_params", TVLQRConfig(), "tvlqr.",
                            sub={"tvlqg.": TVLQGConfig()})
