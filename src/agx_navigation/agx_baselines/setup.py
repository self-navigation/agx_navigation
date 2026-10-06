import os
from glob import glob

from setuptools import find_packages, setup

package_name = "agx_baselines"

# Upstream GMPC sources (git submodule) are installed verbatim under
# share/agx_baselines/third_party/gmpc so the node can put that directory on
# sys.path at runtime; only the pure-python parts the controller imports.
_GMPC = os.path.join("third_party", "GMPC-Tracking-Control")
_gmpc_files = [
    (os.path.join("share", package_name, "third_party", "gmpc", sub), glob(os.path.join(_GMPC, sub, "*.py")))
    for sub in ("controller", "planner", "utils")
]

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test", "third_party", "third_party.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        *[(d, f) for d, f in _gmpc_files if f],
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="gmatiukhin",
    maintainer_email="contact@gmatiukhin.site",
    description="Comparison baselines; not part of the agx navigation stack",
    license="MIT",
    entry_points={"console_scripts": [
        "gmpc_controller = agx_baselines.gmpc.node:main",
    ]},
)
