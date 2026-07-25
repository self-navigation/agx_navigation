import os
from setuptools import find_packages, setup

package_name = "agx_planning"

_here = os.path.dirname(os.path.abspath(__file__))
_skfmm_fork = os.path.realpath(os.path.join(_here, "../../../depend/scikit-fmm"))

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=[
        "setuptools",
        f"scikit-fmm @ file://{_skfmm_fork}",
        "numpy",
        "scipy",
        # RL runtime corrector. gymnasium builds the training env; torch is
        # required on-robot for policy inference; stable-baselines3 for training.
        "gymnasium",
        "stable-baselines3",
        "torch",
    ],
    zip_safe=True,
    maintainer="agilex",
    maintainer_email="you@example.com",
    description="TODO: Package description",
    license="MIT",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            f"{name} = {package_name}.{name}:main"
            for name in [
                "frontier_explorer",
                "vector_field",
                "pmp_planner",
                "runtime_corrector",
                "calibrator",
                "slip_ident",
                "run_recorder",
            ]
        ]
        # Submodule entrypoints that don't follow the {package}.{name}:main shape.
        + [
            f"rl_corrector_train = {package_name}.rl_corrector.train:main",
        ],
    },
)
