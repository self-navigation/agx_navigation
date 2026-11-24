"""Freeze inputs once using the selected head's canonical input loader."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--field-config", action="append", default=[])
    parser.add_argument("--planner-config", action="append", default=[])
    args = parser.parse_args()
    repo, out = args.repo.resolve(), args.out.resolve()
    sys.path[:0] = [str(repo / "tools"), str(repo / "src/agx_navigation/agx_planning")]
    import numpy as np
    from replanning_baseline.inputs import load_suite, digest

    out.mkdir(parents=True, exist_ok=False)
    index = []
    for experiment, paths in (("field", args.field_config), ("planner", args.planner_config)):
        for number, path in enumerate(paths):
            source = (repo / path).resolve()
            if not source.is_relative_to(repo):
                raise ValueError("config must be a committed path inside the selected head")
            tracked = subprocess.run(["git", "-C", str(repo), "cat-file", "-e",
                                      "HEAD:" + source.relative_to(repo).as_posix()],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if tracked.returncode:
                raise ValueError("config must be a committed path inside the selected head")
            profile, snapshots, metadata = load_suite(source)
            if profile["field"]["early_exit_enable"] or profile["field_path"] != "packed_float32":
                raise ValueError("matched collection requires complete fields and packed_float32")
            suite = f"{experiment}-{number:02d}-{source.stem}"
            dest = out / suite
            dest.mkdir()
            document = {"schema_version": 1, "suite": suite, "experiment": experiment,
                        "profile": profile, "profile_sha256": digest(profile),
                        "source": metadata, "snapshots": [entry for entry, _ in snapshots]}
            (dest / "manifest.json").write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
            np.savez_compressed(dest / "inputs.npz", **{f"map_{i}": value for i, (_, value) in enumerate(snapshots)})
            shutil.copyfile(source, dest / "scenario.yaml")
            document["input_file_sha256"] = hashlib.sha256((dest / "inputs.npz").read_bytes()).hexdigest()
            index.append({"suite": suite, "experiment": experiment, "directory": suite,
                          "profile_sha256": document["profile_sha256"],
                          "input_file_sha256": document["input_file_sha256"],
                          "snapshots": document["snapshots"]})
    (out / "index.json").write_text(json.dumps(index, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"suites": len(index), "snapshots": sum(len(item["snapshots"]) for item in index)}))


if __name__ == "__main__":
    main()
