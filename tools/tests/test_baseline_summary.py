import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from summarize_replanning import summarize


def test_summary_keeps_failures_excludes_warmups_and_handles_partial_runs(tmp_path):
    manifest = {"schema_version": 1, "profile_sha256": "profile", "environment": {},
                "options": {"repeat": 2, "warmups": 1, "field_only": False},
                "snapshots": [{"case_id": "route", "snapshot_id": "base", "input_sha256": "input"}]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    rows = [{"schema_version": 1, "profile_sha256": "profile", "input_sha256": "input",
             "snapshot_index": 0, "repetition": i, "warmup": i == -1,
             "status": "completed" if i < 1 else "bvp_failed", "total_wall_s": duration,
             "solves": [{"success": i < 1, "wall_s": duration / 2}]}
            for i, duration in [(-1, 100), (0, 2), (1, 4)]]
    results = tmp_path / "results.jsonl"
    results.write_text("".join(json.dumps(r) + "\n" for r in rows))
    report = summarize(tmp_path)
    assert not report["complete"]
    group = report["groups"][0]
    assert group["outcomes"] == {"completed": 1, "bvp_failed": 1}
    assert group["metrics"]["total_wall_s"]["median"] == 3
    assert group["metrics"]["completed_total_wall_s"]["median"] == 2
    assert group["metrics"]["compute_field_s"]["n"] == 0
    assert group["bvp_failures"] == 1
    (tmp_path / "COMPLETE").touch()
    assert summarize(tmp_path)["complete"]
    results.write_text(results.read_text() + json.dumps(rows[-1]) + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        summarize(tmp_path)
