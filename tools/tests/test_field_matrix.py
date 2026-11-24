"""Check field-update fixtures and verification summaries."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "tools"), str(REPO / "src/agx_navigation/agx_planning")]

from replanning_baseline.inputs import cell, load_suite
from summarize_replanning import summarize


@pytest.mark.parametrize("filename,count", [
    ("floor6_field_matrix.yaml", 18), ("separated_field_updates.yaml", 4),
])
def test_matrix_edits_free_cells_and_restores_each_case(filename, count):
    profile, snapshots, _ = load_suite(REPO / "config/replanning" / filename)
    assert len(snapshots) == count
    for metadata, occupancy in snapshots:
        if metadata["snapshot_id"] == "base":
            base = occupancy
        goal = cell(metadata["goal"], metadata["geometry"])
        assert occupancy[goal[1], goal[0]] == 0
        changed = occupancy != base
        assert np.all(base[changed] == 0)
        assert np.all(occupancy[changed] == 100)
        if metadata["snapshot_id"] == "removed":
            np.testing.assert_array_equal(occupancy, base)
        elif metadata["snapshot_id"] != "base":
            assert changed.any()
    assert profile["field"]["centring"]
    assert not profile["field"]["early_exit_enable"]


def test_summary_counts_verification_separately_from_sample_outcomes(tmp_path):
    manifest = dict(schema_version=1, profile_sha256="p", environment={},
                    options=dict(repeat=2, warmups=1, field_only=True),
                    snapshots=[dict(case_id="case", snapshot_id="base", input_sha256="i")])
    rows = [dict(schema_version=1, profile_sha256="p", input_sha256="i",
                 snapshot_index=0, repetition=i, warmup=i < 0, status="field_only",
                 solves=[], total_wall_s=1.0,
                 verification=dict(status=status, wall_s=duration))
            for i, status, duration in [(-1, "match", 100), (0, "match", 2), (1, "mismatch", 4)]]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "COMPLETE").touch()
    report = summarize(tmp_path)
    assert report["complete"]
    group = report["groups"][0]
    assert group["field_verification"] == {"match": 1, "mismatch": 1}
    assert group["outcomes"] == {"field_only": 2}
    assert group["metrics"]["verification_wall_s"]["median"] == 3
    assert group["metrics"]["total_wall_s"]["median"] == 1
