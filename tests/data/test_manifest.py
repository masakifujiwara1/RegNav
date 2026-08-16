import json

import pytest

from regnav.data.manifest import TrajectoryRecord, group_split, load_manifest, record_domain


def _record(trajectory_id, environment, date):
    return TrajectoryRecord(
        trajectory_id=trajectory_id,
        dataset="recon",
        robot="jackal",
        environment=environment,
        date=date,
        path=f"/data/{trajectory_id}",
        sample_period=0.5,
    )


def test_group_split_keeps_environment_date_together():
    records = [
        _record("a", "park", "2026-01-01"),
        _record("b", "park", "2026-01-01"),
        _record("c", "campus", "2026-01-02"),
    ]

    split = group_split(records, seed=7, ratios=(0.8, 0.1, 0.1))

    assert split["a"] == split["b"]
    assert set(split.values()) <= {"train", "validation", "test"}


def test_manifest_rejects_missing_source_fields(tmp_path):
    path = tmp_path / "manifest.jsonl"
    path.write_text(json.dumps({"trajectory_id": "a"}) + "\n")

    with pytest.raises(ValueError, match="dataset"):
        load_manifest(path)


def test_group_split_rejects_invalid_ratios():
    with pytest.raises(ValueError, match="sum to one"):
        group_split([_record("a", "park", "2026-01-01")], 7, (0.8, 0.2, 0.1))

def test_record_domain_keeps_dataset_and_robot_distinct():
    jackal = TrajectoryRecord(
        trajectory_id="jackal",
        dataset="scand",
        robot="jackal",
        environment="campus",
        date="2021-10-29",
        path="/data/jackal",
        sample_period=0.1,
    )
    spot = TrajectoryRecord(
        trajectory_id="spot",
        dataset="scand",
        robot="spot",
        environment="campus",
        date="2021-11-10",
        path="/data/spot",
        sample_period=0.1,
    )

    assert record_domain(jackal) == "scand/jackal"
    assert record_domain(spot) == "scand/spot"
