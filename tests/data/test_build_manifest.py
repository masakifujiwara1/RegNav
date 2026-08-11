import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).parents[2]))

import pytest

from scripts.build_manifest import date_from_trajectory_id


def test_date_from_recon_trajectory_id():
    assert date_from_trajectory_id("jackal_2019-10-31-16-57-31_4_r01") == "2019-10-31"


def test_date_from_trajectory_id_rejects_missing_date():
    with pytest.raises(ValueError, match="route_without_date"):
        date_from_trajectory_id("route_without_date")


def test_date_from_trajectory_id_rejects_invalid_date():
    with pytest.raises(ValueError, match="2019-02-30"):
        date_from_trajectory_id("jackal_2019-02-30-12-00-00")


def test_cli_derives_each_manifest_date(tmp_path):
    root = tmp_path / "recon"
    trajectory_ids = [
        "jackal_2019-10-31-16-57-31_4_r01",
        "jackal_2020-01-03-15-35-35_1_r02",
    ]
    for trajectory_id in trajectory_ids:
        trajectory = root / trajectory_id
        trajectory.mkdir(parents=True)
        (trajectory / "traj_data.pkl").touch()
    output = tmp_path / "manifest.jsonl"

    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parents[2] / "scripts" / "build_manifest.py"),
            str(root),
            str(output),
            "--dataset",
            "recon",
            "--robot",
            "jackal",
            "--environment",
            "recon",
            "--date-from-trajectory",
            "--sample-period",
            "0.5",
        ],
        check=True,
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["date"] for row in rows] == ["2019-10-31", "2020-01-03"]
