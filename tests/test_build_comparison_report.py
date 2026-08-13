import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from scripts.build_comparison_report import build_report


def _write_report(
    path: Path,
    *,
    split: str = "test",
    accepted: bool = True,
    checkpoint_hash: str = "abc123",
    ade: float = 0.1,
    fde: float = 0.2,
    heading_error: float = 0.3,
    selected_collision_rate: float | None = None,
    mean_proposal_collision_rate: float | None = None,
) -> Path:
    path.write_text(
        json.dumps(
            {
                "checkpoint_hash": checkpoint_hash,
                "split": split,
                "aggregate": {
                    "meets_baseline_acceptance": accepted,
                    "model": {
                        "ade": ade,
                        "fde": fde,
                        "heading_error": heading_error,
                    },
                    "selected_collision_rate": selected_collision_rate,
                    "mean_proposal_collision_rate": mean_proposal_collision_rate,
                },
            }
        )
        + "\n"
    )
    return path


def test_build_report_writes_markdown_with_metrics_thresholds_and_notes(tmp_path):
    regnav_test = _write_report(
        tmp_path / "regnav.json",
        checkpoint_hash="regnav-hash",
        ade=0.12026054412126541,
        fde=0.1595885008573532,
        heading_error=0.044,
    )
    lite_test = _write_report(
        tmp_path / "lite.json",
        checkpoint_hash="lite-hash",
        ade=0.08003418892621994,
        fde=0.017145568504929543,
        heading_error=0.022,
    )
    output = tmp_path / "comparison.md"

    build_report(
        regnav_test,
        lite_test,
        output,
        {
            "regnav": {"p50": 25.34463250049157, "p95": 38.1562896509422, "p99": 50.93840801928309},
            "lite": {"p50": 3.8236, "p95": 4.4414, "p99": 4.9249},
        },
    )

    report = output.read_text()

    assert "RegNav vs RegNav-Lite Comparison" in report
    assert "0.12026054412126541" in report
    assert "0.017145568504929543" in report
    assert "0.044" in report
    assert "38.1562896509422 ms" in report
    assert "4.4414 ms" in report
    assert "p95 < 100 ms" in report
    assert "regnav-hash" in report
    assert "lite-hash" in report
    assert "unavailable, not passed" in report


@pytest.mark.parametrize(
    ("split", "accepted", "match"),
    [
        ("validation", True, "split"),
        ("test", False, "baseline"),
    ],
)
def test_build_report_rejects_non_test_and_failed_acceptance(tmp_path, split, accepted, match):
    regnav_test = _write_report(tmp_path / "regnav.json", split=split, accepted=accepted)
    lite_test = _write_report(tmp_path / "lite.json")

    with pytest.raises(ValueError, match=match):
        build_report(
            regnav_test,
            lite_test,
            tmp_path / "comparison.md",
            {
                "regnav": {"p50": 1.0, "p95": 2.0, "p99": 3.0},
                "lite": {"p50": 1.0, "p95": 2.0, "p99": 3.0},
            },
        )

def test_build_report_rejects_non_object_json(tmp_path):
    regnav_test = tmp_path / "regnav.json"
    regnav_test.write_text("[]\n")
    lite_test = _write_report(tmp_path / "lite.json")

    with pytest.raises(ValueError, match="object"):
        build_report(
            regnav_test,
            lite_test,
            tmp_path / "comparison.md",
            {
                "regnav": {"p50": 1.0, "p95": 2.0, "p99": 3.0},
                "lite": {"p50": 1.0, "p95": 2.0, "p99": 3.0},
            },
        )

def _valid_latencies():
    return {"regnav": {"p50": 1.0, "p95": 2.0, "p99": 3.0}, "lite": {"p50": 1.0, "p95": 2.0, "p99": 3.0}}


@pytest.mark.parametrize("checkpoint_hash", [None, ""])
def test_build_report_requires_checkpoint_hash(tmp_path, checkpoint_hash):
    regnav_test = _write_report(tmp_path / "regnav.json", checkpoint_hash=checkpoint_hash)
    lite_test = _write_report(tmp_path / "lite.json")

    with pytest.raises(ValueError, match="checkpoint_hash"):
        build_report(regnav_test, lite_test, tmp_path / "comparison.md", _valid_latencies())

def test_build_report_rejects_existing_output(tmp_path):
    regnav_test = _write_report(tmp_path / "regnav.json")
    lite_test = _write_report(tmp_path / "lite.json")
    output = tmp_path / "comparison.md"
    output.write_text("original")

    with pytest.raises(FileExistsError):
        build_report(regnav_test, lite_test, output, _valid_latencies())

    assert output.read_text() == "original"



def test_build_report_does_not_clobber_output_created_after_precheck(tmp_path, monkeypatch):
    regnav_test = _write_report(tmp_path / "regnav.json")
    lite_test = _write_report(tmp_path / "lite.json")
    output = tmp_path / "comparison.md"
    original_open = Path.open
    injected = {"done": False}

    def fake_open(self, *args, **kwargs):
        mode = kwargs.get("mode")
        if mode is None and args:
            mode = args[0]
        if self == output and mode and any(flag in mode for flag in ("w", "x")) and not injected["done"]:
            injected["done"] = True
            self.write_text("original")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fake_open)

    with pytest.raises(FileExistsError):
        build_report(regnav_test, lite_test, output, _valid_latencies())

    assert output.read_text() == "original"

@pytest.mark.parametrize("invalid", [True, float("nan"), float("inf"), -1.0])
def test_build_report_rejects_invalid_latency(tmp_path, invalid):
    regnav_test = _write_report(tmp_path / "regnav.json")
    lite_test = _write_report(tmp_path / "lite.json")
    latencies = _valid_latencies()
    latencies["regnav"]["p95"] = invalid

    with pytest.raises(ValueError, match="latency"):
        build_report(regnav_test, lite_test, tmp_path / "comparison.md", latencies)
