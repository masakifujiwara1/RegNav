from pathlib import Path

import pytest
from torch import nn

from regnav.train import (
    _build_parser,
    _checkpoint_metric,
    _ensure_output_dir_ready,
    _maybe_compile,
    _summarize_validation_ade,
)


def _required_train_args():
    return [
        "--model-config", "model.yaml",
        "--training-config", "training.yaml",
        "--manifest", "manifest.jsonl",
        "--split", "train",
        "--output-dir", "run",
    ]


def test_mlflow_is_enabled_by_default():
    args = _build_parser().parse_args(_required_train_args())

    assert args.no_mlflow is False
    assert args.mlflow_experiment == "regnav"


def test_no_mlflow_is_explicit_opt_out():
    args = _build_parser().parse_args(_required_train_args() + ["--no-mlflow"])

    assert args.no_mlflow is True


def test_validation_summary_reports_domains_macro_and_worst():
    summary = _summarize_validation_ade(
        {"scand/jackal": 0.3, "scand/spot": 0.1}
    )

    assert summary == pytest.approx(
        {
            "validation/scand_jackal_ade": 0.3,
            "validation/scand_spot_ade": 0.1,
            "validation/macro_ade": 0.2,
            "validation/worst_ade": 0.3,
        }
    )


def test_checkpoint_metric_prefers_worst_validation_domain():
    assert _checkpoint_metric(
        {"total": 0.1, "validation/worst_ade": 0.3}
    ) == ("validation/worst_ade", 0.3)
    assert _checkpoint_metric({"total": 0.1}) == ("train/total", 0.1)


def test_maybe_compile_uses_reduce_overhead(monkeypatch):
    model = nn.Linear(2, 1)
    compiled = nn.Linear(2, 1)
    called = {}

    def fake_compile(candidate, *, mode):
        called.update(candidate=candidate, mode=mode)
        return compiled

    monkeypatch.setattr("regnav.train.torch.compile", fake_compile)

    assert _maybe_compile(model, False) is model
    assert _maybe_compile(model, True) is compiled
    assert called == {"candidate": model, "mode": "reduce-overhead"}


def test_output_guard_allows_new_and_unrelated_directories(tmp_path):
    _ensure_output_dir_ready(tmp_path / "new", None)
    unrelated = tmp_path / "evaluation.json"
    unrelated.write_text("{}")
    _ensure_output_dir_ready(tmp_path, None)


def test_output_guard_rejects_existing_training_artifacts(tmp_path):
    marker = tmp_path / "best.pt"
    marker.write_bytes(b"checkpoint")

    with pytest.raises(FileExistsError, match="best.pt"):
        _ensure_output_dir_ready(tmp_path, None)


def test_output_guard_allows_resume_with_existing_training_artifacts(tmp_path):
    (tmp_path / "metrics.jsonl").write_text("{}\n")
    _ensure_output_dir_ready(tmp_path, Path("previous.pt"))
