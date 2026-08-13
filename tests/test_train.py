from pathlib import Path

import pytest
from torch import nn

from regnav.train import _ensure_output_dir_ready, _maybe_compile


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
