import hashlib
import os

import pytest

from regnav.training.tracking import (
    TrainingTracker,
    checkpoint_manifest,
    flatten_config,
    resolve_tracking_uri,
)


class FakeMlflow:
    def __init__(self):
        self.set_tracking_uri_calls = []
        self.set_experiment_calls = []
        self.start_run_calls = []
        self.log_params_calls = []
        self.set_tags_calls = []
        self.log_metrics_calls = []
        self.log_artifact_calls = []
        self.end_run_calls = []

    def set_tracking_uri(self, uri):
        self.set_tracking_uri_calls.append(uri)

    def set_experiment(self, experiment):
        self.set_experiment_calls.append(experiment)

    def start_run(self, **kwargs):
        self.start_run_calls.append(kwargs)

    def log_params(self, params):
        self.log_params_calls.append(params)

    def set_tags(self, tags):
        self.set_tags_calls.append(tags)

    def log_metrics(self, metrics, step):
        self.log_metrics_calls.append((metrics, step))

    def log_artifact(self, path):
        self.log_artifact_calls.append(path)

    def end_run(self, **kwargs):
        self.end_run_calls.append(kwargs)


def _tracker(tmp_path, enabled=True):
    return TrainingTracker(
        enabled=enabled,
        output_dir=tmp_path / "run",
        model_config={"d_model": 16},
        training_config={"epochs": 2},
        model_config_path=tmp_path / "model.yaml",
        training_config_path=tmp_path / "training.yaml",
        split="train",
        manifest_hash="manifest-hash",
        device="cpu",
        compile_enabled=False,
        git_revision="git-revision",
        tracking_uri="file:///tmp/mlruns",
        experiment="regnav",
        run_name="test-run",
    )


def test_resolve_tracking_uri_prefers_explicit_then_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://tracking.example")
    assert resolve_tracking_uri(tmp_path / "run", "sqlite:///explicit") == "sqlite:///explicit"
    assert resolve_tracking_uri(tmp_path / "run", None) == "http://tracking.example"


def test_resolve_tracking_uri_defaults_to_sibling_mlruns(monkeypatch, tmp_path):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert resolve_tracking_uri(tmp_path / "runs" / "run", None) == (
        tmp_path / "runs" / "mlruns"
    ).resolve().as_uri()


def test_flatten_config_serializes_nested_values():
    assert flatten_config("model", {"image_size": (448, 252), "d_model": 256}) == {
        "model.image_size": "[448, 252]",
        "model.d_model": "256",
    }


def test_checkpoint_manifest_describes_present_checkpoints_and_ignores_missing(tmp_path):
    best = tmp_path / "best.pt"
    best.write_bytes(b"best")
    manifest = checkpoint_manifest(tmp_path, final_epoch=4, best_loss=0.25)

    assert manifest["final_epoch"] == 4
    assert manifest["best_loss"] == 0.25
    assert manifest["best.pt"] == {
        "path": str(best.resolve()),
        "size": 4,
        "sha256": hashlib.sha256(b"best").hexdigest(),
    }
    assert manifest["last.pt"] is None


def test_training_tracker_runs_lifecycle_and_logs_epoch(monkeypatch, tmp_path):
    fake = FakeMlflow()
    monkeypatch.setattr(
        "regnav.training.tracking.importlib.import_module", lambda name: fake
    )

    with _tracker(tmp_path) as tracker:
        tracker.log_epoch(3, {"total": 0.25, "trajectory": 0.1})

    assert fake.set_tracking_uri_calls == ["file:///tmp/mlruns"]
    assert fake.set_experiment_calls == ["regnav"]
    assert fake.start_run_calls == [{"run_name": "test-run"}]
    assert fake.log_params_calls == [
        {"model.d_model": "16", "training.epochs": "2"}
    ]
    assert fake.set_tags_calls == [
        {
            "split": "train",
            "manifest_hash": "manifest-hash",
            "device": "cpu",
            "compile_enabled": "False",
            "git_revision": "git-revision",
        }
    ]
    assert fake.log_metrics_calls == [
        ({"train/total": 0.25, "train/trajectory": 0.1}, 3)
    ]
    assert fake.end_run_calls == [{"status": "FINISHED"}]


def test_training_tracker_restores_unset_file_store_environment(monkeypatch, tmp_path):
    fake = FakeMlflow()
    monkeypatch.delenv("MLFLOW_ALLOW_FILE_STORE", raising=False)
    monkeypatch.setattr(
        "regnav.training.tracking.importlib.import_module", lambda name: fake
    )

    with _tracker(tmp_path):
        assert os.environ["MLFLOW_ALLOW_FILE_STORE"] == "true"

    assert "MLFLOW_ALLOW_FILE_STORE" not in os.environ


def test_training_tracker_restores_existing_file_store_environment(monkeypatch, tmp_path):
    fake = FakeMlflow()
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "preserve-me")
    monkeypatch.setattr(
        "regnav.training.tracking.importlib.import_module", lambda name: fake
    )

    with _tracker(tmp_path):
        assert os.environ["MLFLOW_ALLOW_FILE_STORE"] == "true"

    assert os.environ["MLFLOW_ALLOW_FILE_STORE"] == "preserve-me"


def test_training_tracker_restores_file_store_environment_when_enter_fails(
    monkeypatch, tmp_path
):
    fake = FakeMlflow()
    fake.set_experiment = lambda experiment: (_ for _ in ()).throw(RuntimeError("boom"))
    monkeypatch.delenv("MLFLOW_ALLOW_FILE_STORE", raising=False)
    monkeypatch.setattr(
        "regnav.training.tracking.importlib.import_module", lambda name: fake
    )

    with pytest.raises(RuntimeError, match="boom"):
        _tracker(tmp_path).__enter__()

    assert "MLFLOW_ALLOW_FILE_STORE" not in os.environ


def test_training_tracker_leaves_file_store_disabled_for_external_uri(
    monkeypatch, tmp_path
):
    fake = FakeMlflow()
    tracker = _tracker(tmp_path)
    tracker.tracking_uri = "https://tracking.example"
    monkeypatch.delenv("MLFLOW_ALLOW_FILE_STORE", raising=False)
    monkeypatch.setattr(
        "regnav.training.tracking.importlib.import_module", lambda name: fake
    )

    with tracker:
        pass

    assert "MLFLOW_ALLOW_FILE_STORE" not in os.environ


def test_local_tracker_does_not_leak_file_store_setting_to_external_tracker(
    monkeypatch, tmp_path
):
    fake = FakeMlflow()
    monkeypatch.delenv("MLFLOW_ALLOW_FILE_STORE", raising=False)
    monkeypatch.setattr(
        "regnav.training.tracking.importlib.import_module", lambda name: fake
    )

    with _tracker(tmp_path):
        pass
    external_tracker = _tracker(tmp_path)
    external_tracker.tracking_uri = "https://tracking.example"
    with external_tracker:
        pass

    assert "MLFLOW_ALLOW_FILE_STORE" not in os.environ


def test_training_tracker_marks_reraised_errors_as_failed(monkeypatch, tmp_path):
    fake = FakeMlflow()
    monkeypatch.setattr(
        "regnav.training.tracking.importlib.import_module", lambda name: fake
    )

    with pytest.raises(RuntimeError, match="boom"):
        with _tracker(tmp_path):
            raise RuntimeError("boom")

    assert fake.end_run_calls == [{"status": "FAILED"}]


def test_training_tracker_preserves_training_error_when_end_run_fails(
    monkeypatch, tmp_path
):
    fake = FakeMlflow()

    def fail_end_run(**kwargs):
        fake.end_run_calls.append(kwargs)
        raise RuntimeError("end run boom")

    fake.end_run = fail_end_run
    monkeypatch.setattr(
        "regnav.training.tracking.importlib.import_module", lambda name: fake
    )

    with pytest.raises(RuntimeError, match="training boom"):
        with _tracker(tmp_path):
            raise RuntimeError("training boom")

    assert fake.end_run_calls == [{"status": "FAILED"}]


def test_training_tracker_closes_started_run_after_setup_failure(
    monkeypatch, tmp_path
):
    fake = FakeMlflow()

    def fail_log_params(params):
        fake.log_params_calls.append(params)
        raise RuntimeError("setup boom")

    def fail_end_run(**kwargs):
        fake.end_run_calls.append(kwargs)
        raise RuntimeError("end run boom")

    fake.log_params = fail_log_params
    fake.end_run = fail_end_run
    monkeypatch.delenv("MLFLOW_ALLOW_FILE_STORE", raising=False)
    monkeypatch.setattr(
        "regnav.training.tracking.importlib.import_module", lambda name: fake
    )

    with pytest.raises(RuntimeError, match="setup boom"):
        _tracker(tmp_path).__enter__()

    assert fake.start_run_calls == [{"run_name": "test-run"}]
    assert fake.end_run_calls == [{"status": "FAILED"}]
    assert "MLFLOW_ALLOW_FILE_STORE" not in os.environ


def test_disabled_training_tracker_does_not_import_or_call_mlflow(monkeypatch, tmp_path):
    imports = []
    fake = FakeMlflow()
    monkeypatch.delenv("MLFLOW_ALLOW_FILE_STORE", raising=False)
    monkeypatch.setattr(
        "regnav.training.tracking.importlib.import_module",
        lambda name: imports.append(name) or fake,
    )

    with _tracker(tmp_path, enabled=False) as tracker:
        tracker.log_epoch(3, {"total": 0.25})
        tracker.log_artifacts(3, 0.25)

    assert imports == []
    assert all(not value for value in vars(fake).values())
    assert "MLFLOW_ALLOW_FILE_STORE" not in os.environ


def test_local_tracker_does_not_leak_file_store_setting_to_disabled_tracker(
    monkeypatch, tmp_path
):
    fake = FakeMlflow()
    monkeypatch.delenv("MLFLOW_ALLOW_FILE_STORE", raising=False)
    monkeypatch.setattr(
        "regnav.training.tracking.importlib.import_module", lambda name: fake
    )

    with _tracker(tmp_path):
        pass
    with _tracker(tmp_path, enabled=False):
        pass

    assert "MLFLOW_ALLOW_FILE_STORE" not in os.environ


def test_training_tracker_logs_present_artifacts(monkeypatch, tmp_path):
    fake = FakeMlflow()
    monkeypatch.setattr(
        "regnav.training.tracking.importlib.import_module", lambda name: fake
    )
    tracker = _tracker(tmp_path)
    tracker.output_dir.mkdir()
    tracker.model_config_path.write_text("d_model: 16\n")
    tracker.training_config_path.write_text("epochs: 2\n")
    (tracker.output_dir / "metrics.jsonl").write_text('{"total": 0.25}\n')

    with tracker:
        tracker.log_artifacts(final_epoch=2, best_loss=0.25)

    assert fake.log_artifact_calls == [
        str(tracker.output_dir / "metrics.jsonl"),
        str(tracker.model_config_path),
        str(tracker.training_config_path),
        str(tracker.output_dir / "checkpoint-manifest.json"),
    ]
