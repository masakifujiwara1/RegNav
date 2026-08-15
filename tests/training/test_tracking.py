import hashlib

from regnav.training.tracking import checkpoint_manifest, flatten_config, resolve_tracking_uri


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
