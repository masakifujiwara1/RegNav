import hashlib
import importlib
import json
import os
from pathlib import Path
from typing import Mapping


def resolve_tracking_uri(output_dir: Path, explicit_uri: str | None) -> str:
    return explicit_uri or os.environ.get("MLFLOW_TRACKING_URI") or (
        output_dir.parent / "mlruns"
    ).resolve().as_uri()


def flatten_config(prefix: str, values: Mapping[str, object]) -> dict[str, str]:
    return {
        f"{prefix}.{key}": json.dumps(value, sort_keys=True)
        if isinstance(value, (list, tuple, dict))
        else str(value)
        for key, value in values.items()
    }


def checkpoint_manifest(output_dir: Path, final_epoch: int, best_loss: float) -> dict[str, object]:
    manifest: dict[str, object] = {"final_epoch": final_epoch, "best_loss": best_loss}
    for name in ("best.pt", "last.pt"):
        path = output_dir / name
        if not path.is_file():
            manifest[name] = None
            continue
        digest = hashlib.sha256()
        with path.open("rb") as checkpoint:
            for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
                digest.update(chunk)
        manifest[name] = {
            "path": str(path.resolve()),
            "size": path.stat().st_size,
            "sha256": digest.hexdigest(),
        }
    return manifest


class TrainingTracker:
    def __init__(
        self,
        *,
        enabled: bool,
        output_dir: Path,
        model_config: Mapping[str, object],
        training_config: Mapping[str, object],
        model_config_path: Path,
        training_config_path: Path,
        split: str,
        manifest_hash: str,
        device: str,
        compile_enabled: bool,
        git_revision: str | None,
        tracking_uri: str | None,
        experiment: str,
        run_name: str | None,
    ):
        self.enabled = enabled
        self.output_dir = output_dir
        self.model_config = model_config
        self.training_config = training_config
        self.model_config_path = model_config_path
        self.training_config_path = training_config_path
        self.split = split
        self.manifest_hash = manifest_hash
        self.device = device
        self.compile_enabled = compile_enabled
        self.git_revision = git_revision
        self.tracking_uri = tracking_uri
        self.experiment = experiment
        self.run_name = run_name
        self._mlflow = None
        self._file_store_env = None
        self._file_store_env_set = False

    def _restore_file_store_environment(self) -> None:
        if not self._file_store_env_set:
            return
        if self._file_store_env is None:
            os.environ.pop("MLFLOW_ALLOW_FILE_STORE", None)
        else:
            os.environ["MLFLOW_ALLOW_FILE_STORE"] = self._file_store_env
        self._file_store_env_set = False

    def __enter__(self) -> "TrainingTracker":
        if not self.enabled:
            return self

        tracking_uri = resolve_tracking_uri(self.output_dir, self.tracking_uri)
        if tracking_uri.startswith("file:"):
            self._file_store_env = os.environ.get("MLFLOW_ALLOW_FILE_STORE")
            self._file_store_env_set = True
            os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        try:
            self._mlflow = importlib.import_module("mlflow")
            self._mlflow.set_tracking_uri(tracking_uri)
            self._mlflow.set_experiment(self.experiment)
            self._mlflow.start_run(run_name=self.run_name)
            self._mlflow.log_params(
                {
                    **flatten_config("model", self.model_config),
                    **flatten_config("training", self.training_config),
                }
            )
            self._mlflow.set_tags(
                {
                    "split": self.split,
                    "manifest_hash": self.manifest_hash,
                    "device": self.device,
                    "compile_enabled": str(self.compile_enabled),
                    "git_revision": self.git_revision or "unknown",
                }
            )
        except Exception:
            self._restore_file_store_environment()
            raise
        return self

    def log_epoch(self, epoch: int, losses: Mapping[str, float]) -> None:
        if self._mlflow is not None:
            self._mlflow.log_metrics(
                {f"train/{name}": value for name, value in losses.items()},
                step=epoch,
            )

    def log_artifacts(self, final_epoch: int, best_loss: float) -> None:
        if self._mlflow is None:
            return

        manifest_path = self.output_dir / "checkpoint-manifest.json"
        manifest_path.write_text(
            json.dumps(checkpoint_manifest(self.output_dir, final_epoch, best_loss))
        )
        for path in (
            self.output_dir / "metrics.jsonl",
            self.model_config_path,
            self.training_config_path,
            manifest_path,
        ):
            if path.is_file():
                self._mlflow.log_artifact(str(path))

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        try:
            if self._mlflow is not None:
                self._mlflow.end_run(
                    status="FAILED" if exc_type is not None else "FINISHED"
                )
        finally:
            self._restore_file_store_environment()
        return False
