import hashlib
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
