# MLflow Training Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make MLflow tracking standard for `train-regnav`, with a local file-backed `mlruns` store by default, epoch metrics and run metadata recorded, and `--no-mlflow` as an explicit opt-out.

**Architecture:** Keep the existing model, DataLoader, loss, checkpoint, and JSONL code unchanged. Add a lazy-importing `TrainingTracker` context manager that owns MLflow lifecycle, URI resolution, parameter/metric logging, and small final artifacts; wire it around the existing epoch loop in `train.py`.

**Tech Stack:** Python 3.10–3.13, `uv`, MLflow 3.x, pytest, existing PyTorch training stack.

## Global Constraints

- MLflow is a normal project dependency and is enabled by default.
- `--no-mlflow` must preserve the current training behavior and avoid importing/starting MLflow.
- Tracking URI precedence is CLI `--mlflow-tracking-uri`, then `MLFLOW_TRACKING_URI`, then `file://<output-dir.parent>/mlruns`.
- Default experiment name is `regnav`; CLI flags override experiment and run name.
- Existing `metrics.jsonl`, `best.pt`, and `last.pt` writes remain unchanged.
- Checkpoint weights are not uploaded as MLflow artifacts; only small metadata and logs are uploaded.
- MLflow failures when enabled fail the command and preserve the original exception on training errors.
- Use `uv` for dependency, test, and verification commands; do not use `pip`.
- Every commit message must contain `Co-authored-by: Codex <noreply@openai.com>`.

## File Map

- Modify: `pyproject.toml` — add the MLflow dependency.
- Modify: `uv.lock` — resolve the dependency with `uv lock`.
- Create: `src/regnav/training/tracking.py` — URI resolution, parameter flattening, lifecycle, metrics, and artifact manifest.
- Modify: `src/regnav/train.py` — CLI flags and tracker lifecycle around the existing loop.
- Create: `tests/training/test_tracking.py` — tracker unit tests with a fake MLflow boundary.
- Modify: `tests/test_train.py` — parser/default/opt-out coverage.
- Modify: `docs/training.md` — local UI, default behavior, remote override, and opt-out instructions.
- Modify: `.gitignore` — ignore a root-level local `mlruns/` store when users run from the repository.

## Task 1: Add the tracking helper contract and URI/parameter tests

**Files:**
- Create: `tests/training/test_tracking.py`
- Create: `src/regnav/training/tracking.py`

**Interfaces:**
- Produces `resolve_tracking_uri(output_dir: Path, explicit_uri: str | None) -> str`.
- Produces `flatten_config(prefix: str, values: Mapping[str, object]) -> dict[str, str]`.
- Produces `checkpoint_manifest(output_dir: Path, final_epoch: int, best_loss: float) -> dict[str, object]`.

- [ ] **Step 1: Write the failing URI and flattening tests**

```python
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
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `uv run pytest tests/training/test_tracking.py -q`

Expected: FAIL because `regnav.training.tracking` and its functions do not exist.

- [ ] **Step 3: Implement the minimal pure helpers**

```python
def resolve_tracking_uri(output_dir: Path, explicit_uri: str | None) -> str:
    return explicit_uri or os.environ.get("MLFLOW_TRACKING_URI") or (
        output_dir.parent / "mlruns"
    ).resolve().as_uri()


def flatten_config(prefix: str, values: Mapping[str, object]) -> dict[str, str]:
    return {
        f"{prefix}.{key}": json.dumps(value, sort_keys=True) if isinstance(value, (list, tuple, dict)) else str(value)
        for key, value in values.items()
    }
```

`checkpoint_manifest` must inspect only `best.pt` and `last.pt`, include their resolved path, size, SHA256 when present, and never raise for a missing checkpoint.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run: `uv run pytest tests/training/test_tracking.py -q`

Expected: PASS for the URI, environment precedence, default path, and serialization behavior.

- [ ] **Step 5: Commit the helper contract**

```bash
git add tests/training/test_tracking.py src/regnav/training/tracking.py
git commit -m "feat: add MLflow tracking helpers" -m "Co-authored-by: Codex <noreply@openai.com>"
```

## Task 2: Implement and test the MLflow run lifecycle

**Files:**
- Modify: `tests/training/test_tracking.py`
- Modify: `src/regnav/training/tracking.py`

**Interfaces:**
- `TrainingTracker.__enter__() -> TrainingTracker` starts one run when enabled.
- `TrainingTracker.log_epoch(epoch: int, losses: Mapping[str, float]) -> None` logs `train/<name>` metrics with the supplied step.
- `TrainingTracker.log_artifacts(final_epoch: int, best_loss: float) -> None` logs `metrics.jsonl`, both YAML files, and `checkpoint-manifest.json` when present.
- `TrainingTracker.__exit__(exc_type, exc_value, traceback) -> bool` ends the run with `FINISHED` or `FAILED` and returns `False`.
- Its constructor accepts keyword-only `enabled`, `output_dir`, `model_config`, `training_config`, `model_config_path`, `training_config_path`, `split`, `manifest_hash`, `device`, `compile_enabled`, `git_revision`, `tracking_uri`, `experiment`, and `run_name` values.

- [ ] **Step 1: Add a fake-MLflow lifecycle test**

The fake module must record calls to `set_tracking_uri`, `set_experiment`, `start_run`, `log_params`, `set_tags`, `log_metrics`, `log_artifact`, and `end_run`. Define this test fixture before the assertion:

```python
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


with _tracker(tmp_path) as tracker:
    tracker.log_epoch(3, {"total": 0.25, "trajectory": 0.1})

assert fake.log_metrics_calls == [({"train/total": 0.25, "train/trajectory": 0.1}, 3)]
assert fake.end_run_calls == [{"status": "FINISHED"}]
```

- [ ] **Step 2: Run the lifecycle test and verify RED**

Run: `uv run pytest tests/training/test_tracking.py -q`

Expected: FAIL because `TrainingTracker` does not yet start/log/end a run.

- [ ] **Step 3: Implement the minimal tracker**

Use `importlib.import_module("mlflow")` only inside the enabled path. On entry, resolve the URI, call `set_tracking_uri`, `set_experiment`, `start_run`, then log flattened model/training params and tags. `log_epoch` prefixes every loss with `train/` and calls `log_metrics({"train/<name>": value}, step=epoch)`. `__exit__` calls `end_run(status="FAILED")` if an exception occurred, otherwise `end_run(status="FINISHED")`, and never suppresses exceptions.

- [ ] **Step 4: Add failure and opt-out boundary tests**

Assert that an exception inside the context ends the run as `FAILED` and is re-raised. Assert that `_tracker(tmp_path, enabled=False)` does not import MLflow and does not call any fake MLflow method.

- [ ] **Step 5: Run tracker tests and verify GREEN**

Run: `uv run pytest tests/training/test_tracking.py -q`

Expected: all tracker tests PASS without a running server or CUDA.

- [ ] **Step 6: Commit the lifecycle**

```bash
git add tests/training/test_tracking.py src/regnav/training/tracking.py
git commit -m "feat: track training epochs with MLflow" -m "Co-authored-by: Codex <noreply@openai.com>"
```

## Task 3: Wire the default-enabled tracker into the training CLI

**Files:**
- Modify: `src/regnav/train.py`
- Modify: `tests/test_train.py`

**Interfaces:**
- Add parser flags `--no-mlflow`, `--mlflow-tracking-uri`, `--mlflow-experiment` (default `regnav`), and `--mlflow-run-name`.
- Pass the existing `model_config`, `training_config`, split, manifest hash, device, compile flag, config paths, output directory, and git revision to `TrainingTracker`.

- [ ] **Step 1: Add parser tests before wiring**

Extract `_build_parser() -> argparse.ArgumentParser` if needed and add the required-argument helper plus these assertions:

```python
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
```

- [ ] **Step 2: Run parser tests and verify RED**

Run: `uv run pytest tests/test_train.py -q`

Expected: FAIL because the new parser flags and helper do not exist.

- [ ] **Step 3: Add the flags and wrap the existing loop**

Keep `_ensure_output_dir_ready`, dataset construction, optimizer setup, checkpoint writes, and `metrics.jsonl` writes unchanged. Create a `TrainingTracker` before the epoch loop, call `tracker.log_epoch(epoch + 1, losses)` immediately after the local JSONL append, and call `tracker.log_artifacts(final_epoch=training_config.epochs, best_loss=best)` after the final checkpoint write. The context must cover the loop so errors become failed MLflow runs.

- [ ] **Step 4: Run parser and tracker tests and verify GREEN**

Run: `uv run pytest tests/test_train.py tests/training/test_tracking.py -q`

Expected: PASS with no regression in output-directory guard tests.

- [ ] **Step 5: Commit CLI wiring**

```bash
git add src/regnav/train.py tests/test_train.py
git commit -m "feat: enable MLflow tracking by default" -m "Co-authored-by: Codex <noreply@openai.com>"
```

## Task 4: Add the dependency, local-store ignore rule, and user documentation

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `.gitignore`
- Modify: `docs/training.md`

- [ ] **Step 1: Add the dependency and ignore rule**

Add `"mlflow>=3,<4"` to `[project].dependencies` and add a root-level `mlruns/` ignore entry. Keep all run outputs outside Git in documented commands.

- [ ] **Step 2: Resolve with uv and verify the package**

Run: `uv lock && uv sync`

Expected: lockfile updates successfully and `uv run python -c "import mlflow; print(mlflow.__version__)"` prints a 3.x version.

- [ ] **Step 3: Document the default and override commands**

Add to `docs/training.md`:

```bash
# Default local tracking store: <output-dir parent>/mlruns
uv run train-regnav --model-config config/model/regnav.yaml --training-config config/training/rtx3060.yaml --manifest /data/regnav/manifest.jsonl --split train --output-dir /data/regnav/runs/regnav-run --compile

# Open the local UI
uv run mlflow ui --backend-store-uri /data/regnav/runs/mlruns --port 5000

# Explicit remote store
MLFLOW_TRACKING_URI=http://localhost:5000 \
  uv run train-regnav --model-config config/model/regnav.yaml --training-config config/training/rtx3060.yaml --manifest /data/regnav/manifest.jsonl --split train --output-dir /data/regnav/runs/regnav-run --compile

# Explicit opt-out
uv run train-regnav --model-config config/model/regnav.yaml --training-config config/training/rtx3060.yaml --manifest /data/regnav/manifest.jsonl --split train --output-dir /data/regnav/runs/regnav-run --compile --no-mlflow
```

Explain that params are logged at run start, loss metrics after every epoch, and small logs/config/checkpoint metadata at successful completion.

- [ ] **Step 4: Run documentation/config checks**

Run: `uv run pytest tests/test_config.py -q` and `git diff --check`.

Expected: PASS with no whitespace errors.

- [ ] **Step 5: Commit dependency and docs**

```bash
git add pyproject.toml uv.lock .gitignore docs/training.md
git commit -m "docs: document MLflow training monitoring" -m "Co-authored-by: Codex <noreply@openai.com>"
```

## Task 5: Full verification and local tracking smoke test

**Files:** No source changes expected; use a temporary directory outside the repository.

- [ ] **Step 1: Run the complete test suite**

Run: `uv run pytest`

Expected: all existing tests plus new tracking tests PASS.

- [ ] **Step 2: Exercise a real local MLflow run without the large dataset**

Run this CPU-only smoke with the installed MLflow package; it uses temporary YAML/log files and no production data:

```bash
uv run python - <<'PY'
from pathlib import Path
from tempfile import TemporaryDirectory
from regnav.training.tracking import TrainingTracker

with TemporaryDirectory() as raw:
    root = Path(raw)
    output = root / "run"
    output.mkdir()
    (root / "model.yaml").write_text("name: regnav\\n")
    (root / "training.yaml").write_text("epochs: 1\\n")
    (output / "metrics.jsonl").write_text('{"epoch": 1, "total": 0.25}\\n')
    with TrainingTracker(
        enabled=True, output_dir=output, model_config={"d_model": 16},
        training_config={"epochs": 1}, model_config_path=root / "model.yaml",
        training_config_path=root / "training.yaml", split="train",
        manifest_hash="smoke", device="cpu", compile_enabled=False,
        git_revision=None, tracking_uri=(root / "mlruns").resolve().as_uri(),
        experiment="smoke", run_name="local-smoke",
    ) as tracker:
        tracker.log_epoch(1, {"total": 0.25})
        tracker.log_artifacts(1, 0.25)
    assert any((root / "mlruns").rglob("meta.yaml"))
PY
```

Expected: one local experiment/run exists and contains the `train/total` metric plus the JSONL artifact.

- [ ] **Step 3: Verify opt-out behavior**

Run the same tracker construction with `enabled=False`, then assert `not (root / "mlruns").exists()` and that the existing `metrics.jsonl` remains unchanged. The test must not import MLflow in this path.

- [ ] **Step 4: Inspect the final diff**

Run: `git status --short --branch`, `git diff --check`, and `git log --oneline -8`.

Expected: only the intended MLflow changes plus the pre-existing untracked progress HTML remain; no generated `mlruns` data is tracked.

- [ ] **Step 5: Commit any final test-only adjustment**

If verification requires a code correction, repeat its focused red-green test cycle and commit with:

```bash
git commit -m "fix: finalize MLflow tracking verification" -m "Co-authored-by: Codex <noreply@openai.com>"
```

