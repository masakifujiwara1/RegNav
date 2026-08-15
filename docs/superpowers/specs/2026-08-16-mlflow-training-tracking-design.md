# MLflow Training Tracking Design

**Status:** Approved for implementation on 2026-08-16

## Goal

Make MLflow training monitoring standard for `train-regnav`, using a local file-backed `mlruns` store by default while allowing an explicit remote Tracking Server URI.

## Context

`src/regnav/train.py` owns the training CLI and writes one JSON object per epoch to `metrics.jsonl`. The training loop already returns named loss values from `train_epoch`, so MLflow can observe the same values without changing model or optimizer behavior. MLflow is not currently installed.

The local default must not put run data in Git. Given an output directory such as `/data/regnav/runs/regnav-run`, the default store will be `/data/regnav/runs/mlruns`, shared by sibling runs.

## User-facing behavior

MLflow is enabled by default:

```bash
uv sync
uv run train-regnav \
  --model-config config/model/regnav.yaml \
  --training-config config/training/rtx3060.yaml \
  --manifest /data/regnav/manifest.jsonl \
  --split train \
  --output-dir /data/regnav/runs/regnav-run \
  --compile
```

Disable it only when explicitly requested:

```bash
uv run train-regnav ... --no-mlflow
```

Tracking URI precedence is:

1. `--mlflow-tracking-uri`, when supplied;
2. `MLFLOW_TRACKING_URI`, when set;
3. `file://<output-dir.parent>/mlruns`, by default.

The experiment defaults to `regnav`. `--mlflow-experiment` and `--mlflow-run-name` override the experiment and run name. An omitted run name lets MLflow generate one.

## Data recorded

At run start, record these parameters/tags:

- model configuration fields;
- training configuration fields;
- `split`;
- manifest SHA256;
- git revision, when available;
- selected device (`cpu` or `cuda`);
- whether `torch.compile` is enabled;
- output directory and MLflow package version.

After each completed epoch, log every value returned by `train_epoch` as `train/<name>` with `step=epoch_number`. The existing `metrics.jsonl` append remains the source-of-truth local log.

On successful completion, log these small artifacts:

- `metrics.jsonl`;
- the model and training YAML files;
- `checkpoint-manifest.json` containing the final epoch, best loss, checkpoint paths, and SHA256 hashes when the files exist.

Checkpoint weight files themselves are not uploaded automatically. This avoids duplicating large files and preserves the existing verified-artifact workflow; the manifest points to the externally managed checkpoints.

## Lifecycle and failure behavior

The tracker starts one MLflow run around the existing training loop and closes it in all exit paths. A normal return finishes the run; an exception marks it failed and re-raises the original exception.

When MLflow is enabled, an import, tracking-store, parameter, metric, or artifact error fails the training command rather than silently producing an untracked run. `--no-mlflow` bypasses all MLflow setup and preserves the current training behavior.

No system/GPU polling thread is added in this change. Epoch metrics provide the requested training progress without coupling the trainer to optional NVIDIA monitoring libraries.

## Implementation boundaries

### `pyproject.toml`

Add `mlflow` to the normal project dependencies so the default training command is observable after `uv sync`. Do not add a second optional tracking group.

### `src/regnav/training/tracking.py`

Add a small `TrainingTracker` context manager responsible for:

- URI resolution and `mlflow.set_tracking_uri`;
- experiment/run creation;
- flattening dataclass config values into MLflow-compatible parameter values;
- epoch metric logging;
- final artifact logging;
- success/failure run closure.

The module must not alter model, optimizer, DataLoader, or loss code.

### `src/regnav/train.py`

Add the `--no-mlflow`, `--mlflow-tracking-uri`, `--mlflow-experiment`, and `--mlflow-run-name` flags. Wrap the existing epoch loop with `TrainingTracker`, pass the existing configs and hashes to it, and keep the current checkpoint/JSONL writes unchanged.

### `docs/training.md`

Document the default local store, opt-out flag, remote URI override, and UI command:

```bash
uv run mlflow ui --backend-store-uri /data/regnav/runs/mlruns --port 5000
```

Document that MLflow run data is outside Git alongside the run outputs.

### Tests

Add focused tests for:

- default local URI and explicit/env URI precedence;
- parameter/tag flattening for model and training configs;
- epoch metric names and steps;
- artifact manifest/hash generation;
- success and failure run closure;
- `--no-mlflow` not importing or starting a run.

Tests must use a small fake MLflow module or monkeypatch the tracker boundary; they must not require a running MLflow server or GPU.

## Acceptance criteria

1. A normal `uv run train-regnav` creates a local MLflow run under the default `mlruns` directory and records each completed epoch.
2. `MLFLOW_TRACKING_URI` and the CLI URI override work without code changes.
3. `--no-mlflow` leaves the existing training artifacts and metrics behavior unchanged.
4. Failed training produces a failed MLflow run and preserves the original exception.
5. `uv run pytest` passes, including the new tracker tests.
6. Documentation explains how to open the local UI and how to opt out.

## Non-goals

- Provisioning or operating a remote MLflow server;
- registering models in a Model Registry;
- uploading large checkpoint weights automatically;
- adding GPU utilization polling or a background metrics service;
- changing the model architecture, loss, data split, or checkpoint format.
