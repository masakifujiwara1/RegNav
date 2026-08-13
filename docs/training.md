# Training and evaluation

## Setup

RegNav uses `uv`; do not install project dependencies with `pip`.

```bash
uv sync --group test
uv run python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'
```

Keep public datasets, manifests, runs, and checkpoints outside Git. Follow [data.md](data.md) for source licenses, provenance, checksums, and the processed trajectory format.

Training refuses to overwrite a run directory containing `metrics.jsonl`, `best.pt`, or `last.pt`; use `--resume` or choose a new output directory. An evaluation JSON file alone does not block training.

## RTX 3060 commands

```bash
uv run train-regnav \
  --model-config config/model/regnav_lite.yaml \
  --training-config config/training/default.yaml \
  --manifest /data/regnav/manifest.jsonl \
  --split train \
  --output-dir /data/regnav/runs/lite \
  --compile

uv run train-regnav \
  --model-config config/model/regnav.yaml \
  --training-config config/training/default.yaml \
  --manifest /data/regnav/manifest.jsonl \
  --split train \
  --output-dir /data/regnav/runs/regnav \
  --compile

uv run eval-regnav \
  --checkpoint /data/regnav/runs/regnav/best.pt \
  --manifest /data/regnav/manifest.jsonl \
  --split test \
  --output /data/regnav/runs/regnav/test.json

uv run python -m regnav.benchmark \
  --checkpoint /data/regnav/runs/regnav/best.pt \
  --device cuda \
  --iterations 1000
```

Run the benchmark for both RegNav and RegNav-Lite checkpoints. Acceptance requires:

The `--compile` flag is opt-in for training; checkpoints remain loadable by evaluation without compilation.

- RegNav ADE and FDE below the constant-velocity baseline on the held-out split.
- RegNav ADE below the route-only baseline. Route-only FDE is informational because the baseline receives the exact target endpoint.
- Selected-proposal privileged collision rate below the mean proposal collision rate.
- Batch-1 p95 latency below 100 ms for both models.

Retain evaluation JSON and benchmark output beside the checkpoints, outside Git.

## Preserve and compare a verified run

Preserve only the verified checkpoints and test reports in a new directory. The command refuses a missing source or an existing destination and writes `SHA256SUMS.json`:

```bash
uv run python scripts/preserve_run_artifacts.py \
  --destination /data/regnav/preserved/2026-08-13-regnav-lite \
  --regnav-selected /data/regnav/runs/regnav/selected.pt \
  --regnav-validation /data/regnav/runs/regnav/validation-selected.json \
  --regnav-test /data/regnav/runs/regnav/test.json \
  --lite-best /data/regnav/runs/lite/best.pt \
  --lite-test /data/regnav/runs/lite/test.json
```

Build the Markdown comparison from the preserved test JSON and measured CUDA batch-1 latency (`p50,p95,p99`, in milliseconds):

```bash
uv run python scripts/build_comparison_report.py \
  --regnav-test /data/regnav/preserved/2026-08-13-regnav-lite/regnav/test.json \
  --lite-test /data/regnav/preserved/2026-08-13-regnav-lite/lite/test.json \
  --output /data/regnav/comparison-2026-08-13.md \
  --regnav-latency 25.34463250049157,38.1562896509422,50.93840801928309 \
  --lite-latency 3.8236,4.4414,4.9249
```

The comparison requires both reports to be `split: test` and to pass baseline acceptance. RECON collision metrics are reported as unavailable, not passed when obstacle labels are absent.
