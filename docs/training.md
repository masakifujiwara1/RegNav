# Training and evaluation

## Setup

RegNav uses `uv`; do not install project dependencies with `pip`.

```bash
uv sync --group test
uv run python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'
```

Keep public datasets, manifests, runs, and checkpoints outside Git. Follow [data.md](data.md) for source licenses, provenance, checksums, and the processed trajectory format.

## RTX 3060 commands

```bash
uv run train-regnav \
  --model-config config/model/regnav_lite.yaml \
  --training-config config/training/default.yaml \
  --manifest /data/regnav/manifest.jsonl \
  --split train \
  --output-dir /data/regnav/runs/lite

uv run train-regnav \
  --model-config config/model/regnav.yaml \
  --training-config config/training/default.yaml \
  --manifest /data/regnav/manifest.jsonl \
  --split train \
  --output-dir /data/regnav/runs/regnav

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

- RegNav ADE and FDE below constant-velocity and route-only baselines on the held-out split.
- Selected-proposal privileged collision rate below the mean proposal collision rate.
- Batch-1 p95 latency below 100 ms for both models.

Retain evaluation JSON and benchmark output beside the checkpoints, outside Git.
