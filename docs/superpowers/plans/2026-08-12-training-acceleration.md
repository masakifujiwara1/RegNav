# RegNav Training Acceleration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce RTX 3060 training time without changing RegNav inputs or losses, then train for 15 epochs and continue to 30 only if validation acceptance requires it.

**Architecture:** Measure FP16, BF16, eager, and compiled execution on real RECON batches before changing production code. Add an opt-in compile path and semantics-preserving CUDA input-transfer settings only when they improve steady-state throughput, while saving the original module so checkpoints remain compatible. Use validation—not test—to decide whether epoch 15 is sufficient.

**Tech Stack:** Python, PyTorch 2, CUDA, pytest, uv

## Global Constraints

- Use the uv environment for every Python and test command.
- Do not change model architecture, image size, dataset, loss weights, or acceptance thresholds.
- Keep FP16 unless BF16 is at least 5% faster with finite losses.
- Prefer the lower-VRAM configuration when throughput differs by less than 5%.
- Require at least 1 GiB free VRAM after LoRA is enabled.
- Use the test split exactly once, after selecting the final checkpoint from validation results.
- Every commit message includes `Co-authored-by: Codex <noreply@openai.com>`.

---

### Task 1: Benchmark precision and compilation

**Files:**
- Modify outside Git: `/tmp/regnav_batch_benchmark.py`

**Interfaces:**
- Consumes: real train split from `/home/ubuntu/data/regnav/manifest-recon.jsonl` and `config/model/regnav.yaml`.
- Produces: steady-state samples/second, seconds/step, peak allocated VRAM, and finite-loss status for each candidate.

- [ ] **Step 1: Extend the temporary benchmark with explicit dtype and compile flags**

Add arguments and select the training module and autocast dtype:

```python
parser.add_argument("--amp-dtype", choices=("float16", "bfloat16"), default="float16")
parser.add_argument("--compile", action="store_true")

training_model = (
    torch.compile(model, mode="reduce-overhead") if args.compile else model
)
amp_dtype = getattr(torch, args.amp_dtype)
```

Use `training_model(batch)` inside `torch.autocast("cuda", dtype=amp_dtype)`, enable `GradScaler` only for FP16, and print `finite={torch.isfinite(loss).item()}`.

- [ ] **Step 2: Measure all precision/execution candidates**

Run 20 timed steps after three warm-up steps for:

```bash
uv run python /tmp/regnav_batch_benchmark.py --batch-size 128 --workers 12 --steps 20 --amp-dtype float16
uv run python /tmp/regnav_batch_benchmark.py --batch-size 128 --workers 12 --steps 20 --amp-dtype bfloat16
uv run python /tmp/regnav_batch_benchmark.py --batch-size 128 --workers 12 --steps 20 --amp-dtype float16 --compile
```

Expected: all candidates complete without OOM and report finite loss. If compiled batch 128 leaves less than 1 GiB free, repeat compiled FP16 at batch 96 and 64.

- [ ] **Step 3: Select the candidate deterministically**

Keep BF16 only if it is at least 5% faster than FP16. Keep compilation only if it is at least 5% faster than eager FP16 after warm-up. If two safe batch sizes differ by less than 5%, select the lower-memory batch.

---

### Task 2: Add the proven compile path with compatible checkpoints

**Files:**
- Modify: `src/regnav/train.py`
- Create: `tests/test_train.py`
- Modify: `docs/training.md`

**Interfaces:**
- Consumes: `model: torch.nn.Module` and CLI flag `--compile`.
- Produces: `_maybe_compile(model: nn.Module, enabled: bool) -> nn.Module`; checkpoints continue to store the unwrapped original model.

- [ ] **Step 1: Write the failing compile-selection test**

```python
from torch import nn

from regnav.train import _maybe_compile


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
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `uv run pytest tests/test_train.py -v`

Expected: FAIL because `_maybe_compile` is absent.

- [ ] **Step 3: Implement the minimum opt-in compile path**

Add to `src/regnav/train.py`:

```python
def _maybe_compile(model: torch.nn.Module, enabled: bool) -> torch.nn.Module:
    return torch.compile(model, mode="reduce-overhead") if enabled else model
```

Add `parser.add_argument("--compile", action="store_true")`. Build and retain `model` as today, then set `training_model = _maybe_compile(model, args.compile)` and pass `training_model` only to `train_epoch`. Continue calling `set_training_stage` and `save_checkpoint` with the original `model`.

- [ ] **Step 4: Document the opt-in flag**

Add `--compile` to both RTX 3060 training examples in `docs/training.md` and state that checkpoints remain loadable by evaluation without compilation.

- [ ] **Step 5: Verify focused and full tests**

Run: `uv run pytest tests/test_train.py -v`

Expected: PASS.

Run: `uv run pytest`

Expected: all tests pass.

- [ ] **Step 6: Commit the compile path**

```bash
git add src/regnav/train.py tests/test_train.py docs/training.md
git commit -m "perf: add compiled training path" -m "Co-authored-by: Codex <noreply@openai.com>"
```

If Task 1 found less than 5% compile gain, skip Task 2 entirely rather than carrying an unused option.

---

### Task 3: Optimize CUDA data transfer

**Files:**
- Modify: `src/regnav/train.py`
- Modify: `src/regnav/training/engine.py`
- Modify: `tests/training/test_engine.py`

**Interfaces:**
- Consumes: `device: torch.device`, `training_config.workers`, and a `RegNavBatch`.
- Produces: pinned DataLoader batches and `_to_device` transfers with `non_blocking=True` only on CUDA.

- [ ] **Step 1: Write the failing non-blocking transfer test**

```python
from regnav.training.engine import _to_device


def test_to_device_requests_non_blocking_cuda_transfer():
    class TensorSpy:
        def to(self, device, *, non_blocking):
            self.call = (device, non_blocking)
            return self

    tensor = TensorSpy()
    assert _to_device({"image": tensor}, torch.device("cuda"))["image"] is tensor
    assert tensor.call == (torch.device("cuda"), True)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `uv run pytest tests/training/test_engine.py::test_to_device_requests_non_blocking_cuda_transfer -v`

Expected: FAIL because `_to_device` does not pass `non_blocking`.

- [ ] **Step 3: Implement the minimum transfer changes**

Change `_to_device` to:

```python
return {
    key: value.to(device, non_blocking=device.type == "cuda")
    for key, value in batch.items()
}
```

Move device selection above DataLoader construction in `src/regnav/train.py`, then add:

```python
pin_memory=device.type == "cuda",
persistent_workers=training_config.workers > 0,
```

- [ ] **Step 4: Verify focused and full tests**

Run: `uv run pytest tests/training/test_engine.py -v`

Expected: PASS.

Run: `uv run pytest`

Expected: all tests pass.

- [ ] **Step 5: Re-run the selected Task 1 benchmark with transfer settings**

Apply the same transfer settings to `/tmp/regnav_batch_benchmark.py`:

```python
pin_memory=True,
persistent_workers=args.workers > 0,
```

and transfer each batch with `value.to(device, non_blocking=True)`. Run the selected precision, compile, batch, and worker combination for 20 timed steps. Keep Task 3 only if throughput is at least 5% faster; otherwise revert Task 3 and retain the simpler loader.

- [ ] **Step 6: Commit only a proven transfer improvement**

```bash
git add src/regnav/train.py src/regnav/training/engine.py tests/training/test_engine.py
git commit -m "perf: overlap training data transfers" -m "Co-authored-by: Codex <noreply@openai.com>"
```

---

### Task 4: Smoke-test and train through epoch 15

**Files:**
- Create outside Git: `/tmp/regnav-fast-15.yaml`
- Write outside Git: `/home/ubuntu/data/regnav/runs/regnav/`

**Interfaces:**
- Consumes: selected Task 1 settings and `/home/ubuntu/data/regnav/manifest-recon.jsonl`.
- Produces: `best.pt`, `last.pt`, and 15 JSONL metric rows.

- [ ] **Step 1: Create the 15-epoch training config**

Copy `config/training/default.yaml`, set `epochs: 15`, and set the selected `batch_size` and `workers`. Keep every learning rate and loss weight unchanged.

- [ ] **Step 2: Run a five-step smoke check**

Run the selected benchmark configuration for five post-warm-up steps. Expected: finite loss, no OOM, and at least 1 GiB free VRAM after LoRA is active.

- [ ] **Step 3: Start staged training**

```bash
uv run train-regnav \
  --model-config config/model/regnav.yaml \
  --training-config /tmp/regnav-fast-15.yaml \
  --manifest /home/ubuntu/data/regnav/manifest-recon.jsonl \
  --split train \
  --output-dir /home/ubuntu/data/regnav/runs/regnav \
  --compile
```

Omit `--compile` when Task 1 rejected compilation.

- [ ] **Step 4: Monitor every 30 minutes**

Report process state, completed epochs from `metrics.jsonl`, latest total loss, GPU utilization, VRAM, temperature, and projected completion. Stop on non-finite loss, process exit, or CUDA OOM.

- [ ] **Step 5: Verify epoch-15 artifacts**

Expected: `metrics.jsonl` has exactly 15 rows; `best.pt` and `last.pt` load successfully; the last row contains finite losses.

---

### Task 5: Select duration on validation and evaluate once on test

**Files:**
- Write outside Git: `/home/ubuntu/data/regnav/runs/regnav/validation-epoch15.json`
- Optionally write outside Git: `/home/ubuntu/data/regnav/runs/regnav/validation-epoch30.json`
- Write outside Git: `/home/ubuntu/data/regnav/runs/regnav/test.json`
- Optionally preserve outside Git: `/home/ubuntu/data/regnav/runs/regnav/best-epoch15.pt`
- Write outside Git: `/home/ubuntu/data/regnav/runs/regnav/selected.pt`

**Interfaces:**
- Consumes: epoch-15 checkpoints and `meets_baseline_acceptance` from evaluation.
- Produces: a final selected checkpoint, validation report, test report, and CUDA latency result.

- [ ] **Step 1: Evaluate epoch 15 on validation**

```bash
uv run eval-regnav \
  --checkpoint /home/ubuntu/data/regnav/runs/regnav/best.pt \
  --manifest /home/ubuntu/data/regnav/manifest-recon.jsonl \
  --split validation \
  --output /home/ubuntu/data/regnav/runs/regnav/validation-epoch15.json
```

- [ ] **Step 2: Continue only when validation acceptance is false**

If `aggregate.meets_baseline_acceptance` is false, preserve the checkpoint and derive the 30-epoch config without changing any other setting:

```bash
cp /home/ubuntu/data/regnav/runs/regnav/best.pt /home/ubuntu/data/regnav/runs/regnav/best-epoch15.pt
cp /tmp/regnav-fast-15.yaml /tmp/regnav-fast-30.yaml
sed -i "s/^epochs: 15$/epochs: 30/" /tmp/regnav-fast-30.yaml
```

Then resume optimizer state:

```bash
uv run train-regnav \
  --model-config config/model/regnav.yaml \
  --training-config /tmp/regnav-fast-30.yaml \
  --manifest /home/ubuntu/data/regnav/manifest-recon.jsonl \
  --split train \
  --output-dir /home/ubuntu/data/regnav/runs/regnav \
  --resume /home/ubuntu/data/regnav/runs/regnav/last.pt \
  --compile
```

Omit `--compile` when Task 1 rejected compilation. After epoch 30, evaluate the new best checkpoint:

```bash
uv run eval-regnav \
  --checkpoint /home/ubuntu/data/regnav/runs/regnav/best.pt \
  --manifest /home/ubuntu/data/regnav/manifest-recon.jsonl \
  --split validation \
  --output /home/ubuntu/data/regnav/runs/regnav/validation-epoch30.json
```

Compare `aggregate.model.ade` in the two reports, using FDE as the tie-breaker. Copy the winner explicitly:

```bash
cp /home/ubuntu/data/regnav/runs/regnav/best-epoch15.pt /home/ubuntu/data/regnav/runs/regnav/selected.pt
# Or, when epoch 30 wins:
cp /home/ubuntu/data/regnav/runs/regnav/best.pt /home/ubuntu/data/regnav/runs/regnav/selected.pt
```

If epoch 15 already passed:

```bash
cp /home/ubuntu/data/regnav/runs/regnav/best.pt /home/ubuntu/data/regnav/runs/regnav/selected.pt
```

- [ ] **Step 3: Evaluate the selected checkpoint on test exactly once**

```bash
uv run eval-regnav \
  --checkpoint /home/ubuntu/data/regnav/runs/regnav/selected.pt \
  --manifest /home/ubuntu/data/regnav/manifest-recon.jsonl \
  --split test \
  --output /home/ubuntu/data/regnav/runs/regnav/test.json
```

Expected: report model/baseline metrics and `meets_baseline_acceptance`; collision metrics remain `null` for RECON.

- [ ] **Step 4: Benchmark final CUDA latency**

```bash
uv run python -m regnav.benchmark \
  --checkpoint /home/ubuntu/data/regnav/runs/regnav/selected.pt \
  --device cuda \
  --iterations 1000
```

Expected: report p50, p95, p99, and peak CUDA memory; acceptance requires p95 below 100 ms.
