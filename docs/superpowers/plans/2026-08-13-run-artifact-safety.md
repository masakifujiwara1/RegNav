# Run Artifact Safety Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent accidental retraining into an existing run, preserve verified checkpoints and reports, and produce a reproducible Lite-versus-RegNav comparison.

**Architecture:** Add one pure output-directory guard to `regnav.train` and call it before config/data/model work. Add two stdlib-only scripts: one atomically copies named artifacts and writes a SHA-256 manifest, and one validates evaluation JSON then renders a Markdown comparison from supplied latency values. Keep run-specific data outside Git.

**Tech Stack:** Python 3.13, pathlib, hashlib, json, shutil, argparse, pytest, uv

## Global Constraints

- Use uv for every Python and test command.
- Reject only existing `metrics.jsonl`, `best.pt`, or `last.pt` when `--resume` is absent.
- Allow `--resume` and unrelated files such as evaluation JSON.
- Never overwrite an existing preservation destination.
- Preserve only verified `selected.pt`, `validation-selected.json`, `test.json`, Lite `best.pt`, and Lite `test.json`.
- Do not preserve contaminated RegNav `best.pt`, `last.pt`, or `metrics.jsonl`.
- Generate SHA-256 for every preserved file before recording the source/destination relationship.
- Report RECON collision metrics as unavailable, not passed.
- Every commit message includes `Co-authored-by: Codex <noreply@openai.com>`.

---

### Task 1: Guard training output directories

**Files:**
- Modify: `src/regnav/train.py`
- Test: `tests/test_train.py`
- Modify: `docs/training.md`

**Interfaces:**
- Consumes: `output_dir: Path` and `resume: Path | None`.
- Produces: `_ensure_output_dir_ready(output_dir, resume) -> None`, raising `FileExistsError` only for conflicting training artifacts.

- [ ] **Step 1: Add failing guard tests**

Append these tests to `tests/test_train.py`:

```python
from pathlib import Path

import pytest

from regnav.train import _ensure_output_dir_ready


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
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `uv run pytest tests/test_train.py -v`

Expected: FAIL with `ImportError` because `_ensure_output_dir_ready` is absent.

- [ ] **Step 3: Implement the minimum guard and call it before setup**

Add:

```python
_TRAINING_ARTIFACTS = ("metrics.jsonl", "best.pt", "last.pt")


def _ensure_output_dir_ready(output_dir: Path, resume: Path | None) -> None:
    if resume is not None:
        return
    conflicts = [output_dir / name for name in _TRAINING_ARTIFACTS if (output_dir / name).exists()]
    if conflicts:
        paths = ", ".join(str(path) for path in conflicts)
        raise FileExistsError(f"output directory already contains training artifacts: {paths}; use --resume or choose a new directory")
```

Call `_ensure_output_dir_ready(args.output_dir, args.resume)` immediately after `args = parser.parse_args()` and before `load_yaml_config`.

- [ ] **Step 4: Run focused and full tests**

Run: `uv run pytest tests/test_train.py -v` (expected all guard and compile tests pass).

Run: `uv run pytest` (expected all tests pass).

- [ ] **Step 5: Document the guard**

In `docs/training.md`, state that a run directory containing `metrics.jsonl`, `best.pt`, or `last.pt` requires `--resume` or a new output directory; evaluation JSON alone does not block training.

- [ ] **Step 6: Commit**

```bash
git add src/regnav/train.py tests/test_train.py docs/training.md
git commit -m "fix: guard existing training runs" -m "Co-authored-by: Codex <noreply@openai.com>"
```

---

### Task 2: Preserve verified artifacts with hashes

**Files:**
- Create: `scripts/preserve_run_artifacts.py`
- Create: `tests/test_preserve_run_artifacts.py`

**Interfaces:**
- Consumes: `sources: dict[str, Path]` and `destination: Path`.
- Produces: `preserve_artifacts(sources, destination) -> dict[str, str]` and `SHA256SUMS.json` in the destination.

- [ ] **Step 1: Write failing preservation tests**

Create tests covering byte-for-byte copy, hashes, missing source, and destination collision:

```python
from pathlib import Path

import pytest

from scripts.preserve_run_artifacts import preserve_artifacts


def test_preserve_artifacts_copies_and_hashes_sources(tmp_path):
    sources = {"regnav/selected.pt": tmp_path / "selected.pt"}
    sources["regnav/selected.pt"].write_bytes(b"weights")

    hashes = preserve_artifacts(sources, tmp_path / "preserved")

    assert hashes["regnav/selected.pt"]
    assert (tmp_path / "preserved/regnav/selected.pt").read_bytes() == b"weights"
    assert (tmp_path / "preserved/SHA256SUMS.json").exists()


def test_preserve_artifacts_rejects_missing_source_and_existing_destination(tmp_path):
    with pytest.raises(FileNotFoundError):
        preserve_artifacts({"missing.pt": tmp_path / "missing.pt"}, tmp_path / "new")

    destination = tmp_path / "existing"
    destination.mkdir()
    source = tmp_path / "x.pt"
    source.write_bytes(b"x")
    with pytest.raises(FileExistsError):
        preserve_artifacts({"x.pt": source}, destination)
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `uv run pytest tests/test_preserve_run_artifacts.py -v`

Expected: FAIL with `ModuleNotFoundError` because the script is absent.

- [ ] **Step 3: Implement atomic copy and hash manifest**

Use only `Path`, `hashlib.sha256`, `shutil.copy2`, `tempfile.TemporaryDirectory`, `os.replace`, and `json`. Validate every source before creating the temporary directory. Copy each source to its declared relative path, hash the copied bytes, write sorted `SHA256SUMS.json`, then atomically rename the temporary directory to `destination`. Refuse an existing destination before any copy.

- [ ] **Step 4: Add CLI arguments for the five verified files**

The CLI must require `--destination`, `--regnav-selected`, `--regnav-validation`, `--regnav-test`, `--lite-best`, and `--lite-test`; map them to `regnav/selected.pt`, `regnav/validation-selected.json`, `regnav/test.json`, `lite/best.pt`, and `lite/test.json` respectively.

- [ ] **Step 5: Run focused and full tests**

Run: `uv run pytest tests/test_preserve_run_artifacts.py -v`.

Run: `uv run pytest`.

- [ ] **Step 6: Commit**

```bash
git add scripts/preserve_run_artifacts.py tests/test_preserve_run_artifacts.py
git commit -m "feat: preserve verified run artifacts" -m "Co-authored-by: Codex <noreply@openai.com>"
```

---

### Task 3: Generate comparison report

**Files:**
- Create: `scripts/build_comparison_report.py`
- Create: `tests/test_build_comparison_report.py`
- Modify: `docs/training.md`

**Interfaces:**
- Consumes: Lite/RegNav test JSON paths and six latency floats.
- Produces: a Markdown report with model metrics, acceptance, checkpoint hashes, p95 threshold, and collision availability note.

- [ ] **Step 1: Write failing report tests**

Test that a valid pair produces Markdown containing ADE/FDE/p95 and that a non-test split or failed acceptance raises `ValueError`.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `uv run pytest tests/test_build_comparison_report.py -v`.

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement report generation**

Implement `build_report(regnav_test: Path, lite_test: Path, output: Path, latencies: dict[str, dict[str, float]]) -> None`. Require both JSON files to have `split == "test"` and `aggregate.meets_baseline_acceptance is True`; render exact model ADE/FDE/heading values, p50/p95/p99, checkpoint hashes, `p95 < 100 ms`, and the RECON collision-unavailable note. Do not infer collision acceptance from `null`.

- [ ] **Step 4: Run focused and full tests**

Run: `uv run pytest tests/test_build_comparison_report.py -v`.

Run: `uv run pytest`.

- [ ] **Step 5: Document operational commands and commit**

Document the guard, preservation command, and comparison command in `docs/training.md`, then:

```bash
git add scripts/build_comparison_report.py tests/test_build_comparison_report.py docs/training.md
git commit -m "docs: add run comparison workflow" -m "Co-authored-by: Codex <noreply@openai.com>"
```

---

### Task 4: Preserve current artifacts and write the comparison

**Files:**
- Write outside Git: `/home/ubuntu/data/regnav/preserved/2026-08-13-regnav-lite/`
- Write outside Git: `/home/ubuntu/data/regnav/comparison-2026-08-13.md`

- [ ] **Step 1: Check destination is absent**

Run: `test ! -e /home/ubuntu/data/regnav/preserved/2026-08-13-regnav-lite`.

- [ ] **Step 2: Preserve only the five verified sources**

```bash
uv run python scripts/preserve_run_artifacts.py \
  --destination /home/ubuntu/data/regnav/preserved/2026-08-13-regnav-lite \
  --regnav-selected /home/ubuntu/data/regnav/runs/regnav/selected.pt \
  --regnav-validation /home/ubuntu/data/regnav/runs/regnav/validation-selected.json \
  --regnav-test /home/ubuntu/data/regnav/runs/regnav/test.json \
  --lite-best /home/ubuntu/data/regnav/runs/lite/best.pt \
  --lite-test /home/ubuntu/data/regnav/runs/lite/test.json
```

- [ ] **Step 3: Generate the comparison report**

```bash
uv run python scripts/build_comparison_report.py \
  --regnav-test /home/ubuntu/data/regnav/preserved/2026-08-13-regnav-lite/regnav/test.json \
  --lite-test /home/ubuntu/data/regnav/preserved/2026-08-13-regnav-lite/lite/test.json \
  --output /home/ubuntu/data/regnav/comparison-2026-08-13.md \
  --regnav-latency 25.34463250049157,38.1562896509422,50.93840801928309 \
  --lite-latency 3.8236,4.4414,4.9249
```

- [ ] **Step 4: Verify preservation and comparison**

Check all five destination files exist, `SHA256SUMS.json` hashes match, report contains both acceptance values and p95 values, and no contaminated RegNav artifact appears in the destination tree.
