# RECON Manifest Date Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate RECON manifest rows with each recording date extracted from its trajectory ID while preserving fixed-date behavior.

**Architecture:** Keep all behavior in the existing manifest CLI. Add one standard-library date parser, make the two date sources mutually exclusive in argparse, and exercise the real CLI against the converted RECON smoke data.

**Tech Stack:** Python standard library, pytest, uv

## Global Constraints

- `--date YYYY-MM-DD` must preserve existing fixed-date behavior.
- `--date-from-trajectory` must extract and validate the first `YYYY-MM-DD` segment in each trajectory directory name.
- Exactly one date source is required.
- Invalid or absent trajectory dates must stop before the manifest is written.
- Do not add dependencies or generic regex configuration.

---

### Task 1: Add RECON trajectory date extraction

**Files:**
- Create: `tests/data/test_build_manifest.py`
- Modify: `scripts/build_manifest.py`
- Modify: `docs/data.md`

**Interfaces:**
- Consumes: trajectory directory names and the existing `build_manifest.py ROOT OUTPUT` CLI.
- Produces: `date_from_trajectory_id(trajectory_id: str) -> str` and the `--date-from-trajectory` CLI flag.

- [ ] **Step 1: Write failing parser tests**

```python
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.build_manifest import date_from_trajectory_id


def test_date_from_recon_trajectory_id():
    assert (
        date_from_trajectory_id("jackal_2019-10-31-16-57-31_4_r01")
        == "2019-10-31"
    )


def test_date_from_trajectory_id_rejects_missing_date():
    with pytest.raises(ValueError, match="route_without_date"):
        date_from_trajectory_id("route_without_date")


def test_date_from_trajectory_id_rejects_invalid_date():
    with pytest.raises(ValueError, match="2019-02-30"):
        date_from_trajectory_id("jackal_2019-02-30-12-00-00")


def test_cli_derives_each_manifest_date(tmp_path):
    root = tmp_path / "recon"
    trajectory_ids = [
        "jackal_2019-10-31-16-57-31_4_r01",
        "jackal_2020-01-03-15-35-35_1_r02",
    ]
    for trajectory_id in trajectory_ids:
        trajectory = root / trajectory_id
        trajectory.mkdir(parents=True)
        (trajectory / "traj_data.pkl").touch()
    output = tmp_path / "manifest.jsonl"

    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parents[2] / "scripts" / "build_manifest.py"),
            str(root),
            str(output),
            "--dataset", "recon",
            "--robot", "jackal",
            "--environment", "recon",
            "--date-from-trajectory",
            "--sample-period", "0.5",
        ],
        check=True,
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["date"] for row in rows] == ["2019-10-31", "2020-01-03"]
```

- [ ] **Step 2: Run the parser tests and verify RED**

Run: `uv run pytest tests/data/test_build_manifest.py -v`

Expected: collection fails because `date_from_trajectory_id` is absent.

- [ ] **Step 3: Implement the minimal parser and CLI selection**

Add these imports and parser to `scripts/build_manifest.py`:

```python
from datetime import date
import re


_DATE = re.compile(r"(?<!\d)(\d{4}-\d{2}-\d{2})(?!\d)")


def date_from_trajectory_id(trajectory_id: str) -> str:
    match = _DATE.search(trajectory_id)
    if match is None:
        raise ValueError(f"trajectory {trajectory_id!r} does not contain YYYY-MM-DD")
    value = match.group(1)
    try:
        date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"trajectory {trajectory_id!r} contains invalid date {value!r}") from error
    return value
```

Replace the required fixed date argument with a required mutually-exclusive group:

```python
date_source = parser.add_mutually_exclusive_group(required=True)
date_source.add_argument("--date")
date_source.add_argument("--date-from-trajectory", action="store_true")
```

Resolve every row before `write_text`:

```python
"date": (
    date_from_trajectory_id(trajectory.name)
    if args.date_from_trajectory
    else args.date
),
```

- [ ] **Step 4: Run focused and full tests and verify GREEN**

Run: `uv run pytest tests/data/test_build_manifest.py -v`

Expected: 4 passed.

Run: `uv run pytest`

Expected: all tests pass.

- [ ] **Step 5: Document the real RECON command**

Add this example to `docs/data.md`:

```bash
uv run python scripts/build_manifest.py /home/ubuntu/data/regnav/processed/recon-smoke \
  /home/ubuntu/data/regnav/manifest-smoke.jsonl \
  --dataset recon --robot jackal --environment recon \
  --date-from-trajectory --sample-period 0.5
```

- [ ] **Step 6: Verify the real smoke manifest**

Run the documented command, then:

```bash
uv run python -c "from pathlib import Path; from regnav.data.manifest import load_manifest; rows=load_manifest(Path('/home/ubuntu/data/regnav/manifest-smoke.jsonl')); assert len(rows)==10; assert len({r.date for r in rows})>1; print(f'rows={len(rows)} dates={len({r.date for r in rows})}')"
```

Expected: `rows=10` and at least two dates.

- [ ] **Step 7: Commit**

```bash
git add scripts/build_manifest.py tests/data/test_build_manifest.py docs/data.md
git commit -m "feat: derive RECON manifest dates" -m "Co-authored-by: Codex <noreply@openai.com>"
```
