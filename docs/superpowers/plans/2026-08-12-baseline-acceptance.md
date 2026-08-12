# Baseline Acceptance Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct model acceptance so route-only's construction-guaranteed zero FDE remains reported but does not make acceptance impossible.

**Architecture:** Keep baseline trajectories and all metric calculations unchanged. Replace only the aggregate boolean with an explicit comparison of model ADE/FDE against constant velocity and model ADE against route-only.

**Tech Stack:** Python, PyTorch, pytest, uv

## Global Constraints

- Preserve all existing model, constant-velocity, and route-only metrics.
- Model ADE and FDE must be lower than constant velocity.
- Model ADE must be lower than route-only.
- Route-only FDE must remain reported but must not affect acceptance.
- Remove `beats_both_baselines`; expose `meets_baseline_acceptance`.
- Do not retrain models or change baseline definitions.

---

### Task 1: Correct baseline acceptance and re-evaluate Lite

**Files:**
- Modify: `src/regnav/evaluate.py`
- Modify: `tests/test_evaluate.py`
- Modify: `docs/training.md`

**Interfaces:**
- Consumes: metric dictionaries already produced by `summarize_batch`.
- Produces: `meets_baseline_acceptance: bool` in each aggregate and per-dataset summary.

- [ ] **Step 1: Change the focused test first**

Replace the current acceptance assertion with a case where the model beats constant velocity on ADE/FDE and route-only on ADE while route-only FDE remains zero:

```python
def test_summary_accepts_route_aware_model_without_beating_zero_route_fde():
    route = torch.tensor([[4.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    target = route_only(route, num_poses=8)
    target[:, 1:-1, 1] = 0.5
    prediction = target.clone()
    prediction[:, :-1, 0] += 0.01
    batch = {
        "image": torch.zeros(1, 3, 4, 4),
        "ego": torch.zeros(1, 4),
        "route_goal": route,
        "target_trajectory": target,
        "valid_mask": torch.ones(1, 8, dtype=torch.bool),
    }

    summary = summarize_batch(prediction, batch, interval=0.5)

    assert summary["route_only"]["fde"] == 0.0
    assert summary["meets_baseline_acceptance"] is True
    assert "beats_both_baselines" not in summary
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `uv run pytest tests/test_evaluate.py -v`

Expected: FAIL because `meets_baseline_acceptance` is absent.

- [ ] **Step 3: Implement the minimum comparison**

In `summarize_batch`, replace `beats_both_baselines` with:

```python
result["meets_baseline_acceptance"] = (
    model_metrics["ade"] < result["constant_velocity"]["ade"]
    and model_metrics["fde"] < result["constant_velocity"]["fde"]
    and model_metrics["ade"] < result["route_only"]["ade"]
)
```

- [ ] **Step 4: Verify focused and full tests GREEN**

Run: `uv run pytest tests/test_evaluate.py -v`

Expected: PASS.

Run: `uv run pytest`

Expected: all tests pass.

- [ ] **Step 5: Correct the documented acceptance rule**

In `docs/training.md`, state that acceptance requires ADE/FDE below constant velocity and ADE below route-only. State that route-only FDE is informational because the baseline receives the route goal endpoint.

- [ ] **Step 6: Re-evaluate the existing Lite checkpoint**

Run:

```bash
uv run eval-regnav \
  --checkpoint /home/ubuntu/data/regnav/runs/lite/best.pt \
  --manifest /home/ubuntu/data/regnav/manifest-recon.jsonl \
  --split test \
  --output /home/ubuntu/data/regnav/runs/lite/test.json
```

Expected: the report contains `"meets_baseline_acceptance": true` and no `beats_both_baselines` field.

- [ ] **Step 7: Commit**

```bash
git add src/regnav/evaluate.py tests/test_evaluate.py docs/training.md
git commit -m "fix: make baseline acceptance achievable" -m "Co-authored-by: Codex <noreply@openai.com>"
```
