# Run Artifact Safety Design

## Goal

Prevent accidental retraining into an existing run directory, preserve the verified final artifacts, and record a reproducible Lite-versus-RegNav comparison.

## Training Output Guard

Before loading the manifest, dataset, or model, `train-regnav` checks the output directory for `metrics.jsonl`, `best.pt`, or `last.pt`. If any exists and `--resume` is absent, training exits with an error that names the conflicting paths and tells the operator to choose a new output directory or use `--resume`.

`--resume` keeps the current continuation behavior. Other files, including evaluation JSON, do not block a new training run. The guard does not add a PID lock: the observed failure was a completed run being restarted into the same directory, which a process-only lock would not prevent.

## Artifact Preservation

Create a new immutable-by-convention directory outside Git. Copy only verified artifacts:

- RegNav `selected.pt`
- RegNav `validation-selected.json`
- RegNav `test.json`
- Lite `best.pt`
- Lite `test.json`

Do not copy the contaminated RegNav `best.pt`, `last.pt`, or `metrics.jsonl`. Generate a SHA-256 manifest for every preserved file. Existing destination files are not overwritten; a conflict stops preservation.

## Comparison Report

Create one Markdown report beside the preserved artifacts. It records model ADE/FDE/heading error, baseline acceptance, CUDA p50/p95/p99 latency, checkpoint hash, and provenance notes.

The report must state:

- Lite test: ADE `0.08003418892621994`, FDE `0.017145568504929543`, p95 `4.4414 ms`.
- RegNav test: ADE `0.12026054412126541`, FDE `0.1595885008573532`, p95 `38.1562896509422 ms`.
- Both pass baseline acceptance and the 100 ms p95 latency threshold.
- RECON has no obstacle labels, so privileged collision acceptance is unavailable, not passed.
- RegNav's original `best.pt`, `last.pt`, and `metrics.jsonl` were contaminated by an unexplained second writer and are excluded.

## Verification

- TDD covers a fresh output directory, conflicting training artifacts, unrelated files, and `--resume` bypass.
- The full test suite remains green.
- Every preserved file hash matches its source before the source/destination relationship is recorded.
- JSON checkpoint hashes match the preserved checkpoints.
