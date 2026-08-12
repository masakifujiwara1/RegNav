# Baseline acceptance correction design

## Goal

Make model acceptance achievable and meaningful when the route-only baseline receives the ground-truth route goal and therefore has zero FDE by construction.

## Acceptance rule

Keep every existing model and baseline metric in the evaluation report. Replace the single strict comparison with these explicit requirements:

- Model ADE and FDE must be lower than constant velocity.
- Model ADE must be lower than route-only.
- Route-only FDE remains reported but is not an acceptance criterion.

The report exposes one aggregate `meets_baseline_acceptance` boolean. The old `beats_both_baselines` field is removed because its meaning is mathematically impossible under the existing route-only contract.

## Scope

Change only evaluation acceptance logic, its focused test, and user-facing training documentation. Do not change `route_only`, metric calculations, checkpoints, datasets, or training behavior.

## Verification

A focused test must show acceptance succeeds when the model beats constant velocity on ADE/FDE and route-only on ADE even though route-only FDE is zero. The full test suite and a fresh evaluation of the existing Lite checkpoint must pass without retraining.
