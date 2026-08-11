# RECON manifest date extraction design

## Goal

Prevent RECON trajectories from being assigned one artificial date and therefore collapsing into a single train, validation, or test group.

## Interface

`scripts/build_manifest.py` accepts exactly one date source:

- `--date YYYY-MM-DD` preserves the existing fixed-date behavior.
- `--date-from-trajectory` extracts `YYYY-MM-DD` from each trajectory directory name, such as `jackal_2019-10-31-16-57-31_4_r01`.

The two options are mutually exclusive and one remains required. No generic regex configuration is added.

## Data flow and errors

For automatic extraction, the manifest builder finds the first date-shaped segment in each trajectory ID and validates it as a calendar date. The extracted value is written to that row's existing `date` field. A trajectory without a valid date stops generation with an error naming that trajectory; no partial manifest is written.

All other fields and fixed-date output remain unchanged. Existing grouped splitting then separates RECON recording dates without changes to training or evaluation code.

## Verification

One focused test covers extraction from a real RECON-style trajectory ID and rejection of a missing date. Existing manifest and full test suites must remain green. A smoke manifest generated from the ten converted trajectories must contain multiple dates and load through `regnav.data.manifest.load_manifest`.
