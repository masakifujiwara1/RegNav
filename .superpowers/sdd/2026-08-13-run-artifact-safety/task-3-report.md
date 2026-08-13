Task 3 report

- RED: focused test initially failed during collection because the comparison module was absent.
- GREEN: implemented `build_report` and the CLI with split/acceptance validation, exact model metrics, latency fields, p95 threshold status, checkpoint hashes, and collision-unavailable note.
- Documentation: added preservation and comparison commands to `docs/training.md`.
- Focused tests: `uv run pytest tests/test_build_comparison_report.py -v` — 3 passed.
- Full tests: `uv run pytest` — 56 passed.
- `git diff --check` passed.
- Implementation was completed in the parent turn after the initial implementer stopped during the edit phase.

Fix round 1/5: added RED tests for malformed JSON roots, missing/empty hashes, existing output, and invalid latency values.
- GREEN focused tests: `uv run pytest tests/test_build_comparison_report.py -v` — 11 passed.
- GREEN full tests: `uv run pytest` — 64 passed.
- Fix commit: `70da93c`.
- Review findings addressed: 3 Important and 1 Minor; scoped re-review pending.


Fix round 1/5 (final review TOCTOU): replaced `output.exists()` + write with exclusive create via `Path.open('x')`.
- RED: `test_build_report_does_not_clobber_output_created_after_precheck` simulates the output file appearing after the precheck and confirmed the old flow could overwrite it.
- GREEN focused tests: `uv run pytest tests/test_preserve_run_artifacts.py tests/test_build_comparison_report.py` — 18 passed.
- GREEN full tests: `uv run pytest` — 66 passed.
- note: report publication stays single-file and atomic at create time; if the path already appears, the original contents are preserved by `FileExistsError`.
