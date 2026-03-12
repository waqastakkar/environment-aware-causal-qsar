# AGENTS.md

## Repository expectations

- Treat scripts/manual/*.sh as the canonical execution path.
- Every step must define a stable I/O contract:
  - required input files and required columns
  - guaranteed output files and locations
- Never select a run directory by "first checkpoint found" when multiple splits exist.
  - Use pointer files (latest_run.json) or explicit run_dir arguments.
- Multi-split mode must never overwrite results.
  - Store by split name under outputs/step5 and outputs/step6.
- Step 09 must be preceded by an explicit external inhibition preparation step that generates:
  data/external/processed/ptp1b_inhibition_chembl335/data/inhibition_external_final.parquet
- Screening must require artifacts/feature_schema.json to exist in the chosen run_dir.
- All steps must fail with clear error messages when prerequisites are missing.
- Never reuse Step 06 checkpoints after feature-schema or model-input changes.
- Require a fresh Step 06 run before Step 09.
