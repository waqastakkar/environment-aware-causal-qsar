# Pipeline changelog

## 2026-03-06

- Step12 was split into Step12a (`scripts/manual/step12a_prepare_library.sh`) and Step12b (`scripts/manual/step12b_screen_library.sh`).
- Added `scripts/prepare_library.py` for explicit parse/clean/deduplicate preprocessing outputs.
- Step12b now consumes prepared `library_dedup.parquet` and no longer auto-generates demo libraries.
- Screening config now supports CSV-first input settings (`input_path`, `input_format: csv`, `sep`, column names).

## 2026-03-05

- Manual pipeline now uses explicit run pointers (`outputs/step5/run_pointer.json`, `outputs/step6/run_pointer.json`) instead of implicit checkpoint discovery in downstream steps.
- Added `scripts/manual/step08a_prepare_external_inhibition.sh` to materialize canonical external inhibition data at:
  `data/external/processed/ptp1b_inhibition_chembl335/data/inhibition_external_final.parquet`.
- Step09 now requires and consumes the canonical external inhibition parquet path above.
- Added preflight checks (missing files/columns) in manual scripts with clear error messages and exit code `2`.
- Step12 manual wrapper now resolves run pointer + screening inputs explicitly and can generate a tiny demo library for smoke tests.
- Added `scripts/pipeline_doctor.py` for end-to-end contract verification.
- Training now writes `artifacts/feature_schema.json` so screening featurization is schema-safe.
- Step04 now writes `outputs/step4/splits_manifest.json` enumerating generated split folders and required files per split.
- Step05/Step06 now support split mode selection via `training.splits_to_run` (unset => `training.split_default`, `all`, or explicit list).
- Step05/Step06 now enforce per-run contracts (`checkpoints/best.pt`, `artifacts/feature_schema.json`, `predictions/test_predictions.parquet`) and write split-aware latest pointers:
  - `outputs/step6/<target>/latest_run.json`
  - `outputs/step6/<target>/<split_name>/latest_run.json`
- Step08/Step10/Step11 manual wrappers now accept explicit `run_dir` or `runs_root` overrides.
- Step09 pointer resolution now prefers split-specific latest pointers when `run_dir` is not provided.
- `scripts/pipeline_doctor.py` now validates step contracts, split manifests, pointer files, and required screening/evaluation artifacts.
