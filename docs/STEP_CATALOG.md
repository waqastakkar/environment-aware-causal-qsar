# Manual pipeline step catalog

- Source of truth for execution order: `scripts/manual/run_all.sh`.
- Logs are written to `outputs/stepX/stepXX_*.log`.
- Each step performs preflight checks and fails with exit code 2 if prerequisites are missing.

## Ordered steps

1. `step01_extract.sh`
2. `step02_postprocess.sh`
3. `step03_assemble_environments.sh`
4. `step04_generate_splits.sh`
5. `step05_benchmark.sh`
6. `step06_train_causal.sh`
7. `step07_counterfactuals.sh`
8. `step08_evaluate_runs.sh`
9. `step08a_prepare_external_inhibition.sh`
10. `step09_cross_endpoint.sh`
11. `step10_interpret.sh`
12. `step11_robustness.sh`
13. `step12a_prepare_library.sh`
14. `step12b_screen_library.sh`
15. `step13_analyze_screening.sh`
16. `step14_match_features.sh`
17. `step15_manuscript.sh`

## Split execution modes

- `training.splits_to_run` unset: run only `training.split_default`.
- `training.splits_to_run=all`: run every split directory under `outputs/step4/`.
- `training.splits_to_run=<comma,list>`: run only those split names.
- Step05/06 publish latest pointers at `outputs/step6/<target>/latest_run.json` and `outputs/step6/<target>/<split>/latest_run.json`.
- Step09/10 resolve those pointers unless `run_dir` is explicitly provided.


## Step 06/09 feature-schema compatibility

- Never reuse an old Step 06 checkpoint after featurization or model-input schema changes.
- Step 06 must produce `checkpoints/best.pt`, `artifacts/feature_schema.json`, and `predictions/test_predictions.parquet` before latest pointers are updated.
- Step 09 reads the selected Step 06 `artifacts/feature_schema.json` and fails early if saved dimensions differ from current featurization code dimensions.
- Step 09 is an external active/inactive inhibition evaluation: it keeps Step 05/06 regression models unchanged and converts `pIC50_hat` to binary calls for external validation (default threshold `5.0`; inhibition active threshold `50%`).

## Smoke mode configuration

- Smoke runs must use `configs/ptp1b_smoke.yaml` or pass `smoke=true` override with `configs/ptp1b.yaml`.
- Baseline behavior remains unchanged in `configs/ptp1b.yaml` (`smoke: false`).
- Typical smoke flow:
  1. `python scripts/smoke/make_tiny_step3_parquet.py`
  2. `bash scripts/manual/step04_generate_splits.sh configs/ptp1b_smoke.yaml`
  3. `bash scripts/manual/step05_benchmark.sh configs/ptp1b_smoke.yaml training.splits_to_run=scaffold_bm`
  4. `bash scripts/manual/step06_train_causal.sh configs/ptp1b_smoke.yaml training.splits_to_run=scaffold_bm`
  5. `bash scripts/manual/step08_evaluate_runs.sh configs/ptp1b_smoke.yaml`


## Step 2 contract (QSAR postprocess)

- Primary dataset: `standard_type=IC50`, `standard_relation=="="`, values normalized to nM (`uM -> nM`), positive numeric values only, and `standard_value <= postprocess.max_value_nM`.
- Primary outputs: `outputs/step2/row_level_primary.csv`, `outputs/step2/compound_level_primary.csv`, `outputs/step2/compound_level_with_properties.csv` (legacy aliases retained).
- Optional secondary endpoint outputs (default Ki) are written separately under `outputs/step2/row_level_secondary_<endpoint>.csv` and are not used by downstream steps unless explicitly selected.
- Diagnostics required: before/after counts by endpoint/relation/units, pIC50 sanity range, unique molecule count, and top drop reasons.

## Step 3 contract (environment assembly)

- Required inputs: `outputs/step2/row_level_with_pIC50.csv`, `outputs/step2/compound_level_with_properties.csv`, `outputs/step1/<target>_qsar_ready.csv`, and BBB rules config.
- Primary outputs: `outputs/step3/multienv_compound_level.parquet`, `outputs/step3/multienv_row_level.parquet`, and environment metadata files (`env_definitions.json`, `env_counts.csv`, `series_assignments.csv`, `env_vector_schema.json`).
- BBB annotation artifact: `outputs/step3/data/bbb_annotations.parquet` with deterministic `molecule_id`-sorted rows and BBB/CNS properties used by downstream metrics.

## Step 8 CNS subset metrics dependency

- `step08_evaluate_runs.sh` passes `outputs/step3/data/bbb_annotations.parquet` (fallback: `outputs/step3/bbb_annotations.parquet`) into evaluation when present.
- CNS subset metrics are computed from this BBB annotation artifact; if it is missing, Step 8 emits a skip warning.
