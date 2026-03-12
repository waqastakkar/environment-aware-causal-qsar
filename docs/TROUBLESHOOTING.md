# Troubleshooting (manual mode)

## Step exits with code 2

The step failed a preflight contract check. Read the error text and step log under `outputs/stepX/`.

## Missing run pointer

Run step06 (or step05 fallback). Required files:
- `outputs/step6/run_pointer.json`
- `outputs/step6/<target>/<split>/<run_id>/checkpoints/best.pt`

## Step09 missing external inhibition parquet

Run:

```bash
bash scripts/manual/step08a_prepare_external_inhibition.sh configs/ptp1b.yaml
```

Expected output:
`data/external/processed/ptp1b_inhibition_chembl335/data/inhibition_external_final.parquet`

## Step12b screening fails due to missing prepared library

Run Step12a first so this file exists:
`outputs/step12/latest_prepare.json` and `<prepared_run>/processed/library_dedup.parquet`.

## Step12 screening fails due to feature schema

Use a run produced by updated step06. It must contain:
`<run_dir>/artifacts/feature_schema.json`.

## Quick diagnosis

```bash
python scripts/pipeline_doctor.py configs/ptp1b.yaml
```
