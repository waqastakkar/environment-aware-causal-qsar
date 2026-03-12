# PTP1B Causal QSAR + Screening Pipeline

**Tagline:** Reproducible, causality-aware QSAR modeling and virtual screening for PTP1B lead discovery.

This repository implements an end-to-end pipeline for **causal QSAR development**, robust model evaluation, and **large-scale screening** around the PTP1B target (e.g., CHEMBL335). It addresses a common scientific challenge in medicinal chemistry: models that perform well in-distribution but fail under assay/domain shifts. By combining multi-environment data assembly, invariance-driven diagnostics, counterfactual analysis, interpretability, and manuscript-ready reporting, the project supports both practical screening and publication-grade reproducibility.

Compared with many conventional QSAR workflows, this pipeline emphasizes:

- explicit environment-aware splitting and validation,
- provenance-first execution (run manifests, hashes, environment capture),
- integrated manuscript pack generation,
- explicit manual step scripts for transparent step-by-step execution (`scripts/manual/*.sh`).

## Table of Contents

- [PTP1B Causal QSAR + Screening Pipeline](#ptp1b-causal-qsar--screening-pipeline)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Usage](#usage)
    - [Manual step-by-step execution (recommended)](#manual-step-by-step-execution-recommended)
    - [Manual step commands (copy/paste)](#manual-step-commands-copypaste)
    - [Step catalog (manual mode)](#step-catalog-manual-mode)
    - [CLI status (deprecated)](#cli-status-deprecated)
  - [Output folders and file explanations](#output-folders-and-file-explanations)
  - [Reproducibility](#reproducibility)
  - [Example end-to-end workflow](#example-end-to-end-workflow)
  - [Troubleshooting](#troubleshooting)
  - [Contributing](#contributing)
  - [Citation](#citation)
  - [License](#license)
  - [Contact](#contact)

## Features

- Explicit manual orchestration scripts for single-step, range, and full pipeline execution.
- Stepwise pipeline for extraction, processing, environment construction, training, evaluation, and screening.
- Causal-QSAR orientation: environment shift diagnostics and robustness analysis.
- Counterfactual generation for molecule-level what-if analysis.
- Interpretability stage for model explanation assets.
- Virtual screening + post-screening analysis + feature matching.
- Publication-oriented report/manuscript artifact generation.
- Reproducibility artifacts per run: resolved config, executed steps, logs, errors, provenance, environment snapshots.

## Prerequisites

| Requirement | Recommendation |
|---|---|
| Python | 3.10+ |
| Core stack | RDKit, PyTorch 2.x, NumPy/Pandas/SciPy/Scikit-learn |
| OS | Linux/macOS (Linux recommended for GPU workflows) |
| Hardware | 16+ GB RAM recommended; NVIDIA GPU optional for acceleration |

Environment setup options:

1. **Recommended:** Conda environment via `environment.yml` in this repository.
2. **Supplemental:** pip packages from `requirements.txt` for pip-only workflows.

For full platform notes and command-level setup, see [`INSTALL.md`](INSTALL.md).

## Installation

1. Create and activate Conda environment (recommended):

```bash
conda env create -f environment.yml
conda activate ptp1b-causal-qsar
```

2. Install the project package in editable mode:

```bash
python -m pip install -e .
```

3. Confirm CLI availability:

```bash
ptp1bqsar --help
```

For additional installation details and step-specific notes, see [`INSTALL.md`](INSTALL.md).

## Configuration

Use `configs/ptp1b.yaml` as your baseline.

Example template:

```yaml
paper_id: ptp1b_causal_qsar_v1
target: CHEMBL335

paths:
  chembl_sqlite: data/raw/chembl_36.db
  data_root: data
  outputs_root: outputs

style:
  svg_only: true
  font: Times New Roman
  bold_text: true
  palette: nature5
  font_title: 16
  font_label: 14
  font_tick: 12
  font_legend: 12

training:
  task: regression
  label_col: pIC50
  seeds: [1, 2, 3, 4, 5]
  split_default: scaffold_bm
  # unset => only split_default; "all" => every outputs/step4/<split>; or comma-list
  splits_to_run: all

robustness:
  ensemble_size: 5
  conformal_coverage: 0.90
  ad_threshold: 0.35

screening:
  input_format: smi
  smi_layout: smiles_id
  header: auto
  smiles_col_name: smiles
  id_col_name: zinc_id
  cns_mpo_threshold: 4.0
  topk: 500
```

### Main configuration keys

| Key | Purpose |
|---|---|
| `paper_id` | Identifier used for manuscript/release artifacts |
| `target` | Target entity (e.g., CHEMBL335 for PTP1B) |
| `paths.*` | Input/output root paths, including ChEMBL SQLite location |
| `style.*` | Consistent figure typography and palette parameters |
| `training.*` | Task type, labels, seeds, default split strategy, and optional `splits_to_run` selector (`all`/list/unset) |
| `robustness.*` | Ensemble/conformal/applicability domain controls |
| `screening.*` | Input schema and prioritization settings for virtual screening |

## Usage

### Manual step-by-step execution (recommended)

> The unified CLI is now **deprecated for day-to-day execution**. Prefer the explicit bash scripts in `scripts/manual/`.

1. Ensure scripts are executable:

```bash
chmod +x scripts/manual/*.sh
```

2. Optionally pin interpreter used by all manual scripts:

```bash
export PIPELINE_PYTHON=/path/to/python
```

3. Run a single step:

```bash
bash scripts/manual/step01_extract.sh configs/ptp1b.yaml
```

4. Run all steps in sequence:

```bash
bash scripts/manual/run_all.sh configs/ptp1b.yaml
```

5. Run a subset of steps (range via env or arg):

```bash
STEPS=1-10 bash scripts/manual/run_all.sh configs/ptp1b.yaml
bash scripts/manual/run_all.sh configs/ptp1b.yaml 5-8
```

6. Pass extra per-step overrides (converted from `KEY=VALUE` to `--KEY VALUE`):

```bash
bash scripts/manual/step05_benchmark.sh configs/ptp1b.yaml training.splits_to_run=all
bash scripts/manual/step06_train_causal.sh configs/ptp1b.yaml training.splits_to_run=random,scaffold_bm training.epochs=1
bash scripts/manual/step07_counterfactuals.sh configs/ptp1b.yaml cns_mpo_threshold=4.5
```

### Smoke mode (fast pre-merge checks)

Smoke mode is **opt-in** and should use either:

- `configs/ptp1b_smoke.yaml`, or
- `configs/ptp1b.yaml` with override `smoke=true`.

Default `configs/ptp1b.yaml` behavior is unchanged (`smoke: false`).

Example smoke run commands:

```bash
python scripts/smoke/make_tiny_step3_parquet.py
bash scripts/manual/step04_generate_splits.sh configs/ptp1b_smoke.yaml
bash scripts/manual/step05_benchmark.sh configs/ptp1b_smoke.yaml training.splits_to_run=scaffold_bm
bash scripts/manual/step06_train_causal.sh configs/ptp1b_smoke.yaml training.splits_to_run=scaffold_bm
bash scripts/manual/step08_evaluate_runs.sh configs/ptp1b_smoke.yaml
```

Equivalent override form:

```bash
bash scripts/manual/step05_benchmark.sh configs/ptp1b.yaml smoke=true training.splits_to_run=scaffold_bm
```


### Step 2 endpoint policy (default)

- Primary training/screening dataset is **IC50-only** with `standard_relation == "="` and standardized units in **nM** (uM is converted to nM).
- Censored rows (`<`, `>`, etc.), non-positive values, unsupported units, and extreme values above `postprocess.max_value_nM` are dropped from the primary dataset.
- `pIC50` is computed as `9 - log10(IC50_nM)` and `activity_label` is generated from the configured postprocess threshold.
- Ki/Kd are not mixed into the primary labels. Optional secondary endpoint datasets are emitted separately and are not consumed by Step3+ unless explicitly configured.

### Manual step commands (copy/paste)

```bash
bash scripts/manual/step01_extract.sh configs/ptp1b.yaml
bash scripts/manual/step02_postprocess.sh configs/ptp1b.yaml
bash scripts/manual/step03_assemble_environments.sh configs/ptp1b.yaml
bash scripts/manual/step04_generate_splits.sh configs/ptp1b.yaml
bash scripts/manual/step05_benchmark.sh configs/ptp1b.yaml
bash scripts/manual/step06_train_causal.sh configs/ptp1b.yaml
bash scripts/manual/step07_counterfactuals.sh configs/ptp1b.yaml
bash scripts/manual/step08_evaluate_runs.sh configs/ptp1b.yaml
bash scripts/manual/step08a_prepare_external_inhibition.sh configs/ptp1b.yaml
bash scripts/manual/step09_cross_endpoint.sh configs/ptp1b.yaml
bash scripts/manual/step10_interpret.sh configs/ptp1b.yaml
bash scripts/manual/step11_robustness.sh configs/ptp1b.yaml
bash scripts/manual/step12a_prepare_library.sh configs/ptp1b.yaml
bash scripts/manual/step12b_screen_library.sh configs/ptp1b.yaml
bash scripts/manual/step13_analyze_screening.sh configs/ptp1b.yaml
bash scripts/manual/step14_match_features.sh configs/ptp1b.yaml
bash scripts/manual/step15_manuscript.sh configs/ptp1b.yaml
```

### Step catalog (manual mode)

| Step | Script | Main input(s) | Main output(s) |
|---|---|---|---|
| 1 | `step01_extract.sh` | `paths.chembl_sqlite` | `outputs/step1/<target>_qsar_ready.csv` |
| 2 | `step02_postprocess.sh` | `outputs/step1/<target>_qsar_ready.csv` (mixed assay records) | primary IC50 outputs (`row_level_primary.csv`, `compound_level_with_properties.csv`) + optional secondary endpoint files (e.g., `row_level_secondary_ki.csv`) |
| 3 | `step03_assemble_environments.sh` | step1/step2 CSVs + env rules | `outputs/step3/multienv_compound_level.parquet` |
| 4 | `step04_generate_splits.sh` | config + step3 dataset | `outputs/step4/<split>/*` + `outputs/step4/splits_manifest.json` |
| 5 | `step05_benchmark.sh` | step3 dataset + selected step4 splits (`training.splits_to_run`) | `outputs/step5/<target>/<split>/<run_id>/*` + latest-run pointers |
| 6 | `step06_train_causal.sh` | step3 dataset + selected step4 splits (`training.splits_to_run`) | `outputs/step6/<target>/<split>/<run_id>/*` + `latest_run.json` pointers |
| 7 | `step07_counterfactuals.sh` | run dir + step3 parquet + MMP rules | `outputs/step7/candidates/*.parquet` |
| 8 | `step08_evaluate_runs.sh` | `run_dir` or `runs_root` + step3/step4 | `outputs/step8/*` |
| 8a | `step08a_prepare_external_inhibition.sh` | external inhibition CSV + step3 parquet + splits | `data/external/processed/ptp1b_inhibition_chembl335/data/inhibition_external_final.parquet` |
| 9 | `step09_cross_endpoint.sh` | explicit `run_dir` or split-aware latest pointer + canonical external parquet + Step06 `artifacts/feature_schema.json` | `outputs/step9/*` external active/inactive inhibition evaluation (ROC-AUC/PR-AUC from continuous `pIC50_hat`; fails early on schema mismatch) |
| 10 | `step10_interpret.sh` | `run_dir` or `runs_root` (multi-run supported) + step3 parquet | `outputs/step10/<split>/<run_id>/*` |
| 11 | `step11_robustness.sh` | `run_dir` or `runs_root` + step3 parquet | `outputs/step11/*` |
| 12a | `step12a_prepare_library.sh` | config + `screening.input_path` CSV | `outputs/step12/prepared/*` + `outputs/step12/latest_prepare.json` |
| 12b | `step12b_screen_library.sh` | step12a deduplicated parquet + explicit `run_dir` (single model) or auto-discovered Step06 split runs (multi-seed ensemble) | `outputs/step12/screening/*` (seed-level + aggregated predictions) + `outputs/step12/latest_screen.json` |
| 13 | `step13_analyze_screening.sh` | config + Step12 screening outputs resolved in this order: `outputs/step12/latest_screen.json`, `outputs/step12/run_pointer.json`, then old Step12 layout | `outputs/step13/*` |
| 14 | `step14_match_features.sh` | config + Step12/Step13 outputs (same Step12 resolution order as Step13) | `outputs/step14/*` |
| 15 | `step15_manuscript.sh` | config + previous outputs; Step12 screen source supports both new and old layouts (new preferred) | `outputs/step15/*` |

### CLI status (deprecated)

The old `ptp1bqsar` CLI (`check`, `step`, `run`, `manuscript`) remains in the codebase for compatibility, but new documentation and recommended execution now use `scripts/manual/*.sh`.


### Pipeline doctor

Run a compatibility/preflight diagnosis at any time:

```bash
python scripts/pipeline_doctor.py configs/ptp1b.yaml
```

This checks core file/column contracts and run-pointer consistency used by manual mode.

Feature-schema compatibility note:
- Any featurization or model-input schema change requires rerunning Step 06 (and downstream steps).
- Step 09 validates `artifacts/feature_schema.json` from the selected Step 06 run and fails early if dimensions are incompatible with current featurization code.
- Step 09 converts regression `pIC50_hat` predictions into binary active/inactive calls for external inhibition validation (default threshold: `pIC50_hat >= 5.0`; inhibition active if `inhibition >= 50%`).

## Output folders and file explanations

Primary runtime outputs are written under:

```text
outputs/
└── pipeline_runs/
    └── <pipeline_run_id>/
        ├── pipeline_config_resolved.yaml
        ├── pipeline_steps_executed.json
        ├── pipeline_log.txt
        ├── pipeline_errors.json          # only when errors occur
        ├── provenance.json
        └── environment.txt
```

Step-specific data products are additionally written into pipeline-defined subfolders (e.g., processed QSAR tables, environment reports, screening outputs, and manuscript package files).

### What each step writes (high level)

- Steps 1–3: extraction/processing tables, environment datasets, and associated reports/figures.
- Steps 4–5: split definitions, model checkpoints, training metrics.
- Steps 7–11: counterfactual sets, evaluation summaries, interpretability and robustness outputs.
- Steps 12–14: screened candidate rankings, screening analytics, and feature matching artifacts.
- Step 15: manuscript-aligned package, checklist, and reproducibility metadata.

## Reproducibility

This project is designed around reproducible execution:

- **Resolved configuration capture** in `pipeline_config_resolved.yaml`.
- **Step manifest** in `pipeline_steps_executed.json`.
- **Runtime logs and failures** in `pipeline_log.txt` and `pipeline_errors.json`.
- **Provenance record** in `provenance.json` with run-level metadata.
- **Environment snapshot** in `environment.txt`.

For manuscript workflows, include and review:

- `manuscript_checklist.md` (completeness and reporting checklist).
- **Reproducibility Fingerprint** (recommended composite of config hash, code commit, dependency environment, and input manifest checksum).

Best practice:

1. Pin config and seed values.
2. Record commit SHA with each run.
3. Archive input file hashes and output manifest.
4. Store run directories immutably for published results.

## Example end-to-end workflow

1. Create/activate environment and install package.
2. Prepare `configs/ptp1b.yaml` and ensure `paths.chembl_sqlite` points to your local ChEMBL SQLite.
3. Make manual scripts executable and set interpreter (optional):

```bash
chmod +x scripts/manual/*.sh
export PIPELINE_PYTHON=$(which python)
```

4. Execute all steps:

```bash
bash scripts/manual/run_all.sh configs/ptp1b.yaml
```

5. Execute a focused range when iterating:

```bash
STEPS=5-10 bash scripts/manual/run_all.sh configs/ptp1b.yaml
```

6. Review artifacts under `outputs/step*/` and step-specific log files (`outputs/stepX/stepXX_*.log`).

## Troubleshooting

| Symptom | Likely cause | Suggested fix |
|---|---|---|
| `RDKit import failed` during `check` | RDKit not installed in active env | Recreate env from `environment.yml`; ensure `conda-forge` channel is available |
| `ModuleNotFoundError: No module named 'torch_geometric'` | PyG dependency missing in active env | Install dependencies from `environment.yml` (or `pip install -r requirements.txt`), then re-run Step 5+ |
| Missing script error for a step | Incomplete checkout or path issue | Confirm repository integrity and run from repo root |
| `Required path does not exist` warnings | Dataset paths not prepared | Create folders/data files referenced by config |
| `outputs_root` is missing or empty | `paths.outputs_root` is unset/wrong or not writable | Set `paths.outputs_root` in config and confirm write permissions; scripts auto-create step subfolders with `mkdir -p` |
| CUDA unavailable | GPU runtime package mismatch | Use `pytorch-cuda` version compatible with your driver per `environment.yml` comments |
| Manuscript build missing files | Upstream steps incomplete | Re-run required upstream steps or full `0-15` range |

Additional setup and pipeline details are documented in [`INSTALL.md`](INSTALL.md).

## Contributing

Recommended contribution workflow:

- Branch naming: `feature/<topic>`, `fix/<topic>`, `docs/<topic>`, `refactor/<topic>`.
- Issue labels (suggested): `bug`, `enhancement`, `docs`, `question`, `reproducibility`, `good first issue`.
- Pull request process:
  1. Open an issue or discussion for major changes.
  2. Create a focused branch.
  3. Add/update tests or validation scripts where applicable.
  4. Run key checks locally.
  5. Submit PR with motivation, methodology, and output summary.

## Citation

If you use this pipeline in research or screening campaigns, please cite it as:

```text
PTP1B Causal QSAR + Screening Pipeline.
GitHub repository, versioned release.
```

You may also include target and commit-specific provenance for exact computational reproducibility.

## License

MIT License (recommended for permissive scientific collaboration).

If your project policy differs, replace this section with the repository’s canonical license text/file.

## Contact

- Email: `waqastakkar@gmail.com`
- GitHub: `@waqastakkar`
