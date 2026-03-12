# Installation and Step 0 Unified Pipeline CLI

## Install

```bash
python -m pip install -e .
```

After installation:

```bash
ptp1bqsar --help
```

## Step 0: Unified Pipeline CLI

Step 0 introduces a single orchestration command (`ptp1bqsar`) that manages Steps 1–15 through scripts in `scripts/`.

### Config template

Use `configs/ptp1b.yaml` as the starting point.

### Commands

Validate setup:

```bash
ptp1bqsar check --config configs/ptp1b.yaml
```

Run one step:

```bash
ptp1bqsar step 12 --config configs/ptp1b.yaml --input_path data/screening/raw/AAAA.smi
```

Run a range:

```bash
ptp1bqsar run --config configs/ptp1b.yaml --steps "1-4,8,10-15"
```

Run full (Step 0 documented, Steps 1–15 executed):

```bash
ptp1bqsar run --config configs/ptp1b.yaml --steps "0-15"
```

Build manuscript pack only:

```bash
ptp1bqsar manuscript --config configs/ptp1b.yaml --paper_id ptp1b_causal_qsar_v1
```

## Pipeline run artifacts

Each run creates `outputs/pipeline_runs/<pipeline_run_id>/` with:

- `pipeline_config_resolved.yaml`
- `pipeline_steps_executed.json`
- `pipeline_log.txt`
- `pipeline_errors.json` (if any errors)
- `provenance.json`
- `environment.txt`


## Step 1: Extract target bioactivity from ChEMBL 36 SQLite (production-grade)

### Inputs

- `data/raw/chembl/chembl_36.db`
- Target ChEMBL ID, e.g. `CHEMBL335`

### Output structure

`data/interim/extracts/<TARGET_CHEMBL_ID>/` will contain:

```text
data/interim/extracts/<TARGET>/
├─ <TARGET>_raw.csv
├─ <TARGET>_qsar_ready.csv
├─ extraction_config.json
├─ provenance.json
├─ summary_tables/
│  ├─ counts_by_standard_type.csv
│  ├─ counts_by_units.csv
│  ├─ counts_by_relation.csv
│  ├─ counts_by_confidence.csv
│  └─ missingness_report.csv
└─ figures/
   ├─ fig_standard_type_distribution.svg
   ├─ fig_units_distribution.svg
   ├─ fig_confidence_distribution.svg
   └─ fig_value_distribution_log.svg
```

### Extraction command

```bash
python scripts/extract_chembl36_sqlite.py \
  --db data/raw/chembl/chembl_36.db \
  --target CHEMBLXXXX \
  --outdir data/interim/extracts/CHEMBLXXXX
```

### Reporting command

```bash
python scripts/extract_report.py \
  --input_dir data/interim/extracts/CHEMBLXXXX \
  --outdir data/interim/extracts/CHEMBLXXXX \
  --svg \
  --font "Times New Roman" \
  --bold_text \
  --palette nature5 \
  --font_title 16 --font_label 14 --font_tick 12 --font_legend 12
```

All report figures are written as SVG and style is globally enforced through Matplotlib rcParams (`savefig.format=svg`, `svg.fonttype=none`) with Times New Roman, bold text, and the fixed 5-color palette (`#E69F00`, `#009E73`, `#0072B2`, `#D55E00`, `#CC79A7`).

## Step 2 — QSAR post-processing (production-grade: tables + provenance + SVG figures)

### Objective

Convert extracted ChEMBL bioactivity into a reproducible QSAR-ready dataset with row-level pIC50 conversion, compound-level aggregation, binary labels, RDKit/Lipinski properties, manuscript-ready SVG figures, and full provenance.

### Inputs

From Step 1:

- `data/interim/extracts/<TARGET>/<TARGET>_qsar_ready.csv`

### Output structure

```text
data/processed/qsar/<TARGET>/
├─ data/
│  ├─ row_level_with_pIC50.csv
│  ├─ compound_level_pIC50.csv
│  ├─ compound_level_with_properties.csv
│  └─ summary.csv
├─ figures/
│  ├─ fig_class_balance.svg
│  ├─ fig_spider_properties_active_vs_inactive.svg
│  ├─ fig_bubble_mw_vs_logp.svg
│  ├─ fig_pIC50_distribution.svg
│  ├─ fig_endpoint_units_relations.svg
│  └─ fig_missingness_properties.svg
└─ provenance/
   ├─ run_config.json
   ├─ provenance.json
   └─ environment.txt
```

### Run Step 2

1) Create folders:

```bash
mkdir -p data/processed/qsar/CHEMBLXXXX/{data,figures,provenance}
```

2) Generate data tables:

```bash
python scripts/qsar_postprocess.py \
  --input data/interim/extracts/CHEMBLXXXX/CHEMBLXXXX_qsar_ready.csv \
  --outdir data/processed/qsar/CHEMBLXXXX/data \
  --endpoint IC50 \
  --threshold 6.0 \
  --aggregate best \
  --prefer_pchembl \
  --svg
```

3) Generate figures + provenance:

```bash
python scripts/qsar_postprocess_report.py \
  --input_dir data/processed/qsar/CHEMBLXXXX/data \
  --outdir data/processed/qsar/CHEMBLXXXX \
  --svg \
  --font "Times New Roman" \
  --bold_text \
  --palette nature5 \
  --font_title 16 --font_label 14 --font_tick 12 --font_legend 12
```

All Step 2 figures are emitted as SVG (editable text, not paths) with `svg.fonttype=none`, Times New Roman, bold text (titles/labels/ticks/legend), and a fixed colorblind-friendly Nature palette (`#E69F00`, `#009E73`, `#0072B2`, `#D55E00`, `#CC79A7`).

## Step 3 — Assemble multi-environment dataset + validate domain shift + latent environment discovery

### Step 3 output structure

```text
data/processed/environments/<TARGET>/
├─ data/
│  ├─ multienv_row_level.parquet
│  ├─ multienv_compound_level.parquet
│  ├─ env_definitions.json
│  ├─ env_vector_schema.json
│  ├─ env_counts.csv
│  ├─ series_assignments.csv
│  ├─ learned_env_assignments.csv
│  ├─ learned_env_feature_matrix.parquet
│  └─ learned_env_scaler.json
├─ reports/
│  ├─ shift_metrics.csv
│  ├─ env_predictability.csv
│  ├─ scaffold_overlap.csv
│  ├─ label_shift.csv
│  ├─ missingness_by_env.csv
│  ├─ alignment_metrics.csv
│  ├─ cluster_profiles.csv
│  ├─ cluster_purity.csv
│  └─ clustering_stability.csv
├─ figures/
│  ├─ fig_env_counts.svg
│  ├─ fig_label_distribution_by_env.svg
│  ├─ fig_active_rate_by_env.svg
│  ├─ fig_scaffold_overlap_heatmap.svg
│  ├─ fig_shift_metrics.svg
│  ├─ fig_env_predictability.svg
│  ├─ fig_cluster_sizes.svg
│  ├─ fig_cluster_profiles.svg
│  ├─ fig_alignment_ari_nmi.svg
│  └─ fig_manual_vs_learned_contingency.svg
└─ provenance/
   ├─ run_config.json
   ├─ provenance.json
   └─ environment.txt
```

### 3.1 Create folders

```bash
mkdir -p data/processed/environments/CHEMBLXXXX/{data,reports,figures,provenance}
```

### 3.2 Assemble explicit environments

```bash
python scripts/assemble_environments.py \
  --target CHEMBLXXXX \
  --row_level_csv data/processed/qsar/CHEMBLXXXX/data/row_level_with_pIC50.csv \
  --compound_level_csv data/processed/qsar/CHEMBLXXXX/data/compound_level_with_properties.csv \
  --raw_extract_csv data/interim/extracts/CHEMBLXXXX/CHEMBLXXXX_raw.csv \
  --outdir data/processed/environments/CHEMBLXXXX/data \
  --env_keys assay_type species readout publication chemistry_regime series \
  --bbb_rules configs/bbb_rules.yaml \
  --series_rules configs/series_rules.yaml
```

### 3.3 Validate environment shift and leakage risk

```bash
python scripts/env_validation_report.py \
  --input_dir data/processed/environments/CHEMBLXXXX/data \
  --outdir data/processed/environments/CHEMBLXXXX \
  --svg \
  --font "Times New Roman" \
  --bold_text \
  --palette nature5 \
  --font_title 16 --font_label 14 --font_tick 12 --font_legend 12
```

### 3.4 Latent environment discovery (unsupervised domains)

Use learned environments to verify domain structure, test alignment with manual environments, and detect hidden assay/publication regimes.

```bash
python scripts/latent_env_discovery.py \
  --input_compound_parquet data/processed/environments/CHEMBLXXXX/data/multienv_compound_level.parquet \
  --outdir data/processed/environments/CHEMBLXXXX \
  --features MW LogP TPSA HBD HBA RotB Rings \
  --method kmeans \
  --k_min 3 --k_max 12 \
  --select_by silhouette \
  --random_seed 42 \
  --svg \
  --font "Times New Roman" \
  --bold_text \
  --palette nature5 \
  --font_title 16 --font_label 14 --font_tick 12 --font_legend 12
```

All Step 3 figures are SVG only with editable text (`svg.fonttype=none`), Times New Roman font, bold text for titles/labels/ticks/legends, configurable font sizes, and the fixed Nature palette: `#E69F00`, `#009E73`, `#0072B2`, `#D55E00`, `#CC79A7`.

## Step 4 — Nature-level strict split suite + leakage audits + shift quantification

### Purpose

Step 4 builds benchmark-grade OOD split manifests (random, scaffold, time, environment holdouts, combo OOD, matched-shift, hard-boundary, and neighbor-similarity) with hard leakage/integrity checks, similarity leakage auditing, split-level shift quantification, reproducible provenance, and manuscript-ready SVG figures.

### Output structure

```text
data/processed/splits/<TARGET>/
├─ splits/
│  ├─ random/
│  ├─ scaffold_bm/
│  ├─ time_publication/
│  ├─ env_holdout_assay/
│  ├─ env_holdout_pubfam/
│  ├─ combo_scaffold_env/
│  ├─ combo_time_env/
│  ├─ scaffold_matched_props/
│  ├─ hard_boundary/
│  └─ neighbor_similarity/
├─ reports/
│  ├─ split_summary.csv
│  ├─ label_shift.csv
│  ├─ covariate_shift.csv
│  ├─ group_integrity_checks.csv
│  ├─ scaffold_overlap.csv
│  ├─ env_overlap.csv
│  ├─ similarity_leakage.csv
│  ├─ matching_quality.csv
│  └─ time_coverage.csv
├─ figures/
│  ├─ fig_split_sizes.svg
│  ├─ fig_label_shift_by_split.svg
│  ├─ fig_covariate_shift_props.svg
│  ├─ fig_scaffold_overlap_by_split.svg
│  ├─ fig_env_overlap_by_split.svg
│  ├─ fig_similarity_leakage.svg
│  ├─ fig_time_split_timeline.svg
│  └─ fig_matching_quality.svg
└─ provenance/
   ├─ run_config.json
   ├─ provenance.json
   └─ environment.txt
```

### Run split generation

```bash
python scripts/make_splits.py \
  --target CHEMBLXXXX \
  --input_parquet data/processed/environments/CHEMBLXXXX/data/multienv_compound_level.parquet \
  --outdir data/processed/splits/CHEMBLXXXX/splits \
  --seed 42 \
  --train_frac 0.8 --val_frac 0.1 --test_frac 0.1 \
  --enable random scaffold_bm time_publication env_holdout_assay env_holdout_pubfam \
           combo_scaffold_env combo_time_env scaffold_matched_props hard_boundary neighbor_similarity \
  --time_key publication_year \
  --assay_holdout_value cell-based \
  --similarity_radius 2 --similarity_nbits 2048 \
  --neighbor_threshold 0.65 \
  --hard_delta 0.3 \
  --match_props MW LogP TPSA HBD HBA RotB Rings
```

### Run split reporting + SVG figures

```bash
python scripts/splits_report.py \
  --input_parquet data/processed/environments/CHEMBLXXXX/data/multienv_compound_level.parquet \
  --splits_dir data/processed/splits/CHEMBLXXXX/splits \
  --outdir data/processed/splits/CHEMBLXXXX \
  --font "Times New Roman" \
  --font_title 16 --font_label 14 --font_tick 12 --font_legend 12
```

### 4.4 BBB-aware stratification

#### Outputs

```text
data/processed/bbb/<TARGET>/
├─ data/
│  ├─ bbb_annotations.parquet
│  ├─ cns_mpo_components.csv
│  ├─ cns_bins.csv
│  └─ pgp_predictions.csv (optional)
├─ reports/
│  ├─ bbb_summary.csv
│  ├─ bbb_shift_by_split.csv
│  └─ cns_vs_non_cns_overlap.csv
├─ figures/
│  ├─ fig_cns_mpo_distribution.svg
│  ├─ fig_cns_like_rate_by_split.svg
│  ├─ fig_potency_vs_cns_mpo.svg
│  ├─ fig_pareto_frontier.svg
│  └─ fig_pgp_risk_distribution.svg (optional)
└─ provenance/
   ├─ run_config.json
   ├─ provenance.json
   └─ environment.txt
```

#### Run command

```bash
python scripts/bbb_stratify.py \
  --target CHEMBLXXXX \
  --input_parquet data/processed/environments/CHEMBLXXXX/data/multienv_compound_level.parquet \
  --splits_dir data/processed/splits/CHEMBLXXXX/splits \
  --outdir data/processed/bbb/CHEMBLXXXX \
  --compute_cns_mpo \
  --cns_mpo_threshold 4.0 \
  --cns_bins 0 2 4 6 \
  --pgp_model_path "" \
  --font "Times New Roman" \
  --font_title 16 --font_label 14 --font_tick 12 --font_legend 12
```

All Step 4 and Step 4.4 figures are SVG-only with editable text (`svg.fonttype=none`), Times New Roman, bold titles/labels/ticks/legend, configurable font sizes, and the fixed Nature 5-color palette (`#E69F00`, `#009E73`, `#0072B2`, `#D55E00`, `#CC79A7`).

## Step 5 — Causal QSAR core (encoder + invariance + benchmark)

### Objectives

- Train a causal QSAR GNN where predictions are made from invariant representation `z_inv` only.
- Enforce environment invariance using GRL adversary (`env_id_manual` from Step 3), optional IRM, and disentanglement (`z_inv ⟂ z_spu`).
- Produce production-grade artifacts (checkpoints, predictions, reports, figures, provenance) with strict reproducibility.

### Required inputs

- Dataset parquet: `data/processed/environments/<TARGET>/data/multienv_compound_level.parquet`
- Splits manifest: `data/processed/splits/<TARGET>/splits/<split_name>/{train_ids.csv,val_ids.csv,test_ids.csv}`
- Optional BBB annotations: `data/processed/bbb/<TARGET>/data/bbb_annotations.parquet`

### Output structure

```text
outputs/runs/<TARGET>/<split_name>/<run_id>/
  checkpoints/{best.pt,last.pt}
  configs/{train_config.yaml,resolved_config.yaml}
  logs/{train.log,metrics.jsonl}
  predictions/{train_predictions.parquet,val_predictions.parquet,test_predictions.parquet}
  reports/{metrics_summary.csv,per_env_metrics.csv,invariance_checks.csv,calibration.csv,bbb_metrics.csv,ablation_table.csv}
  figures/*.svg
  provenance/{provenance.json,run_config.json,environment.txt}
```

### Train command example

```bash
python scripts/train_causal_qsar.py \
  --target CHEMBLXXXX \
  --dataset_parquet data/processed/environments/CHEMBLXXXX/data/multienv_compound_level.parquet \
  --splits_dir data/processed/splits/CHEMBLXXXX/splits \
  --split_name split_01 \
  --outdir outputs/runs \
  --task regression \
  --label_col pIC50 \
  --env_col env_id_manual \
  --encoder gine \
  --z_dim 128 --z_inv_dim 64 --z_spu_dim 64 \
  --lambda_adv 0.5 --lambda_irm 0.1 --lambda_dis 0.1 \
  --epochs 30 --batch_size 64 --lr 1e-3 --seed 42 \
  --svg --font "Times New Roman" --bold_text --palette nature5 \
  --font_title 16 --font_label 14 --font_tick 12 --font_legend 12
```

### Benchmark command example

```bash
python scripts/run_benchmark.py \
  --target CHEMBLXXXX \
  --dataset_parquet data/processed/environments/CHEMBLXXXX/data/multienv_compound_level.parquet \
  --splits_dir data/processed/splits/CHEMBLXXXX/splits \
  --split_names split_01,split_02,split_03 \
  --outdir outputs/runs \
  --task regression \
  --label_col pIC50 \
  --env_col env_id_manual \
  --seeds 42,43 \
  --ablations full,no_adv,no_irm,no_dis \
  --epochs 30 --batch_size 64 --lr 1e-3 \
  --svg --font "Times New Roman" --bold_text --palette nature5 \
  --font_title 16 --font_label 14 --font_tick 12 --font_legend 12
```

All Step 5 figures are SVG only with editable text (`svg.fonttype=none`), Times New Roman, bold titles/labels/ticks/legend, configurable font sizes, and the fixed Nature 5-color palette: `#E69F00`, `#009E73`, `#0072B2`, `#D55E00`, `#CC79A7`.

## Step 6 — Exact training objective (production-grade + Nature-level)

### Objective

Step 6 extends Step 5 training with a fully specified objective:

`L = L_pred + lambda_adv * L_env + lambda_irm * L_irm + lambda_dis * L_dis`

with modular loss modes, environment imbalance correction, warmup+ramp schedules, and per-epoch diagnostics.

### Supported loss options

- Prediction loss:
  - Regression: `--loss_pred {mse,huber}`
  - Classification: `--loss_cls {bce,focal}`
  - Optional sample weighting: `--sample_weight_col <column_name>`
- Adversarial invariance loss: `--loss_env {ce,weighted_ce}`
  - `weighted_ce` uses inverse-frequency env class weights from training split.
- IRM penalty: `--irm_mode {none,irmv1}`
  - IRMv1 computes per-environment risk and squared gradient norm wrt scalar dummy scale.
- Disentanglement loss: `--disentangle {none,orthogonality,hsic}`
  - Logs cosine similarity and HSIC diagnostics each epoch.

### Stable schedule

- Warmup: `--warmup_epochs N` trains with `L_pred` only.
- Ramp: `--ramp_epochs R` linearly ramps `lambda_adv/lambda_irm/lambda_dis` from `0` to target.
- Schedule is saved to `reports/schedule.csv`.

### Step 6 artifacts

Additional outputs in:

`outputs/runs/<TARGET>/<split>/<run_id>/`

- `reports/loss_breakdown.csv`
- `reports/irm_diagnostics.csv` (when `--irm_mode irmv1`)
- `reports/disentanglement_diagnostics.csv` (when `--disentangle != none`)
- `reports/env_balance.csv`
- `reports/schedule.csv`
- `figures/fig_loss_components_over_time.svg`
- `figures/fig_irm_penalty_over_time.svg`
- `figures/fig_disentanglement_over_time.svg`
- `figures/fig_env_weights_distribution.svg`

All Step 6 figures are SVG with editable text (`svg.fonttype=none`) and Times New Roman bold styling with the fixed Nature 5 palette.

### Example command (all components enabled)

```bash
python scripts/train_causal_qsar.py \
  --target CHEMBLXXXX \
  --dataset_parquet data/processed/environments/CHEMBLXXXX/data/multienv_compound_level.parquet \
  --splits_dir data/processed/splits/CHEMBLXXXX/splits \
  --split_name env_holdout_pubfam \
  --outdir outputs/runs \
  --task regression --label_col pIC50 \
  --env_col env_id_manual \
  --encoder gine \
  --lambda_adv 1.0 --lambda_irm 0.1 --lambda_dis 0.1 \
  --loss_pred huber \
  --loss_env weighted_ce \
  --irm_mode irmv1 \
  --disentangle orthogonality \
  --warmup_epochs 10 \
  --ramp_epochs 30 \
  --svg --font "Times New Roman" --bold_text --palette nature5 \
  --font_title 16 --font_label 14 --font_tick 12 --font_legend 12
```

## Step 7 — Counterfactual SAR (valid edits + BBB constraints + consistency training)

### Step 7 output structure

```text
outputs/counterfactuals/<TARGET>/<run_id>/
├─ rules/
│  ├─ mmp_rules.parquet
│  ├─ rule_stats.csv
│  └─ rule_provenance.json
├─ candidates/
│  ├─ seeds.parquet
│  ├─ generated_counterfactuals.parquet
│  ├─ filtered_counterfactuals.parquet
│  ├─ ranked_topk.parquet
│  └─ duplicates_removed.csv
├─ evaluation/
│  ├─ delta_predictions.csv
│  ├─ series_ranking_constraints.csv
│  ├─ monotonicity_checks.csv
│  ├─ validity_sanity.csv
│  └─ bbb_constraint_report.csv
├─ figures/
│  ├─ fig_edit_type_distribution.svg
│  ├─ fig_deltaY_distribution.svg
│  ├─ fig_pareto_potency_vs_cns.svg
│  ├─ fig_counterfactual_success_rate.svg
│  ├─ fig_monotonicity_violations.svg
│  └─ fig_top_edits_examples.svg
└─ provenance/
   ├─ run_config.json
   ├─ provenance.json
   └─ environment.txt
```

### 7.1 Build MMP rules

```bash
python scripts/build_mmp_rules.py \
  --target CHEMBLXXXX \
  --input_parquet data/processed/environments/CHEMBLXXXX/data/multienv_compound_level.parquet \
  --outdir outputs/counterfactuals/CHEMBLXXXX/<RUN_ID> \
  --min_support 10 \
  --max_cut_bonds 1 \
  --svg --font "Times New Roman" --bold_text --palette nature5 \
  --font_title 16 --font_label 14 --font_tick 12 --font_legend 12
```

### 7.2 Generate constrained counterfactuals and rank edits

```bash
python scripts/generate_counterfactuals.py \
  --target CHEMBLXXXX \
  --run_dir outputs/runs/CHEMBLXXXX/scaffold_bm/<RUN_ID> \
  --checkpoint checkpoints/best.pt \
  --dataset_parquet data/processed/environments/CHEMBLXXXX/data/multienv_compound_level.parquet \
  --bbb_parquet data/processed/bbb/CHEMBLXXXX/data/bbb_annotations.parquet \
  --mmp_rules_parquet outputs/counterfactuals/CHEMBLXXXX/<RUN_ID>/rules/mmp_rules.parquet \
  --outdir outputs/counterfactuals/CHEMBLXXXX/<RUN_ID> \
  --preserve scaffold \
  --cns_constraint keep_cns_like \
  --cns_mpo_threshold 4.0 \
  --max_edits_per_seed 50 \
  --topk_per_seed 5 \
  --min_tanimoto 0.3 --max_tanimoto 0.95 \
  --svg --font "Times New Roman" --bold_text --palette nature5 \
  --font_title 16 --font_label 14 --font_tick 12 --font_legend 12
```

### 7.3 Optional fine-tuning with counterfactual consistency loss

```bash
python scripts/finetune_with_counterfactuals.py \
  --target CHEMBLXXXX \
  --base_run_dir outputs/runs/CHEMBLXXXX/scaffold_bm/<RUN_ID> \
  --counterfactuals_parquet outputs/counterfactuals/CHEMBLXXXX/<RUN_ID>/candidates/filtered_counterfactuals.parquet \
  --dataset_parquet data/processed/environments/CHEMBLXXXX/data/multienv_compound_level.parquet \
  --outdir outputs/runs/CHEMBLXXXX/scaffold_bm/<RUN_ID>_cf \
  --lambda_cf 0.2 \
  --cf_mode ranking+monotonic+smooth \
  --epochs 20 --lr 5e-5 --seed 42 \
  --svg --font "Times New Roman" --bold_text --palette nature5 \
  --font_title 16 --font_label 14 --font_tick 12 --font_legend 12
```

Step 7 figure contract: SVG only (`savefig.format=svg`, `svg.fonttype=none`), Times New Roman, all text bold (titles/labels/ticks/legend), configurable font sizes, and the fixed Nature 5-color palette (`#E69F00`, `#009E73`, `#0072B2`, `#D55E00`, `#CC79A7`).

## Step 8 — Evaluation suite (performance + OOD generalization + causal sanity checks)

### 8.1 Evaluate one target across multiple runs

```bash
mkdir -p outputs/evaluation

python scripts/evaluate_runs.py \
  --target CHEMBLXXXX \
  --runs_root outputs/runs/CHEMBLXXXX \
  --splits_dir data/processed/splits/CHEMBLXXXX/splits \
  --dataset_parquet data/processed/environments/CHEMBLXXXX/data/multienv_compound_level.parquet \
  --bbb_parquet data/processed/bbb/CHEMBLXXXX/data/bbb_annotations.parquet \
  --counterfactuals_root outputs/counterfactuals/CHEMBLXXXX \
  --outdir outputs/evaluation/CHEMBLXXXX/eval_v1 \
  --task regression --label_col pIC50 \
  --env_col env_id_manual \
  --compute_envprobe \
  --compute_zinv_stability \
  --compute_cf_consistency \
  --bootstrap 1000 \
  --svg --font "Times New Roman" --bold_text --palette nature5 \
  --font_title 16 --font_label 14 --font_tick 12 --font_legend 12
```

### 8.2 Optional paper figure pack export

```bash
python scripts/make_paper_figures.py \
  --eval_dir outputs/evaluation/CHEMBLXXXX/eval_v1 \
  --outdir outputs/evaluation/CHEMBLXXXX/eval_v1/figures \
  --svg --font "Times New Roman" --bold_text --palette nature5 \
  --font_title 16 --font_label 14 --font_tick 12 --font_legend 12
```

All Step 7/8 figures are SVG-only with editable text (`svg.fonttype=none`), Times New Roman, bold typography, and the Nature 5-color palette through `scripts/plot_style.py`.

## Step 9 — Cross-endpoint externalization (IC50 regression → Inhibition % classification)

### Purpose

Step 9 performs strict cross-endpoint externalization by training on IC50 regression (Steps 5–6) and evaluating on an external Inhibition (%) set as binary classification, using the regression output (`pIC50_hat`) as the scoring function without retraining.

### Output structures

Preparation output:

```text
data/external/processed/ptp1b_inhibition_chembl335/
├─ data/
│  ├─ inhibition_raw_parsed.parquet
│  ├─ inhibition_clean.parquet
│  ├─ inhibition_dedup_internal.parquet
│  └─ inhibition_external_final.parquet
├─ reports/
│  ├─ parsing_summary.csv
│  ├─ value_sanity.csv
│  ├─ overlap_with_ic50.csv
│  ├─ overlap_by_split_membership.csv
│  └─ shift_vs_ic50_train.csv
├─ figures/
│  ├─ fig_inhibition_value_distribution.svg
│  ├─ fig_overlap_breakdown.svg
│  └─ fig_shift_vs_ic50_train.svg
└─ provenance/
   ├─ run_config.json
   ├─ provenance.json
   └─ environment.txt
```

Evaluation output:

```text
outputs/evaluation_cross_endpoint/<TARGET>/<eval_id>/
├─ predictions/
│  ├─ external_predictions.parquet
│  └─ external_scored.parquet
├─ reports/
│  ├─ cross_endpoint_metrics.csv
│  ├─ threshold_sensitivity.csv
│  ├─ calibration_external.csv
│  ├─ cns_stratified_metrics.csv
│  └─ cf_consistency_external.csv
├─ figures/
│  ├─ fig_cross_endpoint_roc.svg
│  ├─ fig_cross_endpoint_pr.svg
│  ├─ fig_threshold_sensitivity.svg
│  ├─ fig_calibration_external.svg
│  └─ fig_cns_stratified_external.svg
└─ provenance/
   ├─ run_config.json
   ├─ provenance.json
   └─ environment.txt
```

### Run Step 9

Prepare external inhibition dataset with strict leakage removal:

```bash
python scripts/prepare_inhibition_external.py \
  --target CHEMBL335 \
  --input_csv data/external/raw/ptp1b_inhibition_chembl335.csv \
  --ic50_parquet data/processed/environments/CHEMBL335/data/multienv_compound_level.parquet \
  --splits_dir data/processed/splits/CHEMBL335/splits \
  --split_name scaffold_bm \
  --outdir data/external/processed/ptp1b_inhibition_chembl335 \
  --inhib_threshold 50 \
  --svg --font "Times New Roman" --bold_text --palette nature5 \
  --font_title 16 --font_label 14 --font_tick 12 --font_legend 12
```

Evaluate trained regression model on external inhibition set:

```bash
python scripts/evaluate_cross_endpoint.py \
  --target CHEMBL335 \
  --run_dir outputs/runs/CHEMBL335/scaffold_bm/<RUN_ID> \
  --checkpoint checkpoints/best.pt \
  --external_parquet data/external/processed/ptp1b_inhibition_chembl335/data/inhibition_external_final.parquet \
  --outdir outputs/evaluation_cross_endpoint/CHEMBL335/inhib_ext_v1 \
  --pIC50_threshold 6.0 \
  --threshold_grid 5.0 5.5 6.0 6.5 7.0 \
  --enable_calibration false \
  --bbb_parquet data/processed/bbb/CHEMBL335/data/bbb_annotations.parquet \
  --svg --font "Times New Roman" --bold_text --palette nature5 \
  --font_title 16 --font_label 14 --font_tick 12 --font_legend 12
```

### Figure style contract (mandatory)

All Step 9 figures are SVG-only with editable text (`matplotlib.rcParams['savefig.format']='svg'`, `matplotlib.rcParams['svg.fonttype']='none'`), use Times New Roman globally, enforce bold text for titles/labels/ticks/legend, expose CLI-configurable font sizes, and use the Nature 5-color palette: `#E69F00`, `#009E73`, `#0072B2`, `#D55E00`, `#CC79A7`.

## Step 10 — Nature-level interpretability: R-groups + fragments + shape + z_inv attributions

### Objective

Produce manuscript-grade interpretability for the final causal QSAR model:

- series-aware R-group SAR with RDKit RGroupDecomposition
- fragment and functional-group enrichment
- docking-free shape analysis (ETKDG conformers + NPR1/NPR2)
- z_inv-focused attributions (IG preferred, Grad×Input fallback)

### Inputs

Required:

- `--run_dir outputs/runs/<TARGET>/<split>/<run_id>`
- `--dataset_parquet data/processed/environments/<TARGET>/data/multienv_compound_level.parquet`

Optional:

- `--bbb_parquet data/processed/bbb/<TARGET>/data/bbb_annotations.parquet`
- `--counterfactuals_parquet outputs/counterfactuals/<TARGET>/<run_id>/candidates/ranked_topk.parquet`

### Outputs

The command writes:

`outputs/interpretability/<TARGET>/<run_id>/` with folders `rgroup/`, `fragments/`, `shape/`, `attribution/`, `figures/`, `figure_data/`, and `provenance/`.

### Run Step 10

```bash
python scripts/interpret_model.py \
  --target CHEMBL335 \
  --run_dir outputs/runs/CHEMBL335/scaffold_bm/<RUN_ID> \
  --dataset_parquet data/processed/environments/CHEMBL335/data/multienv_compound_level.parquet \
  --bbb_parquet data/processed/bbb/CHEMBL335/data/bbb_annotations.parquet \
  --counterfactuals_parquet outputs/counterfactuals/CHEMBL335/<RUN_ID>/candidates/ranked_topk.parquet \
  --outdir outputs/interpretability/CHEMBL335/<RUN_ID> \
  --rgroup_series_min_n 8 \
  --shape_etkdg_confs 10 \
  --shape_seed 42 \
  --shape_select lowest_uff_energy \
  --attribution_method integrated_gradients \
  --attribution_target z_inv \
  --svg --font "Times New Roman" --bold_text --palette nature5 \
  --font_title 16 --font_label 14 --font_tick 12 --font_legend 12
```

### Global figure style contract

All Step 10 figures are SVG-only and use editable text (`savefig.format="svg"`, `svg.fonttype="none"`), Times New Roman, bold text for titles/labels/ticks/legends, and the Nature 5-color palette (`#E69F00`, `#009E73`, `#0072B2`, `#D55E00`, `#CC79A7`). Numeric data used to draw every figure is also saved under `figure_data/` for reproducibility.

## Step 11 — Robustness & Uncertainty Suite

### Objective

Given multi-run training outputs from Steps 5/6, generate:
- deep-ensemble mean prediction and uncertainty (`yhat_mean`, `yhat_std`)
- split conformal prediction intervals (validation-calibrated)
- applicability domain (fingerprint + embedding)
- multi-seed stability summaries and statistical tests
- manuscript-ready SVG figures + paired numeric `figure_data` CSVs

### Inputs

Required:
- `--runs_root outputs/runs/<TARGET>`
- `--dataset_parquet data/processed/environments/<TARGET>/data/multienv_compound_level.parquet`

Optional:
- `--bbb_parquet data/processed/bbb/<TARGET>/data/bbb_annotations.parquet`
- `--external_scored_parquet outputs/evaluation_cross_endpoint/<TARGET>/<eval_id>/predictions/external_scored.parquet`

Each run is discovered recursively and considered valid if it contains:
- `checkpoints/best.pt`
- `predictions/test_predictions.parquet`

### Outputs

`outputs/robustness/<TARGET>/<robust_id>/` with exactly:
- `manifests/` (`runs_index.csv`, `groups_index.csv`, `ensemble_manifest.json`)
- `ensemble/` (`ensemble_predictions_{train,val,test}.parquet`, `ensemble_uncertainty.csv`, `selective_prediction.csv`)
- `conformal/` (`conformal_calibration.csv`, `conformal_intervals_test.parquet`, `conformal_coverage.csv`, `interval_width_by_split.csv`)
- `applicability_domain/` (`ad_fingerprint.parquet`, `ad_embedding.parquet`, `error_vs_ad.csv`, `uncertainty_vs_ad.csv`)
- `stability/` (`seed_stability_metrics.csv`, `ablation_stability_table.csv`, `statistical_tests.csv`)
- `figures/` (all required SVGs)
- `figure_data/` (one CSV per figure)
- `provenance/` (`run_config.json`, `provenance.json`, `environment.txt`)

### Run Step 11

```bash
python scripts/evaluate_robustness.py \
  --target CHEMBL335 \
  --runs_root outputs/runs/CHEMBL335 \
  --dataset_parquet data/processed/environments/CHEMBL335/data/multienv_compound_level.parquet \
  --bbb_parquet data/processed/bbb/CHEMBL335/data/bbb_annotations.parquet \
  --outdir outputs/robustness/CHEMBL335/robust_v1 \
  --task regression --label_col pIC50 \
  --group_by split_name ablation \
  --ensemble_size 5 \
  --conformal_coverage 0.90 \
  --ad_fingerprint morgan --ad_radius 2 --ad_nbits 2048 \
  --ad_embedding z_inv --ad_k 1 \
  --svg --font "Times New Roman" --bold_text --palette nature5 \
  --font_title 16 --font_label 14 --font_tick 12 --font_legend 12
```

Optional external conformal check:

```bash
python scripts/evaluate_robustness.py \
  --target CHEMBL335 \
  --runs_root outputs/runs/CHEMBL335 \
  --dataset_parquet data/processed/environments/CHEMBL335/data/multienv_compound_level.parquet \
  --external_scored_parquet outputs/evaluation_cross_endpoint/CHEMBL335/inhib_ext_v1/predictions/external_scored.parquet \
  --outdir outputs/robustness/CHEMBL335/robust_v1 \
  --task regression --label_col pIC50 \
  --conformal_coverage 0.90 \
  --svg --font "Times New Roman" --bold_text --palette nature5 \
  --font_title 16 --font_label 14 --font_tick 12 --font_legend 12
```

### Global figure style contract

All Step 11 figures are SVG-only with editable text (`savefig.format="svg"`, `svg.fonttype="none"`), use Times New Roman, force bold text for titles/labels/ticks/legend, expose CLI font sizes (`--font_title`, `--font_label`, `--font_tick`, `--font_legend`), and use the Nature palette:
`#E69F00`, `#009E73`, `#0072B2`, `#D55E00`, `#CC79A7`.

## Step 12 — Screening database ingestion + feature parity + scoring + ranking

### Objective

Ingest vendor/library files (`.smi`/`.csv`), parse + clean with full auditability, compute properties and BBB/CNS metrics, featurize with strict training parity (`artifacts/feature_schema.json`), run single-model or ensemble inference, apply applicability domain filtering, and export ranked hitlists with SVG manuscript figures plus paired `figure_data` CSVs.

### Inputs

Required:
- `--run_dir outputs/runs/<TARGET>/<split>/<run_id>`
- `--input_path <library.smi|library.csv>`
- `--input_format {smi,csv}`

Optional:
- `--use_ensemble_manifest outputs/robustness/<TARGET>/<robust_id>/manifests/ensemble_manifest.json`
- CSV mapping: `--sep`, `--header`, `--smiles_col`, `--id_col`, `--name_col`
- SMI mapping: `--smi_layout {smiles_id,smiles_name_id,smiles_only}`
- BBB/AD/ranking: `--compute_bbb`, `--cns_mpo_threshold`, `--compute_ad`, `--ad_mode`, `--ad_threshold`, `--topk`
- Plot style controls: `--svg --font "Times New Roman" --bold_text --palette nature5 --font_title --font_label --font_tick --font_legend`

### Outputs

`outputs/screening/<TARGET>/<screen_id>/` with:
- `input/library_manifest.csv`, `input/input_fingerprint.json`
- `processed/library_raw_parsed.parquet`, `library_clean.parquet`, `library_dedup.parquet`, `library_with_props.parquet`, `featurization_report.csv`
- `predictions/scored_single_model.parquet`, `scored_ensemble.parquet`, `scored_with_uncertainty.parquet`
- `ranking/ranked_all.parquet`, `ranked_cns_like.parquet`, `ranked_in_domain.parquet`, `ranked_cns_like_in_domain.parquet`, `top_100.csv`, `top_500.csv`, `selection_report.csv`
- `figures/fig_score_distribution.svg`, `fig_uncertainty_distribution.svg`, `fig_pareto_score_vs_cns.svg`, `fig_score_vs_ad.svg`, `fig_topk_property_summary.svg`
- `figure_data/score_distribution.csv`, `uncertainty_distribution.csv`, `pareto_score_vs_cns.csv`, `score_vs_ad.csv`, `topk_property_summary.csv`
- `provenance/run_config.json`, `provenance.json`, `environment.txt`

### Run Step 12

SMI example:

```bash
python scripts/screen_library.py \
  --target CHEMBL335 \
  --run_dir outputs/runs/CHEMBL335/scaffold_bm/<RUN_ID> \
  --input_path data/screening/raw/library.smi \
  --input_format smi --smi_layout smiles_id \
  --outdir outputs/screening \
  --use_ensemble_manifest outputs/robustness/CHEMBL335/robust_v1/manifests/ensemble_manifest.json \
  --compute_bbb true --cns_mpo_threshold 4.0 \
  --compute_ad true --ad_mode fingerprint --ad_threshold 0.35 \
  --topk 500 \
  --svg --font "Times New Roman" --bold_text --palette nature5 \
  --font_title 16 --font_label 14 --font_tick 12 --font_legend 12
```

CSV example:

```bash
python scripts/screen_library.py \
  --target CHEMBL335 \
  --run_dir outputs/runs/CHEMBL335/scaffold_bm/<RUN_ID> \
  --input_path data/screening/raw/library.csv \
  --input_format csv --sep "," --header true \
  --smiles_col SMILES --id_col CompoundID --name_col Name \
  --outdir outputs/screening \
  --compute_bbb true --compute_ad true --ad_mode fingerprint --ad_threshold 0.35 \
  --topk 500 \
  --svg --font "Times New Roman" --bold_text --palette nature5 \
  --font_title 16 --font_label 14 --font_tick 12 --font_legend 12
```

### Style contract (mandatory)

Step 12 figures are SVG-only with editable text (`savefig.format="svg"`, `svg.fonttype="none"`), use Times New Roman, and force bold titles/labels/ticks/legend with Nature 5-color palette (`#E69F00`, `#009E73`, `#0072B2`, `#D55E00`, `#CC79A7`).

## Step 13 — Screening Analysis (Hit triage, novelty, diversity, AD/BBB, series discovery)

### Objective

Given `outputs/screening/<TARGET>/<screen_id>/ranking/ranked_*.parquet`, generate manuscript-grade hit triage with quality diagnostics, novelty vs training, diversity-aware selection, risk-controlled filters, liability flags, and chemotype series discovery.

### Inputs

Required:
- `--screen_dir outputs/screening/<TARGET>/<screen_id>`
- `--train_parquet data/processed/environments/<TARGET>/data/multienv_compound_level.parquet`

Optional:
- ranking files if present: `ranking/ranked_all.parquet`, `ranking/ranked_cns_like_in_domain.parquet`
- style/selection controls (`--topk`, `--diverse_k`, thresholds, SVG style args)

### Outputs

`outputs/screening_analysis/<TARGET>/<screen_id>/`
- `reports/screening_summary.csv`, `novelty_report.csv`, `scaffold_novelty.csv`, `clustering_summary.csv`, `diversity_selection.csv`, `risk_controlled_selection.csv`, `property_liability_flags.csv`, `series_discovery.csv`
- `selections/top50_risk_controlled.csv`, `top100_diverse.csv`, `top200_cns_diverse.csv`, `chemotype_leads.csv`
- `figures/fig_hit_score_distribution.svg`, `fig_hit_uncertainty_distribution.svg`, `fig_pareto_score_uncertainty.svg`, `fig_pareto_score_cns.svg`, `fig_score_vs_ad.svg`, `fig_scaffold_novelty.svg`, `fig_cluster_sizes.svg`, `fig_diversity_tradeoff.svg`, `fig_property_distributions_topk.svg`, `fig_triage_score_uncertainty_ad.svg`
- `figure_data/hit_score_distribution.csv`, `hit_uncertainty_distribution.csv`, `pareto_score_uncertainty.csv`, `pareto_score_cns.csv`, `score_vs_ad.csv`, `scaffold_novelty_plot.csv`, `cluster_sizes_plot.csv`, `diversity_tradeoff_plot.csv`, `property_distributions_topk.csv`, `triage_score_uncertainty_ad.csv`
- `provenance/run_config.json`, `provenance.json`, `environment.txt`

### Run Step 13

```bash
python scripts/analyze_screening.py \
  --target CHEMBL335 \
  --screen_dir outputs/screening/CHEMBL335/zinc_screen_v1 \
  --train_parquet data/processed/environments/CHEMBL335/data/multienv_compound_level.parquet \
  --outdir outputs/screening_analysis/CHEMBL335/zinc_screen_v1 \
  --topk 500 \
  --diverse_k 100 \
  --cluster_method butina \
  --cluster_threshold 0.65 \
  --risk_control true \
  --score_threshold 7.0 \
  --uncertainty_threshold 0.25 \
  --ad_threshold 0.35 \
  --cns_mpo_threshold 4.0 \
  --svg --font "Times New Roman" --bold_text --palette nature5 \
  --font_title 16 --font_label 14 --font_tick 12 --font_legend 12
```

### Style contract (mandatory)

All Step 13 figures are SVG-only with editable text (`savefig.format="svg"`, `svg.fonttype="none"`), use Times New Roman, force bold text, and use Nature palette.

## Step 14 — Screening Feature Matching (Shape + Fragments + R-groups + z_inv alignment)

### Objective

Connect screening hits to Step 10 interpretability outputs via shape matching, fragment/functional-group enrichment, z_inv attribution overlap, and two-tier R-group transferability (training scaffold transfer + novel chemotype decomposition).

### Inputs

Required:

- `outputs/screening/<TARGET>/<screen_id>/processed/library_with_props.parquet`
- `outputs/screening/<TARGET>/<screen_id>/ranking/ranked_all.parquet` (or `ranked_cns_like_in_domain.parquet` if present)
- `data/processed/environments/<TARGET>/data/multienv_compound_level.parquet`
- `outputs/interpretability/<TARGET>/<run_id>/shape/shape_descriptors.parquet`
- `outputs/interpretability/<TARGET>/<run_id>/attribution/fragment_attributions.csv`
- `outputs/interpretability/<TARGET>/<run_id>/rgroup/series_scaffolds.csv`

Recommended:

- `outputs/screening_analysis/<TARGET>/<screen_id>/selections/top100_diverse.csv`

### Outputs

`outputs/screening_feature_match/<TARGET>/<screen_id>/`:

- `matched/` (`hits_topk.parquet`, `hits_with_shape.parquet`, `hits_with_fragments.parquet`, `hits_scaffold_mapping.parquet`, `hits_rgroup_transfer.parquet`, `hits_chemotype_clusters.parquet`)
- `reports/` (`shape_shift_report.csv`, `fragment_enrichment_hits_vs_library.csv`, `functional_group_enrichment_hits_vs_library.csv`, `fragment_overlap_with_zinv.csv`, `scaffold_mapping_report.csv`, `rgroup_transferability_report.csv`, `chemotype_summary.csv`, `top_hits_feature_cards.csv`)
- `selections/` (`top50_hits_with_features.csv`, `top100_hits_with_features.csv`, `chemotype_leads.csv`, `chemotype_panels.csv`)
- `figures/` (all SVG)
- `figure_data/` (CSV for each figure)
- `provenance/` (`run_config.json`, `provenance.json`, `environment.txt`)

### Run Step 14

```bash
python scripts/match_screening_features.py \
  --target CHEMBL335 \
  --screen_dir outputs/screening/CHEMBL335/zinc_screen_v1 \
  --screen_analysis_dir outputs/screening_analysis/CHEMBL335/zinc_screen_v1 \
  --train_parquet data/processed/environments/CHEMBL335/data/multienv_compound_level.parquet \
  --interpret_dir outputs/interpretability/CHEMBL335/<RUN_ID> \
  --outdir outputs/screening_feature_match/CHEMBL335/zinc_screen_v1 \
  --hits_source top100_diverse \
  --hits_topk 100 \
  --shape_etkdg_confs 10 --shape_seed 42 --shape_select lowest_uff_energy \
  --fragment_method brics \
  --rgroup_transfer true \
  --scaffold_match exact \
  --chemotype_cluster_method scaffold \
  --svg --font "Times New Roman" --bold_text --palette nature5 \
  --font_title 16 --font_label 14 --font_tick 12 --font_legend 12
```

### Two-tier R-group handling note

- **Tier 1:** map hits to training scaffolds (`exact` or `similarity`) and transfer R-groups using RDKit `RGroupDecomposition`.
- **Tier 2:** cluster unmapped hits by scaffold or Butina and perform within-chemotype decomposition to summarize novel R-group patterns.

## Step 15 — Build manuscript pack

### Objective

Collect all manuscript artifacts from Steps 1–14 into a reproducible pack with canonical names, manifests, provenance, style contract assets, and an auto-generated checklist.

### Inputs

Required:
- `--paper_id`
- `--target`
- `--run_dir outputs/runs/<TARGET>/<split>/<run_id>/`
- `--interpret_dir outputs/interpretability/<TARGET>/<run_id>/...`
- `--robust_dir outputs/robustness/<TARGET>/<robust_id>/...`
- `--cross_endpoint_dir outputs/evaluation_cross_endpoint/<TARGET>/<eval_id>/...`
- `--screen_dir outputs/screening/<TARGET>/<screen_id>/...`
- `--screen_analysis_dir outputs/screening_analysis/<TARGET>/<screen_id>/...`
- `--screen_match_dir outputs/screening_feature_match/<TARGET>/<screen_id>/...`
- `--outdir`

Options:
- `--copy_only true/false`
- `--svg_only true`
- `--export_tables_csv true`
- `--export_tables_xlsx true`
- `--font`, `--bold_text`, `--palette`

### Outputs

```text
outputs/manuscript_pack/<paper_id>/
├─ main_figures/
├─ supp_figures/
├─ main_tables/
├─ supp_tables/
├─ manifests/
│  ├─ figure_manifest.csv
│  ├─ table_manifest.csv
│  ├─ provenance_manifest.json
│  ├─ citations_sources.txt
│  └─ manuscript_checklist.md
├─ assets/
│  ├─ nature_palette.json
│  └─ style_contract.txt
└─ provenance/
   ├─ run_config.json
   ├─ provenance.json
   └─ environment.txt
```

`manuscript_checklist.md` includes:
- Reproducibility Fingerprint (paper_id, target, run_id/paths, git commit, timestamp, key SHA256 hashes).
- Presence checks for all mapped main/supp figures and tables (✅/❌/⚠️).
- Missing artifact section.
- Seeds/splits/ablations section (from run config and robustness indices when available).
- Figure format checks (`.svg`, non-empty, `<text` element heuristic for editable text).

### Run Step 15

```bash
python scripts/build_manuscript_pack.py \
  --paper_id ptp1b_causal_qsar_v1 \
  --target CHEMBL335 \
  --run_dir outputs/runs/CHEMBL335/scaffold_bm/<RUN_ID> \
  --interpret_dir outputs/interpretability/CHEMBL335/<RUN_ID> \
  --robust_dir outputs/robustness/CHEMBL335/robust_v1 \
  --cross_endpoint_dir outputs/evaluation_cross_endpoint/CHEMBL335/inhib_ext_v1 \
  --screen_dir outputs/screening/CHEMBL335/zinc_screen_v1 \
  --screen_analysis_dir outputs/screening_analysis/CHEMBL335/zinc_screen_v1 \
  --screen_match_dir outputs/screening_feature_match/CHEMBL335/zinc_screen_v1 \
  --outdir outputs/manuscript_pack/ptp1b_causal_qsar_v1 \
  --export_tables_csv true \
  --export_tables_xlsx true \
  --copy_only true \
  --svg_only true \
  --font "Times New Roman" --bold_text --palette nature5
```

### Notes on manifests

- `figure_manifest.csv` columns: `fig_id, fig_title, category, source_path, dest_path, step_origin, status, sha256`.
- `table_manifest.csv` columns: `table_id, table_title, category, source_path, dest_path, step_origin, status, sha256`.
- Missing mapped artifacts are recorded as `status=missing` (non-fatal).
- `provenance_manifest.json` records run identifiers, commit hash, tool versions, discovered artifacts, and key input hashes.
