# PolyDB-PSI

PolyDB-PSI is a reproducible reliability-validation workflow for polymer property databases. It combines MACCS molecular fingerprints with TabPFN regression to produce strictly out-of-fold predictions, then converts those predictions into a structure-aware reliability score for both observed and genuinely missing polymer-property cells.

The repository is designed for researchers who need to audit a polymer database, estimate missing properties, and identify predictions that are supported by nearby chemical structures. It contains the executable workflow, the mathematical implementation, configuration, and the results that have already been generated. Raw input tables are intentionally excluded from this public repository; provide your own compatible dataset and model checkpoint when running the workflow.

## What the workflow does

1. **Scope selection.** The configuration classifies properties, applies minimum-observation rules, and decides which targets are eligible for imputation.
2. **Fingerprint generation.** SMILES strings are converted to 167-bit MACCS fingerprints. A local Parquet or pickle cache may be used when available.
3. **Five-repeat cross-fitting.** For every eligible property, `01_train_repeated5fold_cv.py` runs five independent five-fold partitions with repeat seeds `42, 43, 44, 45, 46`. Each observed cell is predicted only by the model whose training fold did not contain that cell. The result is five strict OOF predictions per observed cell, rather than five predictions from models that may have seen its label.
4. **PSI/reliability calculation.** `02_compute_reliability.py` combines property-level NRMSE, prediction dispersion, and MACCS-Tanimoto structural distance. The implementation in `reliability_math.py` computes local consistency and distance risks using only the corresponding training-fold evidence.
5. **Audit-ready outputs.** CSV and JSON files preserve repeat-level predictions, fold assignments, errors, structural distances, local-neighbour statistics, and final reliability scores.

The score is based on three risk components:

```text
R_i = 1 - [NRMSE_j/(1+NRMSE_j)
           + sigma_i/(1+sigma_i)
           + (D_i*sigma_i)/(1+D_i*sigma_i)] / 3
```

`R_i` is the reliability score for cell `i`; `D_i` is the distance from the corresponding training-fold structures and `sigma_i` is the normalized dispersion of the five predictions. Higher scores indicate stronger support. The exact implementation, including edge-case handling and `ddof=1` standard deviations, is in `scripts/reliability_math.py`.

## Repository layout

```text
config/workflow_config.py              # paths, target scope, seeds, thresholds
scripts/01_train_repeated5fold_cv.py  # five-repeat, five-fold TabPFN OOF prediction
scripts/02_compute_reliability.py     # PSI/reliability calculation
scripts/reliability_math.py            # score and neighbour-risk mathematics
scripts/workflow_common.py             # shared data and feature utilities
scripts/run_all.py                     # resumable end-to-end entry point
outputs/01_oof_predictions/             # observed-cell OOF predictions and metadata
outputs/02_oof_reliability_scores/      # reliability/PSI results
outputs/03_missing_predictions/         # missing-cell predictions when available
requirements.txt                         # Python dependencies
```

## Installation

Use Python 3.11 or 3.12. A CUDA-enabled PyTorch installation is recommended for the full dataset, although the scripts support CPU execution for smoke tests and small runs.

```bash
git clone https://github.com/gebojun/PolyDB-PSI.git
cd PolyDB-PSI
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install a compatible CUDA build of PyTorch before `tabpfn` when GPU execution is required. Verify the environment without starting a full training run:

```bash
python scripts/00_check_environment.py
python scripts/00_smoke_test_cpu_150rows.py
```

## Prepare data locally

The public repository does not contain `input_data/`. Supply a merged CSV with at least:

- a `smiles` column;
- numeric property columns matching the names used in `config/workflow_config.py`;
- optional MACCS cache at `<dataset directory>/maccs_cache.parquet` or `maccs_cache.pkl`.

Also provide a TabPFN checkpoint, for example `tabpfn-v3-regressor-v3_default.ckpt`. The default paths are documented in `config/workflow_config.py`; for a different layout, pass `--data-path`, `--checkpoint-path`, and `--output-dir` to the training script. Do not commit private, restricted, or sensitive source data.

## Run the five-repeat TabPFN prediction

Run a small test first:

```bash
python scripts/01_train_repeated5fold_cv.py \
  --data-path /path/to/merged_dataset.csv \
  --checkpoint-path /path/to/tabpfn-v3-regressor-v3_default.ckpt \
  --output-dir outputs \
  --properties Tg_K dielectric_constant thermal_cond_W_mK \
  --test-mode --device cpu
```

For the full run, use CUDA when available and keep the default five repeats and five folds:

```bash
python scripts/01_train_repeated5fold_cv.py \
  --data-path /path/to/merged_dataset.csv \
  --checkpoint-path /path/to/tabpfn-v3-regressor-v3_default.ckpt \
  --output-dir outputs --device cuda --skip-existing
```

The script writes one `{property}_all_folds.csv` and one `{property}_metadata.json` per target. The CSV contains `y_true`, the ensemble mean and standard deviation, fold identifiers, and `y_pred_repeat0` through `y_pred_repeat4`. `--skip-existing` makes interrupted runs resumable.

## Run the PSI/reliability calculation

After the OOF predictions exist, run:

```bash
python scripts/02_compute_reliability.py \
  --data-path /path/to/merged_dataset.csv \
  --output-dir outputs --skip-existing
```

This reads the OOF predictions and metadata, matches each repeat to its training-fold structures, searches MACCS-Tanimoto neighbours, and writes:

- `outputs/02_oof_reliability_scores/{property}_oof_reliability.csv` for observed cells;
- `outputs/03_missing_predictions/{property}_missing_reliability.csv` for genuine missing cells;
- repeat-level distance, local-consistency, and dispersion columns for auditability.

To run both stages and resume automatically:

```bash
python scripts/run_all.py --device cuda --skip-environment-check
```

Use `--properties ...` for a focused run, `--test-mode` for the representative smoke set, and `--skip-training` or `--skip-reliability` when only one stage is required. Plot scripts are optional and are not needed to reproduce the numerical PSI/reliability tables.

## Reproducibility notes

- Repeat seeds are fixed to `42, 43, 44, 45, 46` and each repeat uses five folds.
- Missing cells receive deterministic fold assignments based on SMILES, property name, and repeat seed.
- The target property itself is excluded from its feature matrix.
- No external imputation or fixed absolute epsilon is added to the reliability formula.
- Results should be interpreted together with metadata, observed sample size, and structural distance; a reliability score is an evidence measure, not a guarantee of experimental accuracy.

## License and data

Add the license and data-use terms required by your project before redistributing private source tables or model checkpoints. The code and generated outputs in this repository should be used in accordance with the licenses of TabPFN, PyTorch, RDKit, and all upstream datasets.
