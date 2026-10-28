"""Build the final V11 reliability-aware completed Excel databases.

The script reuses the predictions already produced by the V11 repeated
5-fold cross-fitting run.  It does not fit any model.  For every eligible
missing cell it writes the mean of the five repeat-specific predictions and
recomputes the canonical V11 reliability score:

    R = 1 - [f(NRMSE) + f(sigma_ens_z) + f(D * sigma_ens_z)] / 3
    f(x) = x / (1 + x)

The second worksheet of every delivered workbook has the same shape as the
completed data worksheet.  It contains a score only where a missing cell was
imputed; observed cells and cells that remain unavailable are blank.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.cell import WriteOnlyCell
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DEFAULT_DATA_PATH = (
    ROOT / "input_data" / "merged dataset" /
    "dataset_v10_maccs_drop30_invalidpsmiles.csv"
)
DEFAULT_INPUT_DIR = ROOT / "input_data"
DEFAULT_PREDICTION_DIR = ROOT / "outputs" / "03_missing_predictions"
DEFAULT_OUTPUT_DIR = ROOT / "final_completed_database_v11"
DEFAULT_CACHE_PATH = ROOT / "input_data" / "merged dataset" / "maccs_cache.parquet"

CATEGORIES = [
    "electrical",
    "mechanical",
    "optical",
    "solution",
    "stability",
    "surface",
    "thermal",
    "transport",
]
SMILES_COL = "smiles"
NON_PROPERTY_COLUMNS = {SMILES_COL, "missing_rate"}
N_REPEATS = 5
FORMULA_VERSION = "fraction_equal_distance_sigma_v1"

FORMULA = (
    "R = 1 - [NRMSE/(1+NRMSE) + sigma_ens_z/(1+sigma_ens_z) "
    "+ (D*sigma_ens_z)/(1+D*sigma_ens_z)] / 3"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build V11 completed merged and category Excel files."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument(
        "--prediction-dir", type=Path, default=DEFAULT_PREDICTION_DIR
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--skip-workbook-validation",
        action="store_true",
        help="Skip the post-write score-cell validation (not recommended).",
    )
    return parser.parse_args()


def ensure_new_output_dir(path: Path) -> None:
    """Refuse to mix a new delivery with an existing non-empty directory."""
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(
            f"Output directory is not empty; refusing to overwrite it: {path}"
        )
    path.mkdir(parents=True, exist_ok=True)


def finite_or_none(value: object) -> object:
    """Convert NumPy scalars and NaN/inf values to Excel-safe Python values."""
    if value is None or value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    return value


def safe_fraction(values: np.ndarray, name: str) -> np.ndarray:
    """Evaluate x/(1+x), taking the mathematically correct limit at +inf."""
    values = np.asarray(values, dtype=float)
    if np.any(np.isnan(values)) or np.any(np.isneginf(values)):
        raise ValueError(f"{name} contains NaN or negative infinity")
    if np.any(values < 0):
        minimum = float(np.min(values))
        raise ValueError(f"{name} contains a negative risk value: {minimum}")
    result = np.empty(values.shape, dtype=float)
    finite = np.isfinite(values)
    result[finite] = values[finite] / (1.0 + values[finite])
    result[np.isposinf(values)] = 1.0
    return result


def load_category_headers(
    input_dir: Path, data: pd.DataFrame
) -> tuple[dict[str, list[str]], dict[str, str], dict[str, dict[str, object]]]:
    """Read category workbook headers and verify the shared row convention."""
    category_headers: dict[str, list[str]] = {}
    property_category: dict[str, str] = {}
    input_metadata: dict[str, dict[str, object]] = {}

    for category in CATEGORIES:
        path = input_dir / f"{category}.xlsx"
        if not path.exists():
            raise FileNotFoundError(path)
        workbook = load_workbook(path, read_only=True, data_only=True)
        try:
            if not workbook.sheetnames:
                raise ValueError(f"Workbook has no worksheets: {path}")
            worksheet = workbook[workbook.sheetnames[0]]
            header_rows = list(
                worksheet.iter_rows(
                    min_row=1, max_row=1, values_only=True
                )
            )
            if not header_rows:
                raise ValueError(f"Workbook has no header: {path}")
            headers = [str(value) for value in header_rows[0]]
            if headers[0] != SMILES_COL:
                raise ValueError(
                    f"First column of {path.name} is {headers[0]!r}, "
                    f"expected {SMILES_COL!r}"
                )
            if len(headers) != len(set(headers)):
                raise ValueError(f"Duplicate headers in {path}: {headers}")
            if worksheet.max_row != len(data) + 1:
                raise ValueError(
                    f"Row mismatch in {path.name}: Excel has "
                    f"{worksheet.max_row - 1} data rows, CSV has {len(data)}"
                )
            first_row = next(
                worksheet.iter_rows(
                    min_row=2, max_row=2, min_col=1, max_col=1,
                    values_only=True
                )
            )[0]
            last_row = next(
                worksheet.iter_rows(
                    min_row=worksheet.max_row,
                    max_row=worksheet.max_row,
                    min_col=1,
                    max_col=1,
                    values_only=True,
                )
            )[0]
            if str(first_row) != str(data.iloc[0][SMILES_COL]) or str(last_row) != str(data.iloc[-1][SMILES_COL]):
                raise ValueError(
                    f"SMILES row-order mismatch in {path.name}; refusing to write"
                )
            unknown = [column for column in headers if column not in data.columns]
            if unknown:
                raise ValueError(
                    f"Columns in {path.name} are absent from merged CSV: {unknown}"
                )
            category_headers[category] = headers
            input_metadata[category] = {
                "path": str(path.resolve()),
                "sheet": worksheet.title,
                "rows": int(worksheet.max_row - 1),
                "columns": int(len(headers)),
            }
            for column in headers:
                if column in NON_PROPERTY_COLUMNS:
                    continue
                previous = property_category.get(column)
                if previous is not None and previous != category:
                    raise ValueError(
                        f"Property {column!r} occurs in both {previous} and {category}"
                    )
                property_category[column] = category
        finally:
            workbook.close()

    expected_properties = [
        column for column in data.columns if column not in NON_PROPERTY_COLUMNS
    ]
    missing_category = sorted(set(expected_properties) - set(property_category))
    extra_category = sorted(set(property_category) - set(expected_properties))
    if missing_category or extra_category:
        raise ValueError(
            "Category/merged property mismatch: "
            f"missing={missing_category}, extra={extra_category}"
        )
    return category_headers, property_category, input_metadata


def load_valid_fingerprint_mask(data: pd.DataFrame, cache_path: Path) -> np.ndarray:
    if cache_path.exists():
        try:
            cache = pd.read_parquet(cache_path)
            if len(cache) != len(data) or "maccs_valid" not in cache.columns:
                raise ValueError("MACCS cache does not align with the merged CSV")
            return cache["maccs_valid"].astype(bool).to_numpy()
        except (ImportError, ModuleNotFoundError):
            # The local assembly environment may lack a parquet engine even
            # though the AutoDL run already produced a valid cache.  Rebuild
            # only the validity mask with the existing RDKit helper.
            pass
    pickle_path = cache_path.with_suffix(".pkl")
    if pickle_path.exists():
        cache = pd.read_pickle(pickle_path)
        if len(cache) != len(data) or "maccs_valid" not in cache.columns:
            raise ValueError("MACCS cache does not align with the merged CSV")
        return cache["maccs_valid"].astype(bool).to_numpy()
    sys.path.insert(0, str(SCRIPT_DIR))
    try:
        from workflow_common import compute_maccs_fingerprints

        fingerprints = compute_maccs_fingerprints(data, SMILES_COL)
    except Exception as exc:
        raise RuntimeError(
            "Could not read the MACCS cache or rebuild its validity mask"
        ) from exc
    return fingerprints["maccs_valid"].astype(bool).to_numpy()


def prediction_columns(frame: pd.DataFrame, prefix: str) -> list[str]:
    columns = [f"{prefix}{index}" for index in range(N_REPEATS)]
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing prediction columns: {missing}")
    return columns


def validate_and_score_property(
    property_name: str,
    category: str,
    data: pd.DataFrame,
    prediction_dir: Path,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[str, object]]:
    """Load one V11 missing file, validate it, and calculate final scores."""
    path = prediction_dir / f"{property_name}_missing_reliability.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {
        "polymer_id",
        "y_imputed_mean",
        "y_imputed_std",
        "n_valid_predictions",
        "property_nrmse",
        "nrmse_risk_fraction",
        "ensemble_std_normalized",
        "ensemble_risk_fraction",
        "tanimoto_distance",
        "distance_sigma_risk",
        "distance_sigma_risk_fraction",
        "reliability_formula_version",
        "reliability_score",
        "local_mean",
    }
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(
            f"{property_name}: missing required reliability columns {missing_columns}"
        )

    y_columns = prediction_columns(frame, "y_pred_repeat")
    fold_columns = prediction_columns(frame, "fold_id_repeat")
    ids_float = pd.to_numeric(frame["polymer_id"], errors="coerce").to_numpy(float)
    if not np.isfinite(ids_float).all() or not np.equal(ids_float, ids_float.astype(int)).all():
        raise ValueError(f"{property_name}: polymer_id is not an integer index")
    ids = ids_float.astype(np.int64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"{property_name}: duplicate polymer_id values")
    if np.any(ids < 0) or np.any(ids >= len(data)):
        raise ValueError(f"{property_name}: polymer_id is outside the input row range")

    original_values = data.iloc[ids][property_name].to_numpy()
    if not pd.isna(original_values).all():
        raise ValueError(f"{property_name}: a prediction row targets an observed cell")

    predictions = frame[y_columns].to_numpy(dtype=float)
    if not np.isfinite(predictions).all():
        raise ValueError(f"{property_name}: at least one repeat prediction is non-finite")
    n_valid = pd.to_numeric(frame["n_valid_predictions"], errors="coerce").to_numpy(float)
    if not np.all(n_valid == N_REPEATS):
        raise ValueError(f"{property_name}: not every cell has five valid predictions")

    imputed_mean = np.mean(predictions, axis=1)
    imputed_std = np.std(predictions, axis=1, ddof=1)
    saved_mean = frame["y_imputed_mean"].to_numpy(dtype=float)
    saved_std = frame["y_imputed_std"].to_numpy(dtype=float)
    if not np.allclose(imputed_mean, saved_mean, rtol=1e-10, atol=1e-10):
        raise ValueError(f"{property_name}: saved mean differs from repeat mean")
    if not np.allclose(imputed_std, saved_std, rtol=1e-10, atol=1e-10):
        raise ValueError(f"{property_name}: saved SD differs from repeat SD")

    nrmse = frame["property_nrmse"].to_numpy(dtype=float)
    if not np.isfinite(nrmse).all() or np.any(nrmse < 0):
        raise ValueError(f"{property_name}: invalid property NRMSE")
    if not np.allclose(nrmse, nrmse[0], rtol=1e-12, atol=1e-12):
        raise ValueError(f"{property_name}: property NRMSE is not constant")
    metadata_path = (
        prediction_dir.parent
        / "01_oof_predictions"
        / f"{property_name}_metadata.json"
    )
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    observed_std = float(metadata["observed_std"])
    if not np.isfinite(observed_std) or observed_std <= 0:
        raise ValueError(f"{property_name}: invalid observed standard deviation")

    # The canonical score uses the property-level NRMSE, repeated-prediction
    # dispersion, and its interaction with fold-aligned structural distance.
    sigma = imputed_std / observed_std
    saved_sigma = frame["ensemble_std_normalized"].to_numpy(dtype=float)
    if not np.allclose(sigma, saved_sigma, rtol=1e-8, atol=1e-10):
        raise ValueError(
            f"{property_name}: stored ensemble_std_normalized does not match "
            "the five-repeat SD and observed SD"
        )
    if not np.isfinite(sigma).all() or np.any(sigma < 0):
        raise ValueError(f"{property_name}: invalid normalized ensemble SD")

    distance = frame["tanimoto_distance"].to_numpy(dtype=float)
    local_mean = frame["local_mean"].to_numpy(dtype=float)
    if not np.isfinite(distance).all() or np.any(distance < 0):
        raise ValueError(f"{property_name}: invalid fold-aligned Tanimoto distance")
    if not np.isfinite(local_mean).all():
        raise ValueError(f"{property_name}: invalid fold-aligned local mean")

    nrmse_fraction = safe_fraction(nrmse, f"{property_name} NRMSE")
    sigma_fraction = safe_fraction(sigma, f"{property_name} sigma")
    distance_sigma = distance * sigma
    distance_sigma_fraction = safe_fraction(
        distance_sigma, f"{property_name} distance*sigma"
    )
    scores = 1.0 - (
        nrmse_fraction + sigma_fraction + distance_sigma_fraction
    ) / 3.0
    saved_nrmse_fraction = frame["nrmse_risk_fraction"].to_numpy(dtype=float)
    saved_sigma_fraction = frame["ensemble_risk_fraction"].to_numpy(dtype=float)
    saved_distance_sigma = frame["distance_sigma_risk"].to_numpy(dtype=float)
    saved_distance_sigma_fraction = frame[
        "distance_sigma_risk_fraction"
    ].to_numpy(dtype=float)
    saved_version = frame["reliability_formula_version"].astype(str)
    if set(saved_version) != {FORMULA_VERSION}:
        raise ValueError(
            f"{property_name}: reliability file uses an unsupported formula version"
        )
    if not np.allclose(saved_nrmse_fraction, nrmse_fraction, rtol=1e-8, atol=1e-10):
        raise ValueError(f"{property_name}: stored NRMSE risk fraction is inconsistent")
    if not np.allclose(saved_sigma_fraction, sigma_fraction, rtol=1e-8, atol=1e-10):
        raise ValueError(f"{property_name}: stored ensemble risk fraction is inconsistent")
    if not np.allclose(saved_distance_sigma, distance_sigma, rtol=1e-8, atol=1e-10):
        raise ValueError(f"{property_name}: stored distance*sigma is inconsistent")
    if not np.allclose(
        saved_distance_sigma_fraction,
        distance_sigma_fraction,
        rtol=1e-8,
        atol=1e-10,
    ):
        raise ValueError(
            f"{property_name}: stored distance*sigma risk fraction is inconsistent"
        )
    saved_scores = frame["reliability_score"].to_numpy(dtype=float)
    if not np.allclose(saved_scores, scores, rtol=1e-8, atol=1e-10):
        raise ValueError(
            f"{property_name}: stored reliability score does not match canonical formula"
        )
    if not np.isfinite(scores).all() or np.any(scores < -1e-12) or np.any(scores > 1 + 1e-12):
        raise ValueError(f"{property_name}: final score is outside [0, 1]")
    scores = np.clip(scores, 0.0, 1.0)

    value_array = np.full(len(data), np.nan, dtype=float)
    score_array = np.full(len(data), np.nan, dtype=float)
    value_array[ids] = imputed_mean
    score_array[ids] = scores

    audit_columns = [
        "polymer_id",
        "y_imputed_std",
        "property_nrmse",
        "ensemble_std_normalized",
        "tanimoto_distance",
        "local_mean",
        "structure_local_risk",
        "reliability_formula_version",
    ] + fold_columns + y_columns
    audit = frame[audit_columns].copy()
    audit = audit.rename(
        columns={
            "y_imputed_std": "prediction_sd",
            "ensemble_std_normalized": "sigma_ens_z",
            "structure_local_risk": "G_v11_legacy",
            "tanimoto_distance": "D_fold_aligned",
            "local_mean": "local_mean_fold_aligned",
        }
    )
    audit.insert(1, "smiles", data.iloc[ids][SMILES_COL].to_numpy())
    audit.insert(2, "property", property_name)
    audit.insert(3, "category", category)
    audit.insert(4, "imputed_value", imputed_mean)
    audit.insert(5, "reliability_score", scores)
    audit.insert(6, "nrmse_fraction", nrmse_fraction)
    audit.insert(7, "sigma_fraction", sigma_fraction)
    audit.insert(8, "D_sigma", distance_sigma)
    audit.insert(9, "D_sigma_fraction", distance_sigma_fraction)

    summary = {
        "property": property_name,
        "category": category,
        "n_input_observed": int(data[property_name].notna().sum()),
        "n_input_missing": int(data[property_name].isna().sum()),
        "n_imputed": int(len(ids)),
        "n_scores": int(np.isfinite(scores).sum()),
        "score_mean": float(np.mean(scores)),
        "score_median": float(np.median(scores)),
        "score_q05": float(np.quantile(scores, 0.05)),
        "score_q25": float(np.quantile(scores, 0.25)),
        "score_q75": float(np.quantile(scores, 0.75)),
        "score_q95": float(np.quantile(scores, 0.95)),
        "score_min": float(np.min(scores)),
        "score_max": float(np.max(scores)),
        "property_nrmse": float(nrmse[0]),
        "observed_std": observed_std,
        "D_sigma_median": float(np.median(distance_sigma)),
        "D_sigma_q95": float(np.quantile(distance_sigma, 0.95)),
    }
    return {"values": value_array, "scores": score_array}, audit, summary


def write_audit_chunk(
    audit: pd.DataFrame, handle: gzip.GzipFile, first_chunk: bool
) -> None:
    audit.to_csv(handle, index=False, header=first_chunk, lineterminator="\n")


def make_header_cells(worksheet, headers: list[str]) -> None:
    fill = PatternFill(fill_type="solid", fgColor="1F4E78")
    font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    alignment = Alignment(horizontal="center", vertical="center")
    cells = []
    for header in headers:
        cell = WriteOnlyCell(worksheet, value=header)
        cell.fill = fill
        cell.font = font
        cell.alignment = alignment
        cells.append(cell)
    worksheet.append(cells)


def write_workbook(
    path: Path,
    data: pd.DataFrame,
    headers: list[str],
    value_arrays: dict[str, np.ndarray],
    score_arrays: dict[str, np.ndarray],
) -> None:
    """Stream a two-sheet workbook to keep the 33k-row delivery lightweight."""
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook(write_only=True)
    workbook.properties.creator = "NextPoly V11"
    workbook.properties.title = path.stem
    data_sheet = workbook.create_sheet("Sheet1")
    score_sheet = workbook.create_sheet("Sheet2")
    for worksheet in (data_sheet, score_sheet):
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = (
            f"A1:{get_column_letter(len(headers))}{len(data) + 1}"
        )
        for index, header in enumerate(headers, start=1):
            width = 22 if header == SMILES_COL else min(max(len(header) + 2, 12), 28)
            worksheet.column_dimensions[get_column_letter(index)].width = width
        make_header_cells(worksheet, headers)

    positions = {column: index for index, column in enumerate(data.columns)}
    selected_positions = [positions[column] for column in headers]
    for row_index, row in enumerate(data.itertuples(index=False, name=None)):
        completed_row = []
        score_row = []
        for column, position in zip(headers, selected_positions):
            original = row[position]
            value = original
            if column in value_arrays and pd.isna(original):
                candidate = value_arrays[column][row_index]
                if np.isfinite(candidate):
                    value = float(candidate)
            completed_row.append(finite_or_none(value))

            if column == SMILES_COL:
                score_value = original
            elif column in score_arrays:
                candidate = score_arrays[column][row_index]
                score_value = float(candidate) if np.isfinite(candidate) else None
            else:
                score_value = None
            score_row.append(finite_or_none(score_value))
        data_sheet.append(completed_row)
        score_sheet.append(score_row)
    workbook.save(path)


def validate_score_sheet(
    path: Path,
    headers: list[str],
    expected_scores: dict[str, np.ndarray],
    n_rows: int,
) -> dict[str, object]:
    """Count every score cell after writing and compare it with memory."""
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        if workbook.sheetnames[:2] != ["Sheet1", "Sheet2"]:
            raise ValueError(f"Unexpected sheet names in {path}: {workbook.sheetnames}")
        worksheet = workbook["Sheet2"]
        rows = worksheet.iter_rows(values_only=True)
        header = list(next(rows))
        if header != headers:
            raise ValueError(f"Sheet2 header mismatch in {path}")
        counts = {column: 0 for column in headers}
        row_count = 0
        for row in rows:
            row_count += 1
            for index, (column, value) in enumerate(zip(headers, row)):
                if column == SMILES_COL:
                    continue
                if value is not None and value != "":
                    counts[column] += 1
        if row_count != n_rows:
            raise ValueError(
                f"Sheet2 row mismatch in {path}: {row_count} vs {n_rows}"
            )
        expected_counts = {
            column: int(np.isfinite(expected_scores[column]).sum())
            if column in expected_scores
            else 0
            for column in headers
        }
        if counts != expected_counts:
            raise ValueError(
                f"Score-cell count mismatch in {path}: actual={counts}, "
                f"expected={expected_counts}"
            )
        return {"rows": row_count, "score_cells": int(sum(counts.values()))}
    finally:
        workbook.close()


def write_unfilled_summary(
    path: Path,
    data: pd.DataFrame,
    value_arrays: dict[str, np.ndarray],
    valid_fp: np.ndarray,
    property_category: dict[str, str],
    success_properties: set[str],
) -> None:
    rows = []
    properties = [
        column for column in data.columns if column not in NON_PROPERTY_COLUMNS
    ]
    for property_name in properties:
        missing = data[property_name].isna().to_numpy()
        filled = np.isfinite(value_arrays[property_name]) if property_name in value_arrays else np.zeros(len(data), dtype=bool)
        unfilled = missing & ~filled
        selected = property_name in success_properties
        rows.append(
            {
                "property": property_name,
                "category": property_category[property_name],
                "status": "imputed" if selected else "observed_only",
                "n_observed": int((~missing).sum()),
                "n_missing": int(missing.sum()),
                "n_imputed": int(filled.sum()),
                "n_unfilled": int(unfilled.sum()),
                "n_unfilled_invalid_maccs": int((unfilled & ~valid_fp).sum()),
                "n_unfilled_other": int((unfilled & valid_fp).sum()),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def write_readme(
    path: Path,
    manifest: dict[str, object],
) -> None:
    text = f"""NextPoly V11 final completed database

This delivery reuses the predictions produced by the V11 strict repeated
5-fold cross-fitting run. No model was retrained during database assembly.

Completed value for each eligible missing cell:
  mean of the five repeat-specific, preassigned-fold predictions

Final reliability formula:
  {FORMULA}

For each property, sigma_ens_z is the five-repeat prediction SD divided by
the observed-property sample SD.  D_fold_aligned is the fold-restricted
MACCS-Tanimoto distance and D_sigma is their product.  The stored
structure_local_risk value, local mean, and local consistency fields are
retained as diagnostic audit fields and are not used in the final score.

The three risk fractions are evaluated on their non-negative inputs. When an
input is +infinity, its fraction is assigned its mathematical limit of 1.
This occurs when a local reference set has zero standard deviation and the
prediction differs from its local mean.

Workbook layout:
  Sheet1: original observed values plus eligible V11 imputations
  Sheet2: same row/column layout; scores only at imputed cells
           (observed and unfilled cells are blank)

The original V11 prediction and detailed fold-aligned risk CSV files remain in
the parent outputs directory. This folder also contains a compressed audit
table, per-property summaries, and the unfilled-cell summary.

Rows are aligned by zero-based polymer_id in the V11 CSV files. In Excel,
polymer_id 0 is worksheet row 2.

Rows: {manifest['n_rows']}
Properties in merged input: {manifest['n_properties']}
Properties imputed: {manifest['n_imputed_properties']}
Imputed cells: {manifest['n_imputed_cells']}
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    data_path = args.data_path.resolve()
    input_dir = args.input_dir.resolve()
    prediction_dir = args.prediction_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    if not prediction_dir.exists():
        raise FileNotFoundError(prediction_dir)

    data = pd.read_csv(data_path)
    if SMILES_COL not in data.columns:
        raise ValueError(f"Merged CSV lacks {SMILES_COL!r}")
    category_headers, property_category, input_metadata = load_category_headers(
        input_dir, data
    )
    valid_fp = load_valid_fingerprint_mask(data, data_path.parent / "maccs_cache.parquet")

    ensure_new_output_dir(output_dir)
    audit_dir = output_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / "imputed_reliability_scores.csv.gz"
    summary_rows: list[dict[str, object]] = []
    value_arrays: dict[str, np.ndarray] = {}
    score_arrays: dict[str, np.ndarray] = {}
    success_properties: list[str] = []
    first_audit_chunk = True

    print(f"Input rows={len(data)}, properties={len(data.columns) - len(NON_PROPERTY_COLUMNS)}")
    print(f"Prediction directory={prediction_dir}")
    print(f"Output directory={output_dir}")
    with gzip.open(audit_path, "wt", encoding="utf-8", newline="") as audit_handle:
        for property_name in data.columns:
            if property_name in NON_PROPERTY_COLUMNS:
                continue
            reliability_path = prediction_dir / f"{property_name}_missing_reliability.csv"
            if not reliability_path.exists():
                continue
            category = property_category[property_name]
            scored, audit, summary = validate_and_score_property(
                property_name, category, data, prediction_dir
            )
            value_arrays[property_name] = scored["values"]
            score_arrays[property_name] = scored["scores"]
            write_audit_chunk(audit, audit_handle, first_audit_chunk)
            first_audit_chunk = False
            summary_rows.append(summary)
            success_properties.append(property_name)
            print(
                f"  {property_name}: imputed={summary['n_imputed']}, "
                f"score median={summary['score_median']:.6f}",
                flush=True,
            )

    if not success_properties:
        raise RuntimeError("No V11 missing reliability files were found")

    # Every merged property gets an array so downstream writing can use one
    # uniform lookup path; skipped properties remain all-NaN.
    for property_name in data.columns:
        if property_name in NON_PROPERTY_COLUMNS:
            continue
        value_arrays.setdefault(property_name, np.full(len(data), np.nan))
        score_arrays.setdefault(property_name, np.full(len(data), np.nan))

    merged_headers = list(data.columns)
    workbook_validation: dict[str, object] = {}
    merged_path = output_dir / "PolyDB_v11_completed_merged.xlsx"
    write_workbook(
        merged_path, data, merged_headers, value_arrays, score_arrays
    )
    if not args.skip_workbook_validation:
        workbook_validation[merged_path.name] = validate_score_sheet(
            merged_path, merged_headers, score_arrays, len(data)
        )

    category_paths: dict[str, str] = {}
    for category in CATEGORIES:
        path = output_dir / f"{category}_v11_completed.xlsx"
        headers = category_headers[category]
        write_workbook(path, data, headers, value_arrays, score_arrays)
        if not args.skip_workbook_validation:
            workbook_validation[path.name] = validate_score_sheet(
                path, headers, score_arrays, len(data)
            )
        category_paths[category] = str(path.resolve())

    summary_path = audit_dir / "property_summary.csv"
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    unfilled_path = audit_dir / "unfilled_cells_summary.csv"
    write_unfilled_summary(
        unfilled_path,
        data,
        value_arrays,
        valid_fp,
        property_category,
        set(success_properties),
    )

    imputed_count = int(sum(np.isfinite(value_arrays[p]).sum() for p in success_properties))
    input_missing_success = int(sum(data[p].isna().sum() for p in success_properties))
    skipped_properties = [
        p for p in data.columns
        if p not in NON_PROPERTY_COLUMNS and p not in success_properties
    ]
    manifest: dict[str, object] = {
        "version": "v11_final_completed_database",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "formula": FORMULA,
        "formula_version": FORMULA_VERSION,
        "formula_note": "positive infinity maps to the limit 1 in x/(1+x)",
        "input_data_path": str(data_path),
        "v11_prediction_dir": str(prediction_dir),
        "n_rows": int(len(data)),
        "n_properties": int(len(data.columns) - len(NON_PROPERTY_COLUMNS)),
        "n_imputed_properties": int(len(success_properties)),
        "n_observed_only_properties": int(len(skipped_properties)),
        "n_imputed_cells": imputed_count,
        "n_input_missing_cells_in_imputed_properties": input_missing_success,
        "n_valid_maccs_rows": int(valid_fp.sum()),
        "n_invalid_maccs_rows": int((~valid_fp).sum()),
        "imputation_rule": "mean of five V11 repeat-specific preassigned-fold predictions",
        "polymer_id_rule": "zero-based row index in the merged input CSV",
        "imputed_properties": success_properties,
        "observed_only_properties": skipped_properties,
        "input_workbooks": input_metadata,
        "files": {
            "merged_workbook": str(merged_path.resolve()),
            "category_workbooks": category_paths,
            "reliability_audit_gzip": str(audit_path.resolve()),
            "property_summary": str(summary_path.resolve()),
            "unfilled_cells_summary": str(unfilled_path.resolve()),
        },
        "workbook_validation": workbook_validation,
    }
    manifest_path = output_dir / "final_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_readme(output_dir / "README.txt", manifest)

    print("=" * 80)
    print(json.dumps({
        "status": "ok",
        "imputed_properties": len(success_properties),
        "imputed_cells": imputed_count,
        "skipped_properties": len(skipped_properties),
        "merged_workbook": str(merged_path),
        "output_dir": str(output_dir),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
