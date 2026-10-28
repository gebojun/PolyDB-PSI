"""Compute V11 OOF and true-missing reliability from fold-matched evidence."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_DIR = SCRIPT_DIR.parent / "config"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(CONFIG_DIR))

from reliability_math import (
    aggregate_repeat_risk,
    local_risk_from_candidates,
    normalize_by_observed_std,
    reliability_score,
    top_tanimoto_candidates,
)
from workflow_common import compute_maccs_fingerprints, ensure_dir, save_json
from workflow_config import (
    DEFAULT_OUTPUT_DIR,
    FP_COLS,
    K_NEAREST_NEIGHBORS,
    NEIGHBOR_CANDIDATE_MULTIPLIER,
    SMILES_COL,
    SOURCE_MERGED_DATASET,
    VERSION,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V11 reliability calculation")
    parser.add_argument("--data-path", type=Path, default=SOURCE_MERGED_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--properties", nargs="+", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--query-chunk-size", type=int, default=128)
    parser.add_argument(
        "--candidate-multiplier",
        type=int,
        default=NEIGHBOR_CANDIDATE_MULTIPLIER,
    )
    return parser.parse_args()


def load_fingerprints(data: pd.DataFrame, data_path: Path) -> pd.DataFrame:
    for path, reader in (
        (data_path.parent / "maccs_cache.parquet", pd.read_parquet),
        (data_path.parent / "maccs_cache.pkl", pd.read_pickle),
    ):
        if not path.exists():
            continue
        try:
            cached = reader(path)
            if len(cached) == len(data) and set(FP_COLS).issubset(cached.columns):
                cached.index = data.index
                return cached
        except Exception:
            pass
    return compute_maccs_fingerprints(data, SMILES_COL)


def atomic_to_csv(frame: pd.DataFrame, destination: Path) -> None:
    ensure_dir(destination.parent)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(destination)


def find_candidate_neighbors(
    query_fps: np.ndarray,
    reference_fps: np.ndarray,
    k: int,
    candidate_multiplier: int,
    query_chunk_size: int,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    candidate_count = min(
        len(reference_fps),
        max(k, candidate_multiplier * k),
    )
    last_report = [-1]

    def progress(done: int, total: int) -> None:
        decile = int(10 * done / max(total, 1))
        if decile != last_report[0]:
            last_report[0] = decile
            print(f"    {label}: candidate search {done}/{total}", flush=True)

    return top_tanimoto_candidates(
        query_fps=query_fps,
        reference_fps=reference_fps,
        n_candidates=candidate_count,
        query_chunk_size=query_chunk_size,
        progress_callback=progress,
    )


def write_repeat_columns(
    result: pd.DataFrame,
    components: list[dict[str, np.ndarray]],
) -> None:
    short_names = {
        "mean_tanimoto_similarity": "mean_tanimoto_similarity",
        "tanimoto_distance": "D",
        "local_consistency_risk": "A",
        "structure_local_risk": "G",
    }
    for repeat_id, repeat in enumerate(components):
        for key, prefix in short_names.items():
            result[f"{prefix}_repeat{repeat_id}"] = repeat[key]


def compute_property(
    property_name: str,
    data: pd.DataFrame,
    fp_matrix: np.ndarray,
    output_dir: Path,
    skip_existing: bool,
    candidate_multiplier: int,
    query_chunk_size: int,
) -> dict:
    started = time.time()
    oof_dir = output_dir / "01_oof_predictions"
    reliability_dir = output_dir / "02_oof_reliability_scores"
    missing_dir = output_dir / "03_missing_predictions"
    oof_path = oof_dir / f"{property_name}_all_folds.csv"
    metadata_path = oof_dir / f"{property_name}_metadata.json"
    missing_path = missing_dir / f"{property_name}_missing_predictions.csv"
    oof_output = reliability_dir / f"{property_name}_oof_reliability.csv"
    missing_output = missing_dir / f"{property_name}_missing_reliability.csv"

    if skip_existing and oof_output.exists() and missing_output.exists():
        try:
            existing_columns = set(pd.read_csv(oof_output, nrows=0).columns)
            missing_columns = set(pd.read_csv(missing_output, nrows=0).columns)
        except Exception:
            existing_columns = set()
            missing_columns = set()
        required = {
            "reliability_formula_version",
            "nrmse_risk_fraction",
            "ensemble_risk_fraction",
            "distance_sigma_risk",
            "distance_sigma_risk_fraction",
            "structure_local_risk",
            "local_consistency_risk",
            "D_repeat0",
            "A_repeat0",
            "G_repeat0",
        }
        if required.issubset(existing_columns) and required.issubset(missing_columns):
            return {"property": property_name, "status": "skipped"}
    if not (oof_path.exists() and metadata_path.exists() and missing_path.exists()):
        return {"property": property_name, "status": "missing_input"}

    oof = pd.read_csv(oof_path)
    missing = pd.read_csv(missing_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    n_repeats = int(metadata["n_repeats"])
    observed_std = float(metadata["observed_std"])
    property_nrmse = float(metadata["oof_nrmse"])
    observed_ids = oof["polymer_id"].to_numpy(dtype=int)
    missing_ids = missing["polymer_id"].to_numpy(dtype=int)
    reference_fps = fp_matrix[observed_ids]
    reference_values = oof["y_true"].to_numpy(dtype=float)
    print(
        f"  {property_name}: observed={len(oof)}, missing={len(missing)}, repeats={n_repeats}",
        flush=True,
    )

    # Similarity ranking depends on structures only, so calculate it once and
    # apply each repeat's fold restriction afterward.
    oof_candidate_indices, oof_candidate_similarities = find_candidate_neighbors(
        query_fps=reference_fps,
        reference_fps=reference_fps,
        k=K_NEAREST_NEIGHBORS,
        candidate_multiplier=candidate_multiplier,
        query_chunk_size=query_chunk_size,
        label="OOF structures",
    )
    missing_candidate_indices, missing_candidate_similarities = find_candidate_neighbors(
        query_fps=fp_matrix[missing_ids],
        reference_fps=reference_fps,
        k=K_NEAREST_NEIGHBORS,
        candidate_multiplier=candidate_multiplier,
        query_chunk_size=query_chunk_size,
        label="missing structures",
    )

    oof_components: list[dict[str, np.ndarray]] = []
    missing_components: list[dict[str, np.ndarray]] = []
    for repeat_id in range(n_repeats):
        reference_fold_ids = oof[f"fold_id_repeat{repeat_id}"].to_numpy(dtype=int)
        oof_repeat = local_risk_from_candidates(
                query_fps=reference_fps,
                predicted_values=oof[f"y_pred_repeat{repeat_id}"].to_numpy(dtype=float),
                query_fold_ids=reference_fold_ids,
                reference_fps=reference_fps,
                reference_values=reference_values,
                candidate_indices=oof_candidate_indices,
                candidate_similarities=oof_candidate_similarities,
                reference_fold_ids=reference_fold_ids,
                k=K_NEAREST_NEIGHBORS,
        )
        missing_repeat = local_risk_from_candidates(
                query_fps=fp_matrix[missing_ids],
                predicted_values=missing[f"y_pred_repeat{repeat_id}"].to_numpy(dtype=float),
                query_fold_ids=missing[f"fold_id_repeat{repeat_id}"].to_numpy(dtype=int),
                reference_fps=reference_fps,
                reference_values=reference_values,
                candidate_indices=missing_candidate_indices,
                candidate_similarities=missing_candidate_similarities,
                reference_fold_ids=reference_fold_ids,
                k=K_NEAREST_NEIGHBORS,
        )
        oof_components.append(oof_repeat)
        missing_components.append(missing_repeat)
        oof_fallbacks = int(oof_repeat["n_candidate_fallbacks"][0]) if len(oof) else 0
        missing_fallbacks = (
            int(missing_repeat["n_candidate_fallbacks"][0]) if len(missing) else 0
        )
        print(
            f"    repeat {repeat_id}: OOF fallbacks={oof_fallbacks}, "
            f"missing fallbacks={missing_fallbacks}",
            flush=True,
        )

    oof_aggregate = aggregate_repeat_risk(oof_components)
    missing_aggregate = aggregate_repeat_risk(missing_components)

    oof_sigma_normalized = normalize_by_observed_std(
        oof["ensemble_std"].to_numpy(dtype=float), observed_std
    )
    missing_sigma_normalized = normalize_by_observed_std(
        missing["y_imputed_std"].to_numpy(dtype=float), observed_std
    )
    oof_distance = oof_aggregate["tanimoto_distance"]
    missing_distance = missing_aggregate["tanimoto_distance"]
    model_risk_fraction = property_nrmse / (1.0 + property_nrmse)
    oof_ensemble_risk_fraction = oof_sigma_normalized / (1.0 + oof_sigma_normalized)
    missing_ensemble_risk_fraction = missing_sigma_normalized / (
        1.0 + missing_sigma_normalized
    )
    oof_distance_sigma_risk = oof_distance * oof_sigma_normalized
    missing_distance_sigma_risk = missing_distance * missing_sigma_normalized
    oof_distance_sigma_risk_fraction = oof_distance_sigma_risk / (
        1.0 + oof_distance_sigma_risk
    )
    missing_distance_sigma_risk_fraction = missing_distance_sigma_risk / (
        1.0 + missing_distance_sigma_risk
    )
    oof_score = reliability_score(
        property_nrmse,
        oof_sigma_normalized,
        oof_distance,
    )
    missing_score = reliability_score(
        property_nrmse,
        missing_sigma_normalized,
        missing_distance,
    )

    oof_result = pd.DataFrame(
        {
            "polymer_id": oof["polymer_id"],
            "fold_id": oof["fold_id"],
            "y_true": oof["y_true"],
            "y_pred_oof": oof["y_pred_oof"],
            "ensemble_mean": oof["ensemble_mean"],
            "ensemble_std": oof["ensemble_std"],
            "oof_error": oof["oof_error"],
            "oof_error_normalized": oof["oof_error_normalized"],
            "property_nrmse": np.full(len(oof), property_nrmse),
            "nrmse_risk_fraction": np.full(len(oof), model_risk_fraction),
            "ensemble_std_normalized": oof_sigma_normalized,
            "ensemble_risk_fraction": oof_ensemble_risk_fraction,
            "mean_tanimoto_similarity": oof_aggregate[
                "mean_tanimoto_similarity"
            ],
            "tanimoto_distance": oof_aggregate["tanimoto_distance"],
            "distance_sigma_risk": oof_distance_sigma_risk,
            "distance_sigma_risk_fraction": oof_distance_sigma_risk_fraction,
            "local_mean": oof_aggregate["local_mean"],
            "local_std": oof_aggregate["local_std"],
            "local_z": oof_aggregate["local_z"],
            "local_consistency_risk": oof_aggregate[
                "local_consistency_risk"
            ],
            "structure_local_risk": oof_aggregate["structure_local_risk"],
            "reliability_formula_version": "fraction_equal_distance_sigma_v1",
            "reliability_score": oof_score,
        }
    )
    write_repeat_columns(oof_result, oof_components)

    missing_result = pd.DataFrame(
        {
            "polymer_id": missing["polymer_id"],
            "y_imputed_mean": missing["y_imputed_mean"],
            "y_imputed_std": missing["y_imputed_std"],
            "n_valid_predictions": missing["n_valid_predictions"],
            "property_nrmse": np.full(len(missing), property_nrmse),
            "nrmse_risk_fraction": np.full(len(missing), model_risk_fraction),
            "ensemble_std_normalized": missing_sigma_normalized,
            "ensemble_risk_fraction": missing_ensemble_risk_fraction,
            "mean_tanimoto_similarity": missing_aggregate[
                "mean_tanimoto_similarity"
            ],
            "tanimoto_distance": missing_aggregate["tanimoto_distance"],
            "distance_sigma_risk": missing_distance_sigma_risk,
            "distance_sigma_risk_fraction": missing_distance_sigma_risk_fraction,
            "local_mean": missing_aggregate["local_mean"],
            "local_std": missing_aggregate["local_std"],
            "local_z": missing_aggregate["local_z"],
            "local_consistency_risk": missing_aggregate[
                "local_consistency_risk"
            ],
            "structure_local_risk": missing_aggregate["structure_local_risk"],
            "reliability_formula_version": "fraction_equal_distance_sigma_v1",
            "reliability_score": missing_score,
        }
    )
    for repeat_id in range(n_repeats):
        missing_result[f"fold_id_repeat{repeat_id}"] = missing[
            f"fold_id_repeat{repeat_id}"
        ]
        missing_result[f"y_pred_repeat{repeat_id}"] = missing[
            f"y_pred_repeat{repeat_id}"
        ]
    write_repeat_columns(missing_result, missing_components)

    atomic_to_csv(oof_result, oof_output)
    atomic_to_csv(missing_result, missing_output)
    return {
        "property": property_name,
        "status": "success",
        "n_oof": len(oof_result),
        "n_missing": len(missing_result),
        "mean_oof_reliability": float(np.nanmean(oof_score)),
        "mean_missing_reliability": float(np.nanmean(missing_score)),
        "seconds": time.time() - started,
    }


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    data = pd.read_csv(args.data_path)
    fingerprints = load_fingerprints(data, args.data_path)
    fp_matrix = fingerprints[FP_COLS].to_numpy(dtype=np.uint8)
    oof_dir = output_dir / "01_oof_predictions"
    reliability_dir = ensure_dir(output_dir / "02_oof_reliability_scores")
    logs_dir = ensure_dir(output_dir / "logs")
    available = sorted(
        path.name[: -len("_all_folds.csv")]
        for path in oof_dir.glob("*_all_folds.csv")
    )
    properties = args.properties or available
    unknown = sorted(set(properties) - set(available))
    if unknown:
        raise ValueError(f"No OOF predictions found for: {unknown}")

    print("=" * 80)
    print(f"{VERSION}: fold-matched SI reliability scoring")
    print(f"Properties={len(properties)}, kNN={K_NEAREST_NEIGHBORS}")
    print(f"Outputs={output_dir}")
    print("=" * 80, flush=True)

    results: list[dict] = []
    failures = 0
    for index, prop in enumerate(properties, start=1):
        print(f"\n[{index}/{len(properties)}] {prop}", flush=True)
        try:
            result = compute_property(
                property_name=prop,
                data=data,
                fp_matrix=fp_matrix,
                output_dir=output_dir,
                skip_existing=args.skip_existing,
                candidate_multiplier=args.candidate_multiplier,
                query_chunk_size=args.query_chunk_size,
            )
        except Exception as exc:
            result = {"property": prop, "status": "error", "error": str(exc)}
            failures += 1
            print(f"  ERROR: {exc}", flush=True)
        results.append(result)
        print(f"  {result}", flush=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_json(
        {
            "version": VERSION,
            "formula": (
                "R_i=1-[NRMSE_j/(1+NRMSE_j)+sigma_i/(1+sigma_i)+"
                "(D_i*sigma_i)/(1+D_i*sigma_i)]/3; "
                "sigma_i=sample_std(repeated_predictions)/observed_std; "
                "D_i=sqrt(2*(1-mean_Tanimoto))"
            ),
            "formula_version": "fraction_equal_distance_sigma_v1",
            "diagnostic_only_columns": (
                "local_mean, local_std, local_z, local_consistency_risk, "
                "structure_local_risk, A_repeat*, G_repeat*"
            ),
            "neighbor_scope": "each query's repeat-specific training fold only",
            "ensemble_std_ddof": 1,
            "local_std_ddof": 1,
            "absolute_epsilon_used": False,
            "results": results,
        },
        logs_dir / f"02_reliability_summary_{timestamp}.json",
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
