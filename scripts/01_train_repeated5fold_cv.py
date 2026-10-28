"""V11: five-repeat, five-fold strict cross-fitted TabPFN predictions."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from tabpfn import TabPFNRegressor

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_DIR = SCRIPT_DIR.parent / "config"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(CONFIG_DIR))

from reliability_math import mae, nrmse, r2, rmse, sample_std
from workflow_common import (
    compute_maccs_fingerprints,
    ensure_dir,
    filter_feature_columns,
    format_time,
    get_same_category_properties,
    infer_category,
    log_message,
    make_feature_matrix,
    property_columns,
    save_json,
)
from workflow_config import (
    DEFAULT_OUTPUT_DIR,
    FP_COLS,
    MIN_ELIGIBLE_OBSERVED,
    MODEL_RANDOM_STATE,
    N_FOLDS,
    N_REPEATS,
    REPEAT_RANDOM_STATES,
    SMILES_COL,
    SOURCE_CHECKPOINT,
    SOURCE_MERGED_DATASET,
    TEST_PROPERTIES,
    VERSION,
    decide_scope,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V11 strict repeated 5-fold TabPFN cross-fitting"
    )
    parser.add_argument("--data-path", type=Path, default=SOURCE_MERGED_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint-path", type=Path, default=SOURCE_CHECKPOINT)
    parser.add_argument("--properties", nargs="+", default=None)
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--min-free-gpu-gb", type=float, default=2.0)
    parser.add_argument("--n-repeats", type=int, default=N_REPEATS)
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    parser.add_argument("--repeat-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--n-rows", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    return requested


def check_gpu_capacity(device: str, minimum_free_gb: float) -> None:
    if device != "cuda" or minimum_free_gb <= 0:
        return
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gb = free_bytes / 1024**3
    total_gb = total_bytes / 1024**3
    if free_gb < minimum_free_gb:
        raise RuntimeError(
            f"CUDA free memory is {free_gb:.2f}/{total_gb:.2f} GB; "
            f"at least {minimum_free_gb:.2f} GB is required"
        )


def build_model(checkpoint_path: Path, device: str) -> TabPFNRegressor:
    return TabPFNRegressor(
        model_path=str(checkpoint_path),
        device=device,
        ignore_pretraining_limits=True,
        random_state=MODEL_RANDOM_STATE,
    )


def load_fingerprints(data: pd.DataFrame, data_path: Path) -> pd.DataFrame:
    parquet_path = data_path.parent / "maccs_cache.parquet"
    pickle_path = data_path.parent / "maccs_cache.pkl"
    for cache_path, reader in (
        (parquet_path, pd.read_parquet),
        (pickle_path, pd.read_pickle),
    ):
        if not cache_path.exists():
            continue
        try:
            cached = reader(cache_path)
            if len(cached) >= len(data) and set(FP_COLS).issubset(cached.columns):
                cached = cached.iloc[: len(data)].copy()
                cached.index = data.index
                if "maccs_valid" not in cached:
                    cached["maccs_valid"] = True
                print(f"Loaded MACCS cache: {cache_path.name}", flush=True)
                return cached
        except Exception as exc:
            print(f"Could not read {cache_path.name}: {exc}; recomputing", flush=True)

    fingerprints = compute_maccs_fingerprints(data, SMILES_COL)
    try:
        fingerprints.to_parquet(parquet_path)
    except Exception:
        try:
            fingerprints.to_pickle(pickle_path)
        except Exception:
            pass
    return fingerprints


def stable_missing_fold_ids(
    smiles: pd.Series,
    property_name: str,
    repeat_seed: int,
    n_folds: int,
) -> np.ndarray:
    """Assign missing cells before prediction using a stable content hash."""
    fold_ids = np.empty(len(smiles), dtype=np.int16)
    for position, value in enumerate(smiles.astype(str)):
        payload = f"{value}|{property_name}|{repeat_seed}".encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        fold_ids[position] = int.from_bytes(digest[:8], "big") % n_folds
    return fold_ids


def atomic_to_csv(frame: pd.DataFrame, destination: Path) -> None:
    ensure_dir(destination.parent)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(destination)


def build_scope_manifest(data: pd.DataFrame, fingerprints: pd.DataFrame) -> pd.DataFrame:
    valid_fp = fingerprints["maccs_valid"].astype(bool)
    rows: list[dict] = []
    for prop in property_columns(data):
        category = infer_category(prop)
        n_observed = int(data[prop].notna().sum())
        n_missing = int(data[prop].isna().sum())
        n_observed_eligible = int((data[prop].notna() & valid_fp).sum())
        n_missing_eligible = int((data[prop].isna() & valid_fp).sum())
        decision = decide_scope(category, prop, n_observed, n_missing)
        rows.append(
            {
                "property_name": prop,
                "category": category,
                "n_observed_total": n_observed,
                "n_missing_total": n_missing,
                "n_observed_eligible": n_observed_eligible,
                "n_missing_eligible": n_missing_eligible,
                "min_observed_required": decision.min_observed_required,
                "scope": decision.scope,
                "scope_reason": decision.reason,
                "v11_will_run": (
                    decision.scope == "candidate_for_imputation"
                    and n_observed_eligible >= MIN_ELIGIBLE_OBSERVED
                ),
            }
        )
    return pd.DataFrame(rows)


def choose_targets(
    args: argparse.Namespace,
    data: pd.DataFrame,
    manifest: pd.DataFrame,
) -> list[str]:
    available = set(property_columns(data))
    if args.properties:
        unknown = [name for name in args.properties if name not in available]
        if unknown:
            raise ValueError(f"Unknown properties: {unknown}")
        return list(dict.fromkeys(args.properties))
    if args.test_mode:
        return [name for name in TEST_PROPERTIES if name in available]
    return manifest.loc[manifest["v11_will_run"], "property_name"].tolist()


def run_property(
    data: pd.DataFrame,
    model_data: pd.DataFrame,
    fingerprints: pd.DataFrame,
    target_property: str,
    checkpoint_path: Path,
    output_dir: Path,
    repeat_seeds: list[int],
    n_folds: int,
    device: str,
    minimum_free_gpu_gb: float,
    skip_existing: bool,
    log_file: Path,
) -> dict:
    started = time.time()
    oof_dir = output_dir / "01_oof_predictions"
    missing_dir = output_dir / "03_missing_predictions"
    oof_path = oof_dir / f"{target_property}_all_folds.csv"
    metadata_path = oof_dir / f"{target_property}_metadata.json"
    missing_path = missing_dir / f"{target_property}_missing_predictions.csv"

    if skip_existing and oof_path.exists() and metadata_path.exists() and missing_path.exists():
        try:
            existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            existing_metadata = {}
        if (
            existing_metadata.get("version") == VERSION
            and existing_metadata.get("method") == "strict_repeated_cross_fitting"
            and existing_metadata.get("n_repeats") == len(repeat_seeds)
            and existing_metadata.get("n_folds") == n_folds
            and existing_metadata.get("repeat_random_states") == repeat_seeds
        ):
            log_message(f"Skip {target_property}: complete V11 outputs already exist", log_file)
            return {"property_name": target_property, "status": "skipped"}
        log_message(
            f"Recompute {target_property}: existing files do not match this V11 method",
            log_file,
        )

    category = infer_category(target_property)
    valid_fp = fingerprints["maccs_valid"].astype(bool)
    observed_idx = data.index[data[target_property].notna() & valid_fp]
    missing_idx = data.index[data[target_property].isna() & valid_fp]
    n_observed = len(observed_idx)
    n_missing = len(missing_idx)

    log_message("\n" + "=" * 80, log_file)
    log_message(f"Property: {target_property} ({category})", log_file)
    log_message(f"Eligible observed={n_observed}, missing={n_missing}", log_file)

    if n_observed < MIN_ELIGIBLE_OBSERVED:
        log_message(
            f"Skipped: eligible observed count is below {MIN_ELIGIBLE_OBSERVED}",
            log_file,
        )
        return {
            "property_name": target_property,
            "status": "skipped",
            "reason": "insufficient_eligible_observed",
            "n_observed": n_observed,
        }
    if n_observed < n_folds:
        return {
            "property_name": target_property,
            "status": "skipped",
            "reason": "fewer_observations_than_folds",
            "n_observed": n_observed,
        }

    all_properties = property_columns(data)
    same_category = get_same_category_properties(target_property, all_properties)
    requested_features = FP_COLS + same_category
    y_true = data.loc[observed_idx, target_property].to_numpy(dtype=float)
    observed_mean = float(np.mean(y_true))
    observed_std = float(np.std(y_true, ddof=1))
    if not np.isfinite(observed_std) or observed_std <= 0:
        return {
            "property_name": target_property,
            "status": "skipped",
            "reason": "zero_or_nonfinite_observed_sample_std",
            "n_observed": n_observed,
        }

    n_repeats = len(repeat_seeds)
    observed_predictions = np.full((n_observed, n_repeats), np.nan, dtype=float)
    observed_folds = np.full((n_observed, n_repeats), -1, dtype=np.int16)
    missing_predictions = np.full((n_missing, n_repeats), np.nan, dtype=float)
    missing_folds = np.full((n_missing, n_repeats), -1, dtype=np.int16)
    repeat_metadata: list[dict] = []

    check_gpu_capacity(device, minimum_free_gpu_gb)
    model = build_model(checkpoint_path, device)
    log_message(
        f"Model loaded once; repeats={n_repeats}, folds={n_folds}, "
        f"requested features=MACCS 167 + {len(same_category)} same-category properties",
        log_file,
    )

    try:
        for repeat_id, seed in enumerate(repeat_seeds):
            repeat_started = time.time()
            splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            fold_ids = np.full(n_observed, -1, dtype=np.int16)
            split_pairs = list(splitter.split(np.arange(n_observed)))
            for fold_id, (_, test_positions) in enumerate(split_pairs):
                fold_ids[test_positions] = fold_id
            observed_folds[:, repeat_id] = fold_ids

            if n_missing:
                assigned_missing_folds = stable_missing_fold_ids(
                    data.loc[missing_idx, SMILES_COL], target_property, seed, n_folds
                )
                missing_folds[:, repeat_id] = assigned_missing_folds
            else:
                assigned_missing_folds = np.empty(0, dtype=np.int16)

            fold_rows: list[dict] = []
            for fold_id, (train_positions, test_positions) in enumerate(split_pairs):
                fold_started = time.time()
                check_gpu_capacity(device, minimum_free_gpu_gb)
                train_idx = observed_idx[train_positions]
                test_idx = observed_idx[test_positions]
                train_frame = model_data.loc[train_idx]
                usable_features = filter_feature_columns(train_frame, requested_features)
                if not usable_features:
                    raise RuntimeError(
                        f"{target_property} repeat {repeat_id} fold {fold_id}: no usable features"
                    )

                x_train = make_feature_matrix(train_frame, usable_features)
                y_train = data.loc[train_idx, target_property].to_numpy(dtype=np.float32)
                x_test = make_feature_matrix(model_data.loc[test_idx], usable_features)

                model.fit(x_train, y_train)
                test_prediction = np.asarray(model.predict(x_test), dtype=float).reshape(-1)
                if len(test_prediction) != len(test_positions):
                    raise RuntimeError("TabPFN returned an unexpected OOF prediction length")
                observed_predictions[test_positions, repeat_id] = test_prediction

                missing_positions = np.flatnonzero(assigned_missing_folds == fold_id)
                if len(missing_positions):
                    fold_missing_idx = missing_idx[missing_positions]
                    x_missing = make_feature_matrix(
                        model_data.loc[fold_missing_idx], usable_features
                    )
                    fold_missing_prediction = np.asarray(
                        model.predict(x_missing), dtype=float
                    ).reshape(-1)
                    missing_predictions[missing_positions, repeat_id] = (
                        fold_missing_prediction
                    )

                fold_true = y_true[test_positions]
                fold_rows.append(
                    {
                        "repeat_id": repeat_id,
                        "repeat_seed": seed,
                        "fold_id": fold_id,
                        "n_train": int(len(train_positions)),
                        "n_test": int(len(test_positions)),
                        "n_missing_predicted": int(len(missing_positions)),
                        "n_features": int(len(usable_features)),
                        "n_fingerprint_features": int(
                            sum(name in FP_COLS for name in usable_features)
                        ),
                        "n_property_features": int(
                            sum(name in same_category for name in usable_features)
                        ),
                        "rmse": rmse(fold_true, test_prediction),
                        "mae": mae(fold_true, test_prediction),
                        "r2": r2(fold_true, test_prediction),
                        "seconds": time.time() - fold_started,
                    }
                )
                log_message(
                    f"Repeat {repeat_id + 1}/{n_repeats}, fold {fold_id + 1}/{n_folds}: "
                    f"train={len(train_positions)}, test={len(test_positions)}, "
                    f"missing={len(missing_positions)}, features={len(usable_features)}, "
                    f"time={format_time(time.time() - fold_started)}",
                    log_file,
                )
                del x_train, y_train, x_test
                if device == "cuda":
                    torch.cuda.empty_cache()

            repeat_prediction = observed_predictions[:, repeat_id]
            if not np.all(np.isfinite(repeat_prediction)):
                raise RuntimeError(
                    f"{target_property} repeat {repeat_id} has incomplete OOF predictions"
                )
            if n_missing and not np.all(np.isfinite(missing_predictions[:, repeat_id])):
                raise RuntimeError(
                    f"{target_property} repeat {repeat_id} has incomplete missing predictions"
                )

            repeat_metrics = {
                "repeat_id": repeat_id,
                "repeat_seed": seed,
                "rmse": rmse(y_true, repeat_prediction),
                "mae": mae(y_true, repeat_prediction),
                "r2": r2(y_true, repeat_prediction),
                "nrmse": nrmse(y_true, repeat_prediction, observed_std),
                "seconds": time.time() - repeat_started,
                "folds": fold_rows,
            }
            repeat_metadata.append(repeat_metrics)
            log_message(
                f"Repeat {repeat_id + 1} complete: R2={repeat_metrics['r2']:.4f}, "
                f"NRMSE={repeat_metrics['nrmse']:.4f}",
                log_file,
            )
    finally:
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    ensemble_mean = np.mean(observed_predictions, axis=1)
    ensemble_std = sample_std(observed_predictions, axis=1)
    oof_error = np.abs(y_true - ensemble_mean)
    oof_error_normalized = oof_error / observed_std

    oof_result = pd.DataFrame(
        {
            "polymer_id": observed_idx.to_numpy(dtype=int),
            "fold_id": observed_folds[:, 0],
            "y_true": y_true,
        }
    )
    for repeat_id in range(n_repeats):
        oof_result[f"fold_id_repeat{repeat_id}"] = observed_folds[:, repeat_id]
        oof_result[f"y_pred_repeat{repeat_id}"] = observed_predictions[:, repeat_id]
    oof_result["ensemble_mean"] = ensemble_mean
    oof_result["ensemble_std"] = ensemble_std
    oof_result["y_pred_oof"] = ensemble_mean
    oof_result["oof_error"] = oof_error
    oof_result["oof_error_normalized"] = oof_error_normalized

    missing_result = pd.DataFrame(
        {"polymer_id": missing_idx.to_numpy(dtype=int)}
    )
    for repeat_id in range(n_repeats):
        missing_result[f"fold_id_repeat{repeat_id}"] = missing_folds[:, repeat_id]
        missing_result[f"y_pred_repeat{repeat_id}"] = missing_predictions[:, repeat_id]
        # Compatibility alias for V10 consumers; each value is the prediction
        # from that repeat's preassigned fold, not all five fold models.
        missing_result[f"y_pred_fold{repeat_id}"] = missing_predictions[:, repeat_id]
    if n_missing:
        missing_result["y_imputed_mean"] = np.mean(missing_predictions, axis=1)
        missing_result["y_imputed_std"] = sample_std(missing_predictions, axis=1)
        missing_result["n_valid_predictions"] = np.sum(
            np.isfinite(missing_predictions), axis=1
        )
    else:
        missing_result["y_imputed_mean"] = pd.Series(dtype=float)
        missing_result["y_imputed_std"] = pd.Series(dtype=float)
        missing_result["n_valid_predictions"] = pd.Series(dtype=int)

    atomic_to_csv(oof_result, oof_path)
    atomic_to_csv(missing_result, missing_path)

    main_rmse = rmse(y_true, ensemble_mean)
    main_nrmse = nrmse(y_true, ensemble_mean, observed_std)
    metadata = {
        "version": VERSION,
        "property_name": target_property,
        "category": category,
        "status": "success",
        "method": "strict_repeated_cross_fitting",
        "n_observed": n_observed,
        "n_missing": n_missing,
        "n_repeats": n_repeats,
        "n_folds": n_folds,
        "repeat_random_states": repeat_seeds,
        "model_random_state": MODEL_RANDOM_STATE,
        "model_reused_across_fits": True,
        "observed_mean": observed_mean,
        "observed_std": observed_std,
        "observed_std_ddof": 1,
        "ensemble_std_ddof": 1,
        "oof_rmse": main_rmse,
        "oof_mae": mae(y_true, ensemble_mean),
        "oof_r2": r2(y_true, ensemble_mean),
        "oof_nrmse": main_nrmse,
        "property_nrmse_definition": (
            "RMSE of the mean of the strict OOF repeat predictions divided "
            "by the observed sample standard deviation (ddof=1)"
        ),
        "repeat_nrmse_values": [row["nrmse"] for row in repeat_metadata],
        "repeat_nrmse_mean": float(
            np.mean([row["nrmse"] for row in repeat_metadata])
        ),
        "repeat_nrmse_std_ddof1": float(
            np.std([row["nrmse"] for row in repeat_metadata], ddof=1)
        ),
        "repeat_metrics": repeat_metadata,
        "requested_feature_count": len(requested_features),
        "same_category_property_features": same_category,
        "missing_fold_assignment": "sha256(smiles|property|repeat_seed) mod n_folds",
        "total_time_seconds": time.time() - started,
    }
    save_json(metadata, metadata_path)
    log_message(
        f"Completed {target_property}: pooled-mean R2={metadata['oof_r2']:.4f}, "
        f"NRMSE={main_nrmse:.4f}, time={format_time(metadata['total_time_seconds'])}",
        log_file,
    )
    return metadata


def main() -> int:
    args = parse_args()
    if args.n_repeats < 2:
        raise ValueError("At least two repeats are required to estimate sample dispersion")
    if args.n_folds < 2:
        raise ValueError("At least two folds are required")
    repeat_seeds = args.repeat_seeds or REPEAT_RANDOM_STATES[: args.n_repeats]
    if len(repeat_seeds) != args.n_repeats:
        raise ValueError("The number of repeat seeds must equal --n-repeats")
    if len(set(repeat_seeds)) != len(repeat_seeds):
        raise ValueError("Repeat seeds must be unique")

    device = resolve_device(args.device)
    if not args.data_path.exists():
        raise FileNotFoundError(args.data_path)
    if not args.checkpoint_path.exists():
        raise FileNotFoundError(args.checkpoint_path)

    output_dir = args.output_dir.resolve()
    logs_dir = ensure_dir(output_dir / "logs")
    ensure_dir(output_dir / "01_oof_predictions")
    ensure_dir(output_dir / "03_missing_predictions")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"01_train_repeated5fold_cv_{timestamp}.log"

    log_message("=" * 80, log_file)
    log_message(f"{VERSION}", log_file)
    log_message(
        f"Device={device}; repeats={args.n_repeats}; folds={args.n_folds}; "
        f"repeat seeds={repeat_seeds}",
        log_file,
    )
    data = pd.read_csv(args.data_path, nrows=args.n_rows)
    fingerprints = load_fingerprints(data, args.data_path)
    model_data = pd.concat([data, fingerprints], axis=1)
    manifest = build_scope_manifest(data, fingerprints)
    manifest_path = logs_dir / "target_scope_decisions.csv"
    manifest.to_csv(manifest_path, index=False)
    targets = choose_targets(args, data, manifest)
    log_message(
        f"Data shape={data.shape}; valid MACCS={int(fingerprints['maccs_valid'].sum())}; "
        f"selected targets={len(targets)}",
        log_file,
    )

    run_manifest = {
        "version": VERSION,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "data_path": str(args.data_path.resolve()),
        "checkpoint_path": str(args.checkpoint_path.resolve()),
        "device": device,
        "n_repeats": args.n_repeats,
        "n_folds": args.n_folds,
        "repeat_seeds": repeat_seeds,
        "target_properties": targets,
        "minimum_eligible_observed": MIN_ELIGIBLE_OBSERVED,
    }
    save_json(run_manifest, logs_dir / "run_method_manifest.json")

    results: list[dict] = []
    failed = 0
    for position, target in enumerate(targets, start=1):
        log_message(f"\nProgress {position}/{len(targets)}", log_file)
        try:
            result = run_property(
                data=data,
                model_data=model_data,
                fingerprints=fingerprints,
                target_property=target,
                checkpoint_path=args.checkpoint_path,
                output_dir=output_dir,
                repeat_seeds=repeat_seeds,
                n_folds=args.n_folds,
                device=device,
                minimum_free_gpu_gb=args.min_free_gpu_gb,
                skip_existing=args.skip_existing,
                log_file=log_file,
            )
        except Exception as exc:
            result = {
                "property_name": target,
                "status": "error",
                "error": str(exc),
            }
            failed += 1
            log_message(f"ERROR {target}: {exc}", log_file)
        results.append(result)

    summary = {
        "version": VERSION,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "n_targets": len(targets),
        "n_success": sum(row.get("status") == "success" for row in results),
        "n_skipped": sum(row.get("status") == "skipped" for row in results),
        "n_failed": failed,
        "results": results,
    }
    save_json(summary, logs_dir / f"01_training_summary_{timestamp}.json")
    log_message(json.dumps({k: v for k, v in summary.items() if k != "results"}), log_file)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
