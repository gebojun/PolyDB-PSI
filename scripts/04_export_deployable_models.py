"""Export fitted V11 fold models for offline online-prediction deployment.

This script deliberately does not regenerate V11 predictions or reliability
scores. It refits the already completed strict repeated 5-fold design and
stores each fitted TabPFN estimator with TabPFN's portable fitted-estimator
format. The large foundation checkpoint is not duplicated inside each model
archive; it remains a separate deployment file.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from json import JSONDecodeError

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from tabpfn import TabPFNRegressor
from tabpfn.model_loading import save_fitted_tabpfn_model

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
CONFIG_DIR = ROOT / "config"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(CONFIG_DIR))

from workflow_common import (  # noqa: E402
    ensure_dir,
    filter_feature_columns,
    format_time,
    get_same_category_properties,
    infer_category,
    make_feature_matrix,
    property_columns,
)
from workflow_config import (  # noqa: E402
    FP_COLS,
    MIN_ELIGIBLE_OBSERVED,
    MODEL_RANDOM_STATE,
    N_FOLDS,
    N_REPEATS,
    REPEAT_RANDOM_STATES,
    SMILES_COL,
    SOURCE_CHECKPOINT,
    SOURCE_MERGED_DATASET,
    VERSION,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export deployable fitted models from the completed V11 design"
    )
    parser.add_argument("--data-path", type=Path, default=SOURCE_MERGED_DATASET)
    parser.add_argument("--checkpoint-path", type=Path, default=SOURCE_CHECKPOINT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "deployable_models_v11",
    )
    parser.add_argument(
        "--properties",
        nargs="+",
        default=None,
        help="Export only these properties; default is every successful V11 property",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--min-free-gpu-gb", type=float, default=2.0)
    parser.add_argument("--n-repeats", type=int, default=N_REPEATS)
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    parser.add_argument(
        "--repeat-seeds",
        nargs="+",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing export directory after an explicit request",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted export and preserve completed properties",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    return requested


def check_gpu_capacity(device: str, minimum_free_gpu_gb: float) -> None:
    if device != "cuda" or minimum_free_gpu_gb <= 0:
        return
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gb = free_bytes / 1024**3
    total_gb = total_bytes / 1024**3
    if free_gb < minimum_free_gpu_gb:
        raise RuntimeError(
            f"CUDA free memory is {free_gb:.2f}/{total_gb:.2f} GB; "
            f"at least {minimum_free_gpu_gb:.2f} GB is required"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_fingerprints(data: pd.DataFrame, data_path: Path) -> pd.DataFrame:
    parquet_path = data_path.parent / "maccs_cache.parquet"
    pickle_path = data_path.parent / "maccs_cache.pkl"
    for cache_path, reader in (
        (parquet_path, pd.read_parquet),
        (pickle_path, pd.read_pickle),
    ):
        if not cache_path.exists():
            continue
        cached = reader(cache_path)
        if len(cached) < len(data) or not set(FP_COLS).issubset(cached.columns):
            continue
        cached = cached.iloc[: len(data)].copy()
        cached.index = data.index
        if "maccs_valid" not in cached:
            cached["maccs_valid"] = True
        print(f"Loaded MACCS cache: {cache_path}", flush=True)
        return cached
    raise FileNotFoundError(
        "V11 MACCS cache was not found beside the merged dataset; "
        "refusing to silently regenerate a deployment feature matrix"
    )


def successful_v11_properties() -> list[str]:
    metadata_dir = ROOT / "outputs" / "01_oof_predictions"
    properties: list[str] = []
    for path in sorted(metadata_dir.glob("*_metadata.json")):
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if metadata.get("status") == "success":
            name = str(metadata.get("property_name", ""))
            if name:
                properties.append(name)
    return properties


def portable_save_fitted_model(
    model: TabPFNRegressor,
    destination: Path,
) -> None:
    """Save a fitted model and remove the machine-specific checkpoint path."""
    # TabPFN requires the temporary path itself to end in .tabpfn_fit.
    temporary = destination.parent / f".{destination.stem}.tmp.tabpfn_fit"
    temporary.unlink(missing_ok=True)
    save_fitted_tabpfn_model(model, temporary)

    # TabPFN's official fitted-estimator archive stores the initialization
    # parameters as JSON. Replace the AutoDL absolute path with a stable token;
    # the local service will resolve this token to its downloaded checkpoint.
    patched = destination.with_suffix(destination.suffix + ".patched")
    patched.unlink(missing_ok=True)
    with zipfile.ZipFile(temporary, "r") as source, zipfile.ZipFile(
        patched, "w", compression=zipfile.ZIP_DEFLATED
    ) as target:
        for item in source.infolist():
            payload = source.read(item.filename)
            if item.filename == "init_params.json":
                params = json.loads(payload.decode("utf-8"))
                params["model_path"] = "__V11_CHECKPOINT__"
                payload = (json.dumps(params, ensure_ascii=False, indent=2) + "\n").encode(
                    "utf-8"
                )
            target.writestr(item, payload)
    temporary.unlink(missing_ok=True)
    patched.replace(destination)


def prepare_output(path: Path, overwrite: bool, resume: bool) -> None:
    if overwrite and resume:
        raise ValueError("--overwrite and --resume cannot be used together")
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            if not resume:
                raise RuntimeError(
                    f"Output directory is not empty: {path}; "
                    "use --resume to continue it or --overwrite to replace it"
                )
        else:
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    if args.n_repeats < 2 or args.n_folds < 2:
        raise ValueError("at least two repeats and two folds are required")
    repeat_seeds = args.repeat_seeds or REPEAT_RANDOM_STATES[: args.n_repeats]
    if len(repeat_seeds) != args.n_repeats:
        raise ValueError("the number of repeat seeds must equal --n-repeats")
    if len(set(repeat_seeds)) != len(repeat_seeds):
        raise ValueError("repeat seeds must be unique")

    data_path = args.data_path.resolve()
    checkpoint_path = args.checkpoint_path.resolve()
    output_dir = args.output_dir.resolve()
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    device = resolve_device(args.device)
    data = pd.read_csv(data_path)
    if SMILES_COL not in data.columns:
        raise ValueError(f"input data must contain {SMILES_COL!r}")
    fingerprints = load_fingerprints(data, data_path)
    model_data = pd.concat([data, fingerprints], axis=1)
    valid_fp = fingerprints["maccs_valid"].astype(bool).to_numpy()

    available = set(property_columns(data))
    targets = args.properties or successful_v11_properties()
    unknown = [name for name in targets if name not in available]
    if unknown:
        raise ValueError(f"unknown properties: {unknown}")
    if not targets:
        raise RuntimeError(
            "no successful V11 properties were found; complete V11 first or pass --properties"
        )

    prepare_output(output_dir, args.overwrite, args.resume)
    properties_dir = ensure_dir(output_dir / "properties")
    manifest: dict[str, object] = {
        "deployment_format": "tabpfn_fitted_v11_portable_v1",
        "source_version": VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_filename": data_path.name,
        "data_sha256": sha256_file(data_path),
        "checkpoint_filename": checkpoint_path.name,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_token": "__V11_CHECKPOINT__",
        "device_used_for_export": device,
        "model_random_state": MODEL_RANDOM_STATE,
        "n_repeats": args.n_repeats,
        "n_folds": args.n_folds,
        "repeat_random_states": repeat_seeds,
        "fold_assignment_missing_query": "sha256(smiles|property|repeat_seed) mod n_folds",
        "feature_protocol": "MACCS 167 + same-category V11 properties, fold-wise usable-column filtering",
        "properties": {},
    }

    print(
        f"V11 deployment export: properties={len(targets)}, "
        f"models={len(targets) * args.n_repeats * args.n_folds}, device={device}",
        flush=True,
    )
    started = time.time()
    manifest_path = output_dir / "manifest.json"
    for property_position, target_property in enumerate(targets, start=1):
        category = infer_category(target_property)
        observed_idx = data.index[data[target_property].notna() & valid_fp]
        if len(observed_idx) < MIN_ELIGIBLE_OBSERVED:
            raise RuntimeError(
                f"{target_property}: eligible observed count {len(observed_idx)} "
                f"is below V11 minimum {MIN_ELIGIBLE_OBSERVED}"
            )
        same_category = get_same_category_properties(
            target_property, property_columns(data)
        )
        requested_features = FP_COLS + same_category
        property_dir = ensure_dir(properties_dir / target_property)
        property_meta: dict[str, object] = {
            "property_name": target_property,
            "category": category,
            "n_observed_eligible": int(len(observed_idx)),
            "requested_feature_count": len(requested_features),
            "same_category_property_features": same_category,
            "models": [],
        }
        existing_meta_path = property_dir / "metadata.json"
        if args.resume and existing_meta_path.is_file():
            try:
                existing_meta = json.loads(existing_meta_path.read_text(encoding="utf-8"))
                existing_models = existing_meta.get("models", [])
                complete_existing = (
                    existing_meta.get("property_name") == target_property
                    and len(existing_models) == args.n_repeats * args.n_folds
                    and all(
                        (output_dir / str(item["path"])).is_file()
                        for item in existing_models
                        if isinstance(item, dict) and "path" in item
                    )
                )
            except (OSError, JSONDecodeError, TypeError, KeyError):
                complete_existing = False
                existing_meta = None
            if complete_existing:
                manifest["properties"][target_property] = existing_meta
                print(
                    f"[{property_position}/{len(targets)}] {target_property}: "
                    "complete export found, skipped",
                    flush=True,
                )
                continue
        print(
            f"[{property_position}/{len(targets)}] {target_property}: "
            f"observed={len(observed_idx)}, category={category}",
            flush=True,
        )

        # Match V11's training implementation: load the foundation checkpoint
        # once per property and reuse the estimator for its 25 fit states.
        model = TabPFNRegressor(
            model_path=str(checkpoint_path),
            device=device,
            ignore_pretraining_limits=True,
            random_state=MODEL_RANDOM_STATE,
        )
        try:
            for repeat_id, repeat_seed in enumerate(repeat_seeds):
                splitter = KFold(
                    n_splits=args.n_folds,
                    shuffle=True,
                    random_state=repeat_seed,
                )
                split_pairs = list(splitter.split(np.arange(len(observed_idx))))
                for fold_id, (train_positions, _test_positions) in enumerate(split_pairs):
                    check_gpu_capacity(device, args.min_free_gpu_gb)
                    train_idx = observed_idx[train_positions]
                    train_frame = model_data.loc[train_idx]
                    usable_features = filter_feature_columns(
                        train_frame, requested_features
                    )
                    if not usable_features:
                        raise RuntimeError(
                            f"{target_property} repeat {repeat_id} fold {fold_id}: "
                            "no usable features"
                        )
                    x_train = make_feature_matrix(train_frame, usable_features)
                    y_train = data.loc[train_idx, target_property].to_numpy(dtype=np.float32)
                    model.fit(x_train, y_train)
                    fitted_model_path = property_dir / (
                        f"repeat_{repeat_id:02d}_fold_{fold_id:02d}.tabpfn_fit"
                    )
                    portable_save_fitted_model(model, fitted_model_path)
                    property_meta["models"].append(
                        {
                            "repeat_id": repeat_id,
                            "repeat_seed": repeat_seed,
                            "fold_id": fold_id,
                            "path": str(fitted_model_path.relative_to(output_dir)),
                            "feature_columns": usable_features,
                            "n_train": int(len(train_idx)),
                            "n_fingerprint_features": int(
                                sum(name in FP_COLS for name in usable_features)
                            ),
                            "n_property_features": int(
                                sum(name in same_category for name in usable_features)
                            ),
                        }
                    )
                    del x_train, y_train
                    gc.collect()
                    if device == "cuda":
                        torch.cuda.empty_cache()
                    print(
                        f"  repeat {repeat_id + 1}/{args.n_repeats}, "
                        f"fold {fold_id + 1}/{args.n_folds} exported",
                        flush=True,
                    )
        finally:
            del model
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

        property_meta_path = property_dir / "metadata.json"
        property_meta_path.write_text(
            json.dumps(property_meta, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest["properties"][target_property] = property_meta
        manifest["elapsed_seconds"] = time.time() - started
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    manifest["elapsed_seconds"] = time.time() - started
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"DONE: exported {len(targets) * args.n_repeats * args.n_folds} models "
        f"in {format_time(time.time() - started)}",
        flush=True,
    )
    print(f"Output: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
