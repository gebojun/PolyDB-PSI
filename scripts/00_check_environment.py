"""Preflight checks for the self-contained V11 AutoDL upload bundle."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_DIR = SCRIPT_DIR.parent / "config"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(CONFIG_DIR))

from workflow_common import infer_category, property_columns
from workflow_config import (
    FP_COLS,
    MIN_ELIGIBLE_OBSERVED,
    N_FOLDS,
    N_REPEATS,
    REPEAT_RANDOM_STATES,
    SOURCE_CHECKPOINT,
    SOURCE_MERGED_DATASET,
    VERSION,
    decide_scope,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check V11 AutoDL environment")
    parser.add_argument("--data-path", type=Path, default=SOURCE_MERGED_DATASET)
    parser.add_argument("--checkpoint-path", type=Path, default=SOURCE_CHECKPOINT)
    return parser.parse_args()


def version(module, fallback: str = "unknown") -> str:
    return str(getattr(module, "__version__", fallback))


def main() -> int:
    args = parse_args()
    failures: list[str] = []
    report: dict = {
        "workflow": VERSION,
        "python": sys.version,
        "platform": platform.platform(),
        "method": {
            "n_repeats": N_REPEATS,
            "n_folds": N_FOLDS,
            "repeat_seeds": REPEAT_RANDOM_STATES,
            "minimum_eligible_observed": MIN_ELIGIBLE_OBSERVED,
        },
    }

    try:
        import sklearn
        import torch
        import tabpfn
        import rdkit
        import matplotlib
        import seaborn

        report["packages"] = {
            "numpy": version(np),
            "pandas": version(pd),
            "scikit_learn": version(sklearn),
            "torch": version(torch),
            "tabpfn": version(tabpfn),
            "rdkit": version(rdkit),
            "matplotlib": version(matplotlib),
            "seaborn": version(seaborn),
        }
        report["cuda"] = {
            "available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
        }
        if torch.cuda.is_available():
            report["cuda"]["device_name"] = torch.cuda.get_device_name(0)
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            report["cuda"]["free_gb"] = free_bytes / 1024**3
            report["cuda"]["total_gb"] = total_bytes / 1024**3
    except Exception as exc:
        failures.append(f"required package import failed: {exc}")

    if not hasattr(np, "bitwise_count"):
        failures.append("NumPy >= 2.0 is required for exact packed-bit Tanimoto scoring")

    for label, path in (
        ("data", args.data_path),
        ("checkpoint", args.checkpoint_path),
    ):
        if not path.exists():
            failures.append(f"missing {label}: {path}")
        else:
            report[f"{label}_path"] = str(path.resolve())
            report[f"{label}_size_mb"] = path.stat().st_size / 1024**2

    cache_path = args.data_path.parent / "maccs_cache.parquet"
    if not cache_path.exists():
        failures.append(f"missing MACCS cache: {cache_path}")
    else:
        try:
            data = pd.read_csv(args.data_path)
            cache = pd.read_parquet(cache_path)
            if len(data) != len(cache):
                failures.append(
                    f"data/cache row mismatch: {len(data)} vs {len(cache)}"
                )
            missing_fp_cols = sorted(set(FP_COLS) - set(cache.columns))
            if missing_fp_cols:
                failures.append(f"MACCS cache lacks {len(missing_fp_cols)} bit columns")

            properties = property_columns(data)
            target_rows = []
            scope_candidate_rows = []
            valid_fp = cache["maccs_valid"].astype(bool)
            for prop in properties:
                n_observed = int(data[prop].notna().sum())
                n_missing = int(data[prop].isna().sum())
                decision = decide_scope(
                    infer_category(prop), prop, n_observed, n_missing
                )
                if decision.scope == "candidate_for_imputation":
                    scope_candidate_rows.append(prop)
                n_observed_eligible = int((data[prop].notna() & valid_fp).sum())
                if (
                    decision.scope == "candidate_for_imputation"
                    and n_observed_eligible >= MIN_ELIGIBLE_OBSERVED
                ):
                    target_rows.append(prop)
            report["data"] = {
                "rows": len(data),
                "columns": len(data.columns),
                "physical_properties": len(properties),
                "scope_candidates": len(scope_candidate_rows),
                "selected_targets_after_valid_fp_filter": len(target_rows),
                "selected_target_names": target_rows,
                "unique_smiles": int(data["smiles"].nunique(dropna=True)),
            }
            report["fingerprint_cache"] = {
                "rows": len(cache),
                "n_bits": len(FP_COLS),
                "valid": int(cache["maccs_valid"].astype(bool).sum()),
                "invalid": int((~cache["maccs_valid"].astype(bool)).sum()),
            }
        except Exception as exc:
            failures.append(f"data/cache validation failed: {exc}")

    report["status"] = "ok" if not failures else "failed"
    report["failures"] = failures
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
