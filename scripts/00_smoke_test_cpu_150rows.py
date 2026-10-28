"""CPU smoke test on the first 150 rows without touching production outputs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
CONFIG_DIR = ROOT / "config"
sys.path.insert(0, str(CONFIG_DIR))

from workflow_config import SOURCE_CHECKPOINT, SOURCE_MERGED_DATASET


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V11 150-row CPU smoke test")
    parser.add_argument("--data-path", type=Path, default=SOURCE_MERGED_DATASET)
    parser.add_argument("--checkpoint-path", type=Path, default=SOURCE_CHECKPOINT)
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("RUN:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def verify_outputs(output_dir: Path, prop: str) -> dict:
    oof = pd.read_csv(output_dir / "01_oof_predictions" / f"{prop}_all_folds.csv")
    missing = pd.read_csv(
        output_dir / "03_missing_predictions" / f"{prop}_missing_predictions.csv"
    )
    oof_rel = pd.read_csv(
        output_dir / "02_oof_reliability_scores" / f"{prop}_oof_reliability.csv"
    )
    missing_rel = pd.read_csv(
        output_dir / "03_missing_predictions" / f"{prop}_missing_reliability.csv"
    )
    metadata = json.loads(
        (output_dir / "01_oof_predictions" / f"{prop}_metadata.json").read_text(
            encoding="utf-8"
        )
    )

    prediction_cols = ["y_pred_repeat0", "y_pred_repeat1"]
    fold_cols = ["fold_id_repeat0", "fold_id_repeat1"]
    assert len(oof) >= 50
    assert len(missing) > 0
    assert np.isfinite(oof[prediction_cols].to_numpy(dtype=float)).all()
    assert np.isfinite(missing[prediction_cols].to_numpy(dtype=float)).all()
    assert set(np.unique(oof[fold_cols].to_numpy())) <= {0, 1}
    assert set(np.unique(missing[fold_cols].to_numpy())) <= {0, 1}

    expected_std = np.std(oof[prediction_cols].to_numpy(dtype=float), axis=1, ddof=1)
    np.testing.assert_allclose(oof["ensemble_std"], expected_std, rtol=1e-10, atol=1e-10)
    expected_missing_std = np.std(
        missing[prediction_cols].to_numpy(dtype=float), axis=1, ddof=1
    )
    np.testing.assert_allclose(
        missing["y_imputed_std"], expected_missing_std, rtol=1e-10, atol=1e-10
    )

    for frame in (oof_rel, missing_rel):
        for repeat_id in range(2):
            expected_distance = np.sqrt(
                2.0
                * (
                    1.0
                    - frame[f"mean_tanimoto_similarity_repeat{repeat_id}"].to_numpy(
                        dtype=float
                    )
                )
            )
            np.testing.assert_allclose(
                frame[f"D_repeat{repeat_id}"],
                expected_distance,
                rtol=1e-7,
                atol=1e-7,
            )
            expected_g = np.sqrt(
                (
                    frame[f"D_repeat{repeat_id}"].to_numpy(dtype=float) ** 2
                    + frame[f"A_repeat{repeat_id}"].to_numpy(dtype=float) ** 2
                )
                / 2.0
            )
            np.testing.assert_allclose(
                frame[f"G_repeat{repeat_id}"],
                expected_g,
                rtol=1e-7,
                atol=1e-7,
            )

        aggregate_pairs = {
            "mean_tanimoto_similarity": "mean_tanimoto_similarity_repeat",
            "tanimoto_distance": "D_repeat",
            "local_consistency_risk": "A_repeat",
            "structure_local_risk": "G_repeat",
        }
        for aggregate, prefix in aggregate_pairs.items():
            expected_aggregate = frame[
                [f"{prefix}0", f"{prefix}1"]
            ].mean(axis=1)
            np.testing.assert_allclose(
                frame[aggregate], expected_aggregate, rtol=1e-7, atol=1e-7
            )

        nrmse = frame["property_nrmse"].to_numpy(dtype=float)
        sigma = frame["ensemble_std_normalized"].to_numpy(dtype=float)
        distance = frame["tanimoto_distance"].to_numpy(dtype=float)
        expected_model_risk = nrmse / (1.0 + nrmse)
        expected_ensemble_risk = sigma / (1.0 + sigma)
        expected_interaction = distance * sigma
        expected_interaction_risk = expected_interaction / (1.0 + expected_interaction)
        expected_score = 1.0 - (
            expected_model_risk
            + expected_ensemble_risk
            + expected_interaction_risk
        ) / 3.0
        np.testing.assert_allclose(
            frame["nrmse_risk_fraction"], expected_model_risk, rtol=1e-7, atol=1e-7
        )
        np.testing.assert_allclose(
            frame["ensemble_risk_fraction"], expected_ensemble_risk, rtol=1e-7, atol=1e-7
        )
        np.testing.assert_allclose(
            frame["distance_sigma_risk"], expected_interaction, rtol=1e-7, atol=1e-7
        )
        np.testing.assert_allclose(
            frame["distance_sigma_risk_fraction"],
            expected_interaction_risk,
            rtol=1e-7,
            atol=1e-7,
        )
        assert set(frame["reliability_formula_version"]) == {
            "fraction_equal_distance_sigma_v1"
        }
        np.testing.assert_allclose(
            frame["reliability_score"], expected_score, rtol=1e-7, atol=1e-7
        )

    assert metadata["method"] == "strict_repeated_cross_fitting"
    assert metadata["observed_std_ddof"] == 1
    assert metadata["ensemble_std_ddof"] == 1
    return {
        "property": prop,
        "oof_rows": len(oof),
        "missing_rows": len(missing),
        "n_repeats": metadata["n_repeats"],
        "n_folds": metadata["n_folds"],
        "mean_oof_reliability": float(oof_rel["reliability_score"].mean()),
        "mean_missing_reliability": float(missing_rel["reliability_score"].mean()),
    }


def main() -> int:
    args = parse_args()
    temporary = tempfile.TemporaryDirectory(prefix="v11_smoke_")
    temp_root = Path(temporary.name)
    output_dir = temp_root / "outputs"
    prop = "dielectric_constant"
    try:
        run(
            [
                sys.executable,
                str(SCRIPT_DIR / "01_train_repeated5fold_cv.py"),
                "--data-path",
                str(args.data_path),
                "--output-dir",
                str(output_dir),
                "--checkpoint-path",
                str(args.checkpoint_path),
                "--properties",
                prop,
                "--device",
                "cpu",
                "--min-free-gpu-gb",
                "0",
                "--n-rows",
                "150",
                "--n-repeats",
                "2",
                "--n-folds",
                "2",
                "--repeat-seeds",
                "42",
                "43",
            ]
        )
        run(
            [
                sys.executable,
                str(SCRIPT_DIR / "02_compute_reliability.py"),
                "--data-path",
                str(args.data_path),
                "--output-dir",
                str(output_dir),
                "--properties",
                prop,
                "--query-chunk-size",
                "32",
            ]
        )
        summary = verify_outputs(output_dir, prop)
        summary["smoke_test"] = "ok"
        summary["rows_used"] = 150
        summary["device"] = "cpu"
        if args.keep_temp:
            kept = ROOT / "smoke_test_artifacts"
            if kept.exists():
                raise RuntimeError(f"Refusing to replace existing directory: {kept}")
            temporary.cleanup = lambda: None
            temp_root.replace(kept)
            summary["artifacts"] = str(kept)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    finally:
        if not args.keep_temp:
            temporary.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
