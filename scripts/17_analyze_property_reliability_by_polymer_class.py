"""Summarize missing-cell PSI by polymer class for every property.

Only imputed cells from *_missing_reliability.csv are analyzed. The class map
comes from the V11 representative-property assignment table. ``Mixed/unclear``
is excluded, while every other displayed class is retained.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLASS_FILE = (
    PROJECT_ROOT
    / "outputs/21_representative_property_reliability_umaps_05_10/"
    / "electrical_dielectric_constant/"
    / "polymer_class_and_dielectric_constant_reliability_assignments.csv.gz"
)
DEFAULT_DATASET = PROJECT_ROOT / "input_data/merged dataset/dataset_v10_maccs_drop30_invalidpsmiles.csv"
DEFAULT_RELIABILITY_DIR = PROJECT_ROOT / "outputs/03_missing_predictions"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/31_property_reliability_by_polymer_class"

EXCLUDED_CLASSES = {"Mixed/unclear"}
CLASS_ORDER = [
    "Polyimide",
    "Polyamide",
    "Polyester",
    "Polyether",
    "Polyacrylate",
    "Fluoropolymer",
    "Polysulfone",
    "Polysiloxane",
    "Polyurethane",
    "Polyolefin",
    "Polycarbonate",
    "Polystyrene",
]


def property_from_path(path: Path) -> str:
    suffix = "_missing_reliability.csv"
    if not path.name.endswith(suffix):
        raise ValueError(f"Unexpected reliability filename: {path.name}")
    return path.name[: -len(suffix)]


def load_class_map(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, usecols=["polymer_id", "display_class"])
    table["polymer_id"] = pd.to_numeric(table["polymer_id"], errors="raise").astype(int)
    table["display_class"] = table["display_class"].astype(str)
    if table["polymer_id"].duplicated().any():
        raise ValueError(f"Duplicate polymer_id values in class file: {path}")
    return table


def summarize_one(path: Path, class_map: pd.DataFrame) -> pd.DataFrame:
    property_name = property_from_path(path)
    frame = pd.read_csv(path, usecols=["polymer_id", "reliability_score"])
    frame["polymer_id"] = pd.to_numeric(frame["polymer_id"], errors="coerce")
    frame["reliability_score"] = pd.to_numeric(frame["reliability_score"], errors="coerce")
    frame = frame.dropna(subset=["polymer_id", "reliability_score"]).copy()
    frame["polymer_id"] = frame["polymer_id"].astype(int)
    frame = frame.merge(class_map, on="polymer_id", how="inner", validate="many_to_one")
    frame = frame.loc[~frame["display_class"].isin(EXCLUDED_CLASSES)].copy()
    if frame.empty:
        return pd.DataFrame()
    grouped = (
        frame.groupby("display_class", sort=False)["reliability_score"]
        .agg(
            n_imputed="size",
            psi_mean="mean",
            psi_median="median",
            psi_std="std",
            psi_min="min",
            psi_p10=lambda values: values.quantile(0.10),
            psi_p25=lambda values: values.quantile(0.25),
            psi_p75=lambda values: values.quantile(0.75),
            psi_p90=lambda values: values.quantile(0.90),
            psi_max="max",
        )
        .reset_index()
    )
    grouped.insert(0, "property", property_name)
    grouped["source_file"] = path.name
    return grouped


def observed_support_table(dataset_path: Path, class_map: pd.DataFrame, properties: list[str]) -> pd.DataFrame:
    """Count true observed values in each class for every analyzed property."""
    dataset = pd.read_csv(dataset_path)
    if "smiles" in dataset.columns:
        dataset = dataset.drop(columns=["smiles"])
    dataset.insert(0, "polymer_id", np.arange(len(dataset), dtype=int))
    base = class_map.merge(dataset[["polymer_id"]], on="polymer_id", how="inner", validate="one_to_one")
    class_sizes = base.groupby("display_class").size().rename("n_class_total")
    rows: list[pd.DataFrame] = []
    for property_name in properties:
        if property_name not in dataset.columns:
            continue
        values = dataset[["polymer_id", property_name]].copy()
        values["observed"] = pd.to_numeric(values[property_name], errors="coerce").notna()
        values = values[["polymer_id", "observed"]].merge(
            class_map, on="polymer_id", how="inner", validate="many_to_one"
        )
        values = values.loc[~values["display_class"].isin(EXCLUDED_CLASSES)]
        grouped = values.groupby("display_class")["observed"].agg(
            n_observed="sum", n_total="size"
        ).reset_index()
        grouped.insert(0, "property", property_name)
        grouped["n_class_total"] = grouped["display_class"].map(class_sizes)
        grouped["observed_fraction"] = grouped["n_observed"] / grouped["n_total"]
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def add_within_property_comparisons(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for property_name, group in summary.groupby("property", sort=False):
        current = group.copy()
        medians = current.set_index("display_class")["psi_median"]
        means = current.set_index("display_class")["psi_mean"]
        current["property_psi_median"] = float(current["psi_median"].median())
        current["property_psi_mean"] = float(current["psi_mean"].median())
        current["other_class_median"] = current["display_class"].map(
            lambda name: float(medians.drop(index=name).median())
            if len(medians) > 1
            else np.nan
        )
        current["other_class_mean"] = current["display_class"].map(
            lambda name: float(means.drop(index=name).median())
            if len(means) > 1
            else np.nan
        )
        # Positive values mean that this class is lower than the other-class
        # reference, so larger values are stronger low-PSI candidates.
        current["median_gap_vs_other"] = (
            current["other_class_median"] - current["psi_median"]
        )
        current["mean_gap_vs_other"] = current["other_class_mean"] - current["psi_mean"]
        current["median_rank_within_property"] = current["psi_median"].rank(
            ascending=True, method="min"
        ).astype(int)
        current["mean_rank_within_property"] = current["psi_mean"].rank(
            ascending=True, method="min"
        ).astype(int)
        rows.append(current)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--class-file", type=Path, default=DEFAULT_CLASS_FILE)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--reliability-dir", type=Path, default=DEFAULT_RELIABILITY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--candidate-min-n",
        type=int,
        default=50,
        help="Sample-size flag used only for screening candidates; all rows remain in the full table.",
    )
    args = parser.parse_args()
    args.class_file = args.class_file.resolve()
    args.dataset = args.dataset.resolve()
    args.reliability_dir = args.reliability_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    class_map = load_class_map(args.class_file)
    included_classes = [
        name for name in CLASS_ORDER
        if name in set(class_map["display_class"])
    ]
    # Preserve any additional non-unclear class rather than silently dropping it.
    additional_classes = sorted(
        set(class_map["display_class"]) - set(EXCLUDED_CLASSES) - set(included_classes)
    )
    included_classes.extend(additional_classes)

    files = sorted(args.reliability_dir.glob("*_missing_reliability.csv"))
    if not files:
        raise FileNotFoundError(f"No *_missing_reliability.csv files found in {args.reliability_dir}")

    summaries: list[pd.DataFrame] = []
    for index, path in enumerate(files, start=1):
        print(f"[{index:02d}/{len(files):02d}] {path.name}", flush=True)
        result = summarize_one(path, class_map)
        if not result.empty:
            summaries.append(result)

    summary = pd.concat(summaries, ignore_index=True)
    support = observed_support_table(
        args.dataset, class_map, list(summary["property"].unique())
    )
    summary = summary.merge(
        support,
        on=["property", "display_class"],
        how="left",
        validate="one_to_one",
    )
    summary = add_within_property_comparisons(summary)
    summary["candidate_n_ok"] = summary["n_imputed"] >= args.candidate_min_n
    summary["class_observed_lt_10"] = summary["n_observed"] < 10
    summary["class_observed_lt_50"] = summary["n_observed"] < 50
    summary["low_median_candidate"] = (
        summary["candidate_n_ok"] & (summary["median_rank_within_property"] == 1)
    )
    summary["low_mean_candidate"] = (
        summary["candidate_n_ok"] & (summary["mean_rank_within_property"] == 1)
    )

    property_order = list(dict.fromkeys(summary["property"].tolist()))
    category_order = [name for name in included_classes if name in set(summary["display_class"])]
    summary["property"] = pd.Categorical(summary["property"], categories=property_order, ordered=True)
    summary["display_class"] = pd.Categorical(summary["display_class"], categories=category_order, ordered=True)
    summary = summary.sort_values(["property", "median_rank_within_property", "display_class"])

    long_path = args.output_dir / "psi_by_class_property_long.csv.gz"
    summary.to_csv(long_path, index=False, compression="gzip", encoding="utf-8-sig")

    def pivot(value: str) -> pd.DataFrame:
        table = summary.pivot(index="display_class", columns="property", values=value)
        return table.reindex(index=category_order, columns=property_order)

    median_matrix = pivot("psi_median")
    mean_matrix = pivot("psi_mean")
    n_matrix = pivot("n_imputed")
    median_matrix.to_csv(args.output_dir / "psi_median_by_class_property.csv", encoding="utf-8-sig")
    mean_matrix.to_csv(args.output_dir / "psi_mean_by_class_property.csv", encoding="utf-8-sig")
    n_matrix.to_csv(args.output_dir / "psi_n_by_class_property.csv", encoding="utf-8-sig")

    lowest = (
        summary.sort_values(["property", "psi_median", "psi_mean"])
        .groupby("property", observed=True, sort=False)
        .head(1)
        .copy()
    )
    lowest.to_csv(args.output_dir / "lowest_class_per_property.csv", index=False, encoding="utf-8-sig")

    candidates = summary.sort_values(
        ["candidate_n_ok", "median_gap_vs_other", "mean_gap_vs_other"],
        ascending=[False, False, False],
    ).copy()
    candidates.to_csv(args.output_dir / "low_psi_class_candidates_ranked.csv", index=False, encoding="utf-8-sig")

    method = {
        "class_file": str(args.class_file),
        "dataset": str(args.dataset),
        "reliability_dir": str(args.reliability_dir),
        "n_reliability_files": len(files),
        "n_properties": int(summary["property"].nunique()),
        "included_classes": category_order,
        "excluded_classes": sorted(EXCLUDED_CLASSES),
        "analysis_population": "imputed cells only; reliability_score from *_missing_reliability.csv",
        "candidate_min_n": args.candidate_min_n,
        "outputs": {
            "long": str(long_path),
            "median_matrix": str(args.output_dir / "psi_median_by_class_property.csv"),
            "mean_matrix": str(args.output_dir / "psi_mean_by_class_property.csv"),
            "n_matrix": str(args.output_dir / "psi_n_by_class_property.csv"),
            "lowest_per_property": str(args.output_dir / "lowest_class_per_property.csv"),
            "ranked_candidates": str(args.output_dir / "low_psi_class_candidates_ranked.csv"),
        },
        "elapsed_seconds": round(time.time() - started, 2),
    }
    (args.output_dir / "method.json").write_text(
        json.dumps(method, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(method, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
