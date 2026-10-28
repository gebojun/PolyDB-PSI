"""Build a self-contained interactive Tg UMAP display widget."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_LAYOUTS = {
    "当前布局：k=25, min_dist=0.12": "coordinates_maccs_k_25_min_dist_0_12_current.csv",
    "紧凑布局：k=25, min_dist=0.08": "coordinates_maccs_k_25_min_dist_0_08.csv",
    "紧凑布局：k=25, min_dist=0.08, spread=0.5": "coordinates_maccs_k_25_min_dist_0_08_spread_0_5.csv",
}


def read_assignments(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        return pd.read_csv(
            handle,
            usecols=[
                "polymer_id",
                "smiles",
                "Tg_K",
                "display_class",
                "property_reliability_score",
            ],
        )


def read_coordinates(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["polymer_id", "dimension_1", "dimension_2"])
    return frame.rename(columns={"dimension_1": "x", "dimension_2": "y"})


def build_data(assignments: pd.DataFrame, coordinates_dir: Path) -> dict[str, object]:
    layouts: dict[str, dict[str, list[float]]] = {}
    merged_reference: pd.DataFrame | None = None
    for label, filename in DEFAULT_LAYOUTS.items():
        coords = read_coordinates(coordinates_dir / filename)
        if merged_reference is None:
            merged_reference = coords[["polymer_id"]].copy()
        elif not coords["polymer_id"].equals(merged_reference["polymer_id"]):
            raise ValueError(f"Layout polymer order differs: {filename}")
        layouts[label] = {
            "x": np.round(coords["x"].to_numpy(dtype=float), 5).tolist(),
            "y": np.round(coords["y"].to_numpy(dtype=float), 5).tolist(),
        }

    if merged_reference is None:
        raise ValueError("No layout was loaded")
    if not assignments["polymer_id"].equals(merged_reference["polymer_id"]):
        raise ValueError("Assignment and coordinate polymer order do not match")

    classes = sorted(assignments["display_class"].astype(str).unique())
    counts = assignments["display_class"].value_counts()
    top_classes = counts.sort_values(ascending=False, kind="stable").head(12).index.tolist()
    class_ids = {name: index for index, name in enumerate(classes)}
    observed = pd.to_numeric(assignments["Tg_K"], errors="coerce").notna()
    psi = pd.to_numeric(assignments["property_reliability_score"], errors="coerce")
    return {
        "n": int(len(assignments)),
        "classes": classes,
        "top_classes": top_classes,
        "class_ids": [class_ids[value] for value in assignments["display_class"].astype(str)],
        "observed": observed.astype(int).tolist(),
        "psi": [None if not np.isfinite(value) else round(float(value), 6) for value in psi],
        "layouts": layouts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--assignment-file",
        type=Path,
        default=Path(
            "D:/nexpoly/v11_repeated5fold_cv_reliability_validation/outputs/"
            "21_representative_property_reliability_umaps_05_10/thermal_Tg_K/"
            "polymer_class_and_Tg_K_reliability_assignments.csv.gz"
        ),
    )
    parser.add_argument(
        "--coordinates-dir",
        type=Path,
        default=Path(
            "D:/nexpoly/v11_repeated5fold_cv_reliability_validation/outputs/"
            "14_structural_embedding_comparison"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "D:/nexpoly/v11_repeated5fold_cv_reliability_validation/outputs/"
            "29_tg_umap_interactive_widget"
        ),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assignments = read_assignments(args.assignment_file)
    data = build_data(assignments, args.coordinates_dir)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    template = Path(__file__).with_name("17_tg_umap_interactive_widget.html")
    html = template.read_text(encoding="utf-8").replace("__TG_DATA_JSON__", payload)
    output = args.output_dir / "tg_umap_interactive_widget.html"
    output.write_text(html, encoding="utf-8")
    (args.output_dir / "data_summary.json").write_text(
        json.dumps(
            {
                "n": data["n"],
                "top_classes": data["top_classes"],
                "layouts": list(DEFAULT_LAYOUTS),
                "source_assignment": str(args.assignment_file),
                "source_coordinates_dir": str(args.coordinates_dir),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
