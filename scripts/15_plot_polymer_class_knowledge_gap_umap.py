"""Draw polymer-class and property-specific reliability maps on shared UMAP coordinates.

Panel B assigns an auditable structural class to every valid pSMILES and colors
the MACCS-Jaccard UMAP by that class. Panel C reuses exactly the same coordinates
and colors polymers with a missing target property by its cell-level reliability
score. Observed structures are shown as a neutral background and are not assigned
an imputation score. No cross-property aggregation is used in this figure.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from matplotlib.colors import Normalize, to_rgba
from matplotlib.lines import Line2D
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import MACCSkeys


CLASS_RULE_ORDER = [
    "polysiloxane",
    "polyimide",
    "polyurethane",
    "polyamide",
    "polycarbonate",
    "polyacrylate",
    "polyester",
    "polysulfone",
    "fluoropolymer",
    "polystyrene",
    "polyether",
    "polyolefin",
]

CLASS_DISPLAY = {
    "polyimide": "Polyimide",
    "polyurethane": "Polyurethane",
    "polyamide": "Polyamide",
    "polyester": "Polyester",
    "polycarbonate": "Polycarbonate",
    "polyether": "Polyether",
    "polysulfone": "Polysulfone",
    "polysiloxane": "Polysiloxane",
    "polyolefin": "Polyolefin",
    "fluoropolymer": "Fluoropolymer",
    "polyacrylate": "Polyacrylate",
    "polystyrene": "Polystyrene",
    "aromatic_polymer": "Mixed/unclear",
    "heteroatom_polymer": "Mixed/unclear",
    "halogenated_polymer": "Mixed/unclear",
    "other_polymer": "Mixed/unclear",
    "invalid_smiles": "Mixed/unclear",
}

DISPLAY_ORDER = [
    "Polyimide",
    "Polyurethane",
    "Polyamide",
    "Polyester",
    "Polycarbonate",
    "Polyether",
    "Polysulfone",
    "Polysiloxane",
    "Polyolefin",
    "Fluoropolymer",
    "Polyacrylate",
    "Polystyrene",
    "Mixed/unclear",
]

CLASS_COLORS = {
    # Deliberately separated hues for the 12 most common displayed classes.
    "Polyimide": "#6F2DA8",
    "Polyurethane": "#E66101",
    "Polyamide": "#1F77B4",
    "Polyester": "#2CA02C",
    "Polycarbonate": "#D4A017",
    "Polyether": "#8C564B",
    "Polysulfone": "#D81B60",
    "Polysiloxane": "#00A6A6",
    "Polyolefin": "#7A8B00",
    "Fluoropolymer": "#003F5C",
    "Polyacrylate": "#E41A1C",
    "Polystyrene": "#FF7FAD",
    "Mixed/unclear": "#A7ABB0",
    "Other classes": "#D6D9DD",
}

SMARTS = {
    "imide": "[CX3](=O)[NX3][CX3](=O)",
    "urethane": "[NX3][CX3](=O)[OX2]",
    "amide": "[CX3](=O)[NX3]",
    "carbonate": "[OX2][CX3](=O)[OX2]",
    "ester": "[CX3](=O)[OX2][#6]",
    "ether": "[OD2]([#6])[#6]",
    "sulfone": "[SX4](=O)(=O)",
    "silicon": "[Si]",
    "fluorine": "[F]",
    "aromatic": "a",
}
PATTERNS = {key: Chem.MolFromSmarts(value) for key, value in SMARTS.items()}


def normalize_attachment_placeholders(smiles: str) -> tuple[str, bool]:
    """Map non-standard [R], [R1], ... attachment labels to wildcard atoms."""
    normalized = re.sub(r"\[R\d*\]", "[*]", smiles)
    return normalized, normalized != smiles


def classify_smiles(smiles: str) -> dict[str, object]:
    """Apply the established V10 SMARTS/string class hierarchy."""
    structure_smiles, normalized_attachment_labels = normalize_attachment_placeholders(smiles)
    mol = Chem.MolFromSmiles(structure_smiles)
    if mol is None:
        return {
            "primary_class": "invalid_smiles",
            "display_class": "Mixed/unclear",
            "classification_tags": "invalid_smiles",
            "classification_rule": "invalid_smiles",
            "attachment_labels_normalized": normalized_attachment_labels,
        }

    flags = {
        key: bool(pattern is not None and mol.HasSubstructMatch(pattern))
        for key, pattern in PATTERNS.items()
    }
    symbols = {atom.GetSymbol() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 0}
    halogen = bool(symbols.intersection({"F", "Cl", "Br", "I"}))
    hetero = any(symbol not in {"C", "H"} for symbol in symbols)
    simple_hydrocarbon = symbols.issubset({"C", "H"})
    lower = smiles.lower()
    acrylate_like = ("C(=O)O" in smiles or "C(=O)OC" in smiles) and (
        "C(C*)" in smiles
        or "C([*])" in smiles
        or "C(C)(*" in smiles
        or "C(*)" in smiles
    )
    styrene_like = flags["aromatic"] and simple_hydrocarbon and (
        "c1ccccc1" in lower or "cccc" in lower
    )
    siloxane_like = flags["silicon"] and "O" in symbols

    tags = [key for key, present in flags.items() if present]
    if halogen:
        tags.append("halogen")
    if acrylate_like:
        tags.append("acrylate_like")
    if styrene_like:
        tags.append("styrene_like")
    if simple_hydrocarbon:
        tags.append("hydrocarbon")

    if siloxane_like:
        primary = "polysiloxane"
    elif flags["imide"]:
        primary = "polyimide"
    elif flags["urethane"]:
        primary = "polyurethane"
    elif flags["amide"]:
        primary = "polyamide"
    elif flags["carbonate"]:
        primary = "polycarbonate"
    elif acrylate_like:
        primary = "polyacrylate"
    elif flags["ester"]:
        primary = "polyester"
    elif flags["sulfone"]:
        primary = "polysulfone"
    elif flags["fluorine"]:
        primary = "fluoropolymer"
    elif styrene_like:
        primary = "polystyrene"
    elif flags["ether"]:
        primary = "polyether"
    elif simple_hydrocarbon and not flags["aromatic"]:
        primary = "polyolefin"
    elif flags["aromatic"]:
        primary = "aromatic_polymer"
    elif halogen:
        primary = "halogenated_polymer"
    elif hetero:
        primary = "heteroatom_polymer"
    else:
        primary = "other_polymer"

    return {
        "primary_class": primary,
        "display_class": CLASS_DISPLAY[primary],
        "classification_tags": ";".join(sorted(set(tags))),
        "classification_rule": primary,
        "attachment_labels_normalized": normalized_attachment_labels,
    }


def load_or_classify(
    smiles: pd.Series,
    existing_class_csv: Path | None,
) -> pd.DataFrame:
    if existing_class_csv is not None and existing_class_csv.exists():
        existing = pd.read_csv(existing_class_csv)
        if (
            len(existing) == len(smiles)
            and "smiles" in existing
            and existing["smiles"].astype(str).equals(smiles.astype(str))
            and "primary_class" in existing
        ):
            result = pd.DataFrame(
                {
                    "primary_class": existing["primary_class"].astype(str),
                    "classification_tags": existing.get(
                        "tags", pd.Series("", index=existing.index)
                    ).fillna(""),
                }
            )
            result["display_class"] = result["primary_class"].map(CLASS_DISPLAY).fillna(
                "Mixed/unclear"
            )
            result["classification_rule"] = result["primary_class"]
            result["attachment_labels_normalized"] = False
            invalid = result["primary_class"].eq("invalid_smiles")
            if invalid.any():
                repaired = pd.DataFrame(
                    [classify_smiles(str(value)) for value in smiles.loc[invalid]]
                ).set_index(smiles.loc[invalid].index)
                for column in repaired.columns:
                    result.loc[invalid, column] = repaired[column]
            return result

    rows = [classify_smiles(str(value)) for value in smiles]
    return pd.DataFrame(rows)


def maccs_matrix(smiles: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.zeros((len(smiles), 167), dtype=np.uint8)
    valid = np.zeros(len(smiles), dtype=bool)
    for row, value in enumerate(smiles.astype(str)):
        structure_smiles, _ = normalize_attachment_placeholders(value)
        mol = Chem.MolFromSmiles(structure_smiles)
        if mol is None:
            continue
        fp = MACCSkeys.GenMACCSKeys(mol)
        DataStructs.ConvertToNumpyArray(fp, matrix[row])
        valid[row] = True
    return matrix, valid


def aggregate_property_reliability(
    missing_dir: Path,
    n_polymers: int,
    property_name: str,
    reliability_assignment_file: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = reliability_assignment_file or (
        missing_dir / f"{property_name}_missing_reliability.csv"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing {property_name} reliability file: {path}")

    score_column = (
        "property_reliability_score"
        if reliability_assignment_file is not None
        else "reliability_score"
    )
    frame = pd.read_csv(path, usecols=["polymer_id", score_column]).rename(
        columns={score_column: "reliability_score"}
    )
    if frame["polymer_id"].duplicated().any():
        raise ValueError(
            f"{property_name} reliability file contains duplicate polymer_id values"
        )
    frame = frame[
        frame["polymer_id"].between(0, n_polymers - 1)
        & frame["reliability_score"].notna()
    ].copy()
    frame["polymer_id"] = frame["polymer_id"].astype(int)

    score = np.full(n_polymers, np.nan, dtype=float)
    ids = frame["polymer_id"].to_numpy(dtype=int)
    score[ids] = frame["reliability_score"].to_numpy(dtype=float)
    polymer_stats = pd.DataFrame(
        {
            "polymer_id": np.arange(n_polymers, dtype=int),
            "property_reliability_score": score,
        }
    )
    property_summary = pd.DataFrame(
        [
            {
                "property": property_name,
                "n_scored_missing": len(frame),
                "mean_reliability": float(frame["reliability_score"].mean()),
                "median_reliability": float(frame["reliability_score"].median()),
                "p10_reliability": float(frame["reliability_score"].quantile(0.10)),
                "p90_reliability": float(frame["reliability_score"].quantile(0.90)),
            }
        ]
    )
    print(
        f"Loaded {property_name}-specific reliability: "
        f"{len(frame):,} scored missing cells",
        flush=True,
    )
    return polymer_stats, property_summary


def representative_xy(frame: pd.DataFrame) -> tuple[float, float]:
    center = frame[["umap_1", "umap_2"]].median().to_numpy(dtype=float)
    values = frame[["umap_1", "umap_2"]].to_numpy(dtype=float)
    idx = int(np.argmin(np.sum((values - center) ** 2, axis=1)))
    return float(values[idx, 0]), float(values[idx, 1])


def outward_offset(
    xy: tuple[float, float],
    global_center: np.ndarray,
    x_span: float,
    y_span: float,
    scale: float = 0.10,
) -> tuple[float, float]:
    direction = np.asarray(xy, dtype=float) - global_center
    normalized = np.array(
        [direction[0] / max(x_span, 1e-9), direction[1] / max(y_span, 1e-9)]
    )
    length = float(np.linalg.norm(normalized))
    if length < 1e-8:
        normalized = np.array([0.7, 0.7])
        length = float(np.linalg.norm(normalized))
    normalized /= length
    return (
        float(xy[0] + normalized[0] * x_span * scale),
        float(xy[1] + normalized[1] * y_span * scale),
    )


def style_axis(ax: plt.Axes, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("UMAP 1", fontsize=12, fontweight="bold")
    ax.set_ylabel("UMAP 2", fontsize=12, fontweight="bold")
    ax.tick_params(axis="both", labelsize=9, length=3.5, width=0.8, color="#555555")
    ax.set_facecolor("#FCFCFD")
    ax.grid(color="#C9CDD3", linewidth=0.55, alpha=0.33, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#5E6268")
        spine.set_linewidth(0.8)


def display_property_name(property_name: str) -> str:
    known_names = {
        "Tg_K": "Tg",
        "dielectric_constant": "Dielectric constant",
        "dielectric_loss": "Dielectric loss",
        "dielectric_tandelta": "Dielectric tan delta",
    }
    return known_names.get(property_name, property_name.replace("_", " "))


def top_display_classes(table: pd.DataFrame, n: int = 12) -> list[str]:
    """Return the n most numerous displayed classes in deterministic order."""
    counts = table["display_class"].value_counts()
    return list(counts.sort_values(ascending=False, kind="stable").head(n).index)


def draw_reading_figure(
    table: pd.DataFrame,
    output_png: Path,
    output_pdf: Path,
    display_quantile: float,
    point_size: float,
    figure_width: float,
    figure_height: float,
    property_name: str,
) -> int:
    """Draw a compact reading view without defining a new gap metric."""
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    x = table["umap_1"].to_numpy(dtype=float)
    y = table["umap_2"].to_numpy(dtype=float)
    x_low, x_high = np.quantile(x, [display_quantile, 1.0 - display_quantile])
    y_low, y_high = np.quantile(y, [display_quantile, 1.0 - display_quantile])
    x_margin = (x_high - x_low) * 0.018
    y_margin = (y_high - y_low) * 0.022
    xlim = (float(x_low - x_margin), float(x_high + x_margin))
    ylim = (float(y_low - y_margin), float(y_high + y_margin))
    display_mask = (
        (x >= x_low)
        & (x <= x_high)
        & (y >= y_low)
        & (y <= y_high)
    )
    shown = table.loc[display_mask].copy()
    shown_fraction = float(display_mask.mean())

    fig, (ax_b, ax_c) = plt.subplots(
        1,
        2,
        figsize=(figure_width, figure_height),
        dpi=220,
        gridspec_kw={"width_ratios": [1.0, 1.04]},
    )

    # B. Structural class map.
    rng = np.random.default_rng(20260828)
    shown_random = shown.iloc[rng.permutation(len(shown))]
    retained_classes = set(top_display_classes(table, n=12))
    shown_random = shown_random.loc[
        shown_random["display_class"].isin(retained_classes)
    ].copy()
    rgba = np.asarray(
        [
            to_rgba(
                CLASS_COLORS[name],
                0.86 if name != "Mixed/unclear" else 0.42,
            )
            for name in shown_random["display_class"]
        ]
    )
    ax_b.scatter(
        shown_random["umap_1"],
        shown_random["umap_2"],
        s=point_size,
        c=rgba,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )
    ax_b.set_title("Polymer classes", fontsize=12.5, fontweight="bold", pad=8)
    style_axis(ax_b, xlim, ylim)

    legend_classes = top_display_classes(table, n=12)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=CLASS_COLORS[name],
            markeredgecolor="none",
            markersize=6.0,
            label=name,
        )
        for name in legend_classes
    ]
    ax_b.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        frameon=False,
        ncol=4,
        fontsize=9.0,
        handletextpad=0.25,
        columnspacing=0.9,
        labelspacing=0.3,
    )

    # C. Property-specific cell-level reliability. Observed values are neutral.
    property_observed = (
        pd.to_numeric(table[property_name], errors="coerce").notna().to_numpy()
    )
    reliability = table["property_reliability_score"].to_numpy(dtype=float)
    observed = display_mask & property_observed
    missing_scored = display_mask & ~property_observed & np.isfinite(reliability)
    missing_unscored = display_mask & ~property_observed & ~np.isfinite(reliability)
    if np.any(observed):
        ax_c.scatter(
            x[observed],
            y[observed],
            s=point_size,
            c="#B7BCC3",
            alpha=0.34,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )
    if np.any(missing_unscored):
        ax_c.scatter(
            x[missing_unscored],
            y[missing_unscored],
            s=point_size * 0.85,
            color="#777D85",
            marker="x",
            linewidths=0.75,
            rasterized=True,
            zorder=2,
        )

    order = np.where(missing_scored)[0]
    order = order[np.argsort(reliability[order])[::-1]]
    scatter = ax_c.scatter(
        x[order],
        y[order],
        c=reliability[order],
        cmap="RdYlGn",
        norm=Normalize(vmin=0.5, vmax=1.0, clip=True),
        s=point_size,
        alpha=0.86,
        linewidths=0,
        rasterized=True,
        zorder=3,
    )
    property_label = display_property_name(property_name)
    ax_c.set_title(
        f"{property_label} imputation reliability",
        fontsize=12.5,
        fontweight="bold",
        pad=8,
    )
    style_axis(ax_c, xlim, ylim)
    cbar = fig.colorbar(scatter, ax=ax_c, fraction=0.043, pad=0.025)
    cbar.set_label(
        f"{property_label} cell-level reliability score",
        fontsize=8.8,
        labelpad=6,
    )
    cbar.set_ticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    cbar.ax.tick_params(labelsize=7.5)
    cbar.outline.set_linewidth(0.6)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#B7BCC3",
            markeredgecolor="none",
            alpha=0.65,
            markersize=8.0,
            label=f"Observed {property_label}",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#55A868",
            markeredgecolor="none",
            markersize=8.0,
            label=f"Prediction {property_label}",
        ),
    ]
    ax_c.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        frameon=False,
        ncol=2,
        fontsize=9.0,
        handletextpad=0.3,
        columnspacing=1.0,
        labelspacing=0.35,
    )
    fig.subplots_adjust(
        left=0.065,
        right=0.955,
        top=0.88,
        bottom=0.16,
        wspace=0.26,
    )
    fig.savefig(output_png, dpi=450, bbox_inches="tight")
    fig.savefig(output_pdf, dpi=450, bbox_inches="tight")
    plt.close(fig)
    return int(display_mask.sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="input_data/merged dataset/dataset_v10_maccs_drop30_invalidpsmiles.csv",
    )
    parser.add_argument(
        "--property",
        default="Tg_K",
        help="One target property whose missing-cell reliability should color panel C.",
    )
    parser.add_argument(
        "--missing-dir",
        default="outputs/03_missing_predictions",
    )
    parser.add_argument(
        "--reliability-assignment-file",
        default=None,
        help=(
            "Optional historical assignment snapshot. When supplied, the "
            "property_reliability_score column from this file is used for panel C."
        ),
    )
    parser.add_argument(
        "--existing-class-csv",
        default=(
            "D:/nexpoly/visualization/results/12_v10_motif_category_heatmap/"
            "polymer_classes_v10.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/18_polymer_class_tg_reliability_umap_compact",
    )
    parser.add_argument(
        "--coordinates-csv",
        default=None,
        help=(
            "Optional precomputed unsupervised coordinates. The file must contain "
            "polymer_id, smiles, and either umap_1/umap_2 or dimension_1/dimension_2."
        ),
    )
    parser.add_argument("--n-neighbors", type=int, default=25)
    parser.add_argument("--min-dist", type=float, default=0.12)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--display-quantile",
        type=float,
        default=0.05,
        help=(
            "Trim this fraction from each UMAP axis for the reading view; "
            "coordinates and assignments remain unchanged."
        ),
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=42.0,
        help="Scatter marker area for the figure, in matplotlib points squared.",
    )
    parser.add_argument(
        "--figure-width",
        type=float,
        default=11.4,
        help="Figure width in inches.",
    )
    parser.add_argument(
        "--figure-height",
        type=float,
        default=6.15,
        help="Figure height in inches.",
    )
    parser.add_argument("--force-umap", action="store_true")
    args = parser.parse_args()

    if not 0 <= args.display_quantile < 0.5:
        raise ValueError("--display-quantile must be in [0, 0.5)")
    if args.point_size <= 0:
        raise ValueError("--point-size must be positive")
    if args.figure_width <= 0 or args.figure_height <= 0:
        raise ValueError("--figure-width and --figure-height must be positive")

    RDLogger.DisableLog("rdApp.*")
    dataset_path = Path(args.dataset).resolve()
    missing_dir = Path(args.missing_dir).resolve()
    reliability_assignment_file = (
        Path(args.reliability_assignment_file).resolve()
        if args.reliability_assignment_file
        else None
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    class_path = Path(args.existing_class_csv).resolve() if args.existing_class_csv else None

    if not re.fullmatch(r"[A-Za-z0-9_]+", args.property):
        raise ValueError("--property must contain only letters, numbers, and underscores")
    dataset = pd.read_csv(dataset_path, usecols=["smiles", args.property])
    dataset.insert(0, "polymer_id", np.arange(len(dataset), dtype=int))
    if dataset["smiles"].duplicated().any():
        raise ValueError("Expected one row per unique pSMILES, but duplicates were found")

    print(f"Classifying {len(dataset):,} unique polymers...", flush=True)
    classes = load_or_classify(dataset["smiles"], class_path)
    table = pd.concat([dataset.reset_index(drop=True), classes.reset_index(drop=True)], axis=1)

    stats, property_summary = aggregate_property_reliability(
        missing_dir=missing_dir,
        n_polymers=len(dataset),
        property_name=args.property,
        reliability_assignment_file=reliability_assignment_file,
    )
    table = table.merge(stats, on="polymer_id", how="left", validate="one_to_one")

    coord_path = output_dir / "shared_maccs_jaccard_umap_coordinates.csv"
    can_reuse = False
    external_coordinate_path = (
        Path(args.coordinates_csv).resolve() if args.coordinates_csv else None
    )
    if external_coordinate_path is not None:
        coords_old = pd.read_csv(external_coordinate_path)
        if {"dimension_1", "dimension_2"}.issubset(coords_old.columns):
            coords_old = coords_old.rename(
                columns={"dimension_1": "umap_1", "dimension_2": "umap_2"}
            )
        required_coords = {"polymer_id", "smiles", "umap_1", "umap_2"}
        if not required_coords.issubset(coords_old.columns):
            raise ValueError(
                f"Coordinate file lacks required columns: {required_coords - set(coords_old.columns)}"
            )
        can_reuse = (
            len(coords_old) == len(table)
            and coords_old["polymer_id"].to_numpy(dtype=int).tolist()
            == table["polymer_id"].to_numpy(dtype=int).tolist()
            and coords_old["smiles"].astype(str).equals(table["smiles"].astype(str))
        )
        if not can_reuse:
            raise ValueError("External coordinates do not match the V11 polymer table")
        table["umap_1"] = coords_old["umap_1"].to_numpy(dtype=float)
        table["umap_2"] = coords_old["umap_2"].to_numpy(dtype=float)
        print(f"Loaded selected unsupervised coordinates: {external_coordinate_path}", flush=True)
    elif coord_path.exists() and not args.force_umap:
        coords_old = pd.read_csv(coord_path)
        can_reuse = (
            len(coords_old) == len(table)
            and {"polymer_id", "smiles", "umap_1", "umap_2"}.issubset(coords_old.columns)
            and coords_old["smiles"].astype(str).equals(table["smiles"].astype(str))
        )
        if can_reuse:
            table["umap_1"] = coords_old["umap_1"].to_numpy(dtype=float)
            table["umap_2"] = coords_old["umap_2"].to_numpy(dtype=float)
            print(f"Reused shared UMAP coordinates: {coord_path}", flush=True)

    if not can_reuse:
        print("Computing 167-bit MACCS fingerprints...", flush=True)
        fingerprints, valid_fingerprints = maccs_matrix(table["smiles"])
        if not np.all(valid_fingerprints):
            raise ValueError(
                f"UMAP input contains {int((~valid_fingerprints).sum())} invalid pSMILES"
            )
        print(
            "Fitting shared UMAP (metric=Jaccard, equivalent to binary Tanimoto distance)...",
            flush=True,
        )
        reducer = umap.UMAP(
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            n_components=2,
            metric="jaccard",
            random_state=args.random_state,
            n_jobs=1,
            low_memory=True,
            verbose=True,
        )
        coords = reducer.fit_transform(fingerprints.astype(bool))
        table["umap_1"] = coords[:, 0]
        table["umap_2"] = coords[:, 1]

    coordinate_table = table[["polymer_id", "smiles", "umap_1", "umap_2"]]
    coordinate_table.to_csv(coord_path, index=False, encoding="utf-8-sig")

    valid_reliability = table["property_reliability_score"].notna().to_numpy()

    property_slug = args.property
    assignments_path = (
        output_dir / f"polymer_class_and_{property_slug}_reliability_assignments.csv.gz"
    )
    table.to_csv(assignments_path, index=False, compression="gzip", encoding="utf-8-sig")
    property_summary.to_csv(
        output_dir / f"{property_slug}_reliability_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    class_summary = (
        table.groupby("display_class", as_index=False)
        .agg(
            n_polymers=("polymer_id", "size"),
            n_property_observed=(
                (args.property),
                lambda values: int(values.notna().sum()),
            ),
            n_property_missing_scored=("property_reliability_score", "count"),
            mean_property_reliability=("property_reliability_score", "mean"),
            median_property_reliability=("property_reliability_score", "median"),
        )
        .sort_values("n_polymers", ascending=False)
    )
    class_summary.to_csv(
        output_dir / f"polymer_class_{property_slug}_reliability_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    output_png = output_dir / f"polymer_class_and_{property_slug}_reliability_umap.png"
    output_pdf = output_dir / f"polymer_class_and_{property_slug}_reliability_umap.pdf"
    display_quantile_used = max(args.display_quantile, 0.05)
    point_size_used = args.point_size
    displayed_count = draw_reading_figure(
        table=table,
        output_png=output_png,
        output_pdf=output_pdf,
        display_quantile=display_quantile_used,
        point_size=point_size_used,
        figure_width=args.figure_width,
        figure_height=args.figure_height,
        property_name=args.property,
    )

    method = {
        "dataset": str(dataset_path),
        "n_unique_polymers": len(table),
        "classification": {
            "basis": "RDKit-parsed pSMILES with established V10 SMARTS/string priority rules",
            "priority": CLASS_RULE_ORDER,
            "collapsed_to_mixed_unclear": [
                "aromatic_polymer",
                "heteroatom_polymer",
                "halogenated_polymer",
                "other_polymer",
                "invalid_smiles",
            ],
        },
        "class_display": {
            "retained_class_count": 12,
            "retained_classes": top_display_classes(table, n=12),
            "excluded_classes": [
                name
                for name in sorted(table["display_class"].unique())
                if name not in set(top_display_classes(table, n=12))
            ],
            "excluded_classes_rendered_as": "omitted from the structural-class panel",
        },
        "embedding": {
            "fingerprint": "RDKit MACCS keys, 167 bits",
            "method": "UMAP",
            "metric": "Jaccard on binary fingerprints (1 - binary Tanimoto similarity)",
            "n_neighbors": 200 if external_coordinate_path is not None else args.n_neighbors,
            "min_dist": 0.35 if external_coordinate_path is not None else args.min_dist,
            "initialization": "PCA" if external_coordinate_path is not None else "spectral",
            "random_state": args.random_state,
            "coordinate_source": (
                str(external_coordinate_path)
                if external_coordinate_path is not None
                else "computed_or_cached_by_this_script"
            ),
        },
        "reliability_display": {
            "cell_score_column": "reliability_score",
            "property": args.property,
            "polymer_color_value": (
                "the individual target-property missing-cell reliability_score; "
                "no cross-property "
                "aggregation is used"
            ),
            "observed_property_display": (
                "neutral gray background; not assigned an imputation score"
            ),
            "color_scale": {
                "cmap": "RdYlGn",
                "vmin": 0.5,
                "vmax": 1.0,
                "below_vmin_behavior": "values below 0.5 are clipped to the dark-red endpoint",
            },
            "n_properties": 1,
        },
        "display": {
            "display_quantile_per_axis": display_quantile_used,
            "point_size": point_size_used,
            "figure_width_inches": args.figure_width,
            "figure_height_inches": args.figure_height,
            "displayed_polymer_count": displayed_count,
            "display_note": (
                "Only the central robust coordinate window is shown for readability; "
                "the embedding, scores, and full-space assignments are unchanged."
                if display_quantile_used > 0
                else "Full coordinate range is shown."
            ),
        },
        "outputs": {
            "png": str(output_png),
            "pdf": str(output_pdf),
            "assignments": str(assignments_path),
            "coordinates": str(coord_path),
        },
    }
    (output_dir / "method.json").write_text(
        json.dumps(method, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "n_unique_polymers": len(table),
                "n_properties": len(property_summary),
                "n_polymers_with_missing_reliability": int(valid_reliability.sum()),
                "property_reliability": {
                    "property": args.property,
                    "mean": float(
                        table.loc[valid_reliability, "property_reliability_score"].mean()
                    ),
                    "median": float(
                        table.loc[valid_reliability, "property_reliability_score"].median()
                    ),
                    "p10": float(
                        table.loc[valid_reliability, "property_reliability_score"].quantile(
                            0.10
                        )
                    ),
                    "min": float(
                        table.loc[valid_reliability, "property_reliability_score"].min()
                    ),
                },
                "class_counts": {
                    str(key): int(value)
                    for key, value in table["display_class"].value_counts().items()
                },
                "png": str(output_png),
                "pdf": str(output_pdf),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
