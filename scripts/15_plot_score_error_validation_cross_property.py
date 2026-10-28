from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox


GRID = "#d8dce2"
# Core colors follow the low-saturation category palette used in panel C.
BLUE = "#8697c3"
BLUE_LIGHT = "#d7deee"
BLUE_EDGE = "#667da9"
ORANGE = "#d1847f"
ORANGE_LIGHT = "#fdcbc8"
ORANGE_EDGE = "#d1847f"
TEAL = "#4f9f95"
TEAL_LIGHT = "#d7f0ed"
TEAL_EDGE = "#3d7f76"
RED = "#e64b4b"
GRAY = "#6c6c6c"
AXIS_LABEL_SIZE = 16
TICK_SIZE = 14
LEGEND_SIZE = 15
VALUE_SIZE = 11


def bootstrap_ci(values: np.ndarray, fn=np.median, n_boot: int = 500, seed: int = 20260827):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    if values.size == 1:
        v = float(fn(values))
        return v, v
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    stats = np.apply_along_axis(fn, 1, values[idx])
    return np.percentile(stats, [2.5, 97.5])


def sem(values: np.ndarray):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def load_property_mapping(path: Path) -> pd.DataFrame:
    from docx import Document

    document = Document(path)
    for table in document.tables:
        if not table.rows:
            continue
        headers = [cell.text.strip() for cell in table.rows[0].cells]
        required = {"类别", "性质", "映射标签"}
        if not required.issubset(headers):
            continue
        indices = {name: headers.index(name) for name in required}
        rows = []
        for row in table.rows[1:]:
            cells = [cell.text.strip() for cell in row.cells]
            if len(cells) > max(indices.values()):
                rows.append(
                    {
                        "category": cells[indices["类别"]],
                        "property_name": cells[indices["性质"]],
                        "display_name": cells[indices["映射标签"]],
                    }
                )
        return pd.DataFrame(rows)
    raise ValueError(f"Mapping table not found in {path}")


def assign_rank_quantile(series: pd.Series, q: int, labels: list[str]) -> pd.Series:
    ranks = series.rank(method="first")
    return pd.qcut(ranks, q=q, labels=labels)


def style_axes(ax):
    ax.grid(axis="y", color=GRID, linestyle="--", linewidth=0.8, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.xaxis.label.set_size(AXIS_LABEL_SIZE)
    ax.yaxis.label.set_size(AXIS_LABEL_SIZE)


def export_panel(fig, ax, output_dir: Path, stem: str, dpi: int):
    """Export exactly the visible panel, including labels, legend, and panel letter."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    artists = [ax]
    legend = ax.get_legend()
    if legend is not None:
        artists.append(legend)
    bboxes = [artist.get_tightbbox(renderer) for artist in artists]
    panel_bbox = Bbox.union([bbox for bbox in bboxes if bbox is not None])
    panel_bbox = panel_bbox.transformed(fig.dpi_scale_trans.inverted()).expanded(1.05, 1.08)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"{stem}.{suffix}", dpi=dpi,
                    bbox_inches=panel_bbox, facecolor="white")


def build_score(df: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    observed_std = float(df["y_true"].std(ddof=1))
    property_nrmse = float(df["property_nrmse"].iloc[0])
    df = df.copy()
    df["ensemble_std_normalized_global"] = df["ensemble_std"] / observed_std
    df["nrmse_risk_fraction"] = property_nrmse / (1.0 + property_nrmse)
    df["ensemble_risk_fraction"] = df["ensemble_std_normalized_global"] / (
        1.0 + df["ensemble_std_normalized_global"]
    )
    df["distance_sigma_risk"] = (
        df["tanimoto_distance"] * df["ensemble_std_normalized_global"]
    )
    df["distance_sigma_risk_fraction"] = df["distance_sigma_risk"] / (
        1.0 + df["distance_sigma_risk"]
    )
    df["reliability_score"] = 1.0 - (
        df["nrmse_risk_fraction"]
        + df["ensemble_risk_fraction"]
        + df["distance_sigma_risk_fraction"]
    ) / 3.0
    return df, observed_std, property_nrmse


def load_property(path: Path, name: str, label: str, color: str, light: str, edge: str):
    required = [
        "y_true",
        "y_pred_oof",
        "oof_error",
        "oof_error_normalized",
        "property_nrmse",
        "ensemble_std",
        "local_mean",
        "tanimoto_distance",
    ]
    df = pd.read_csv(path)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")
    df = df[required].replace([np.inf, -np.inf], np.nan).dropna().copy()
    df = df[df["oof_error_normalized"] >= 0].copy()
    df, observed_std, property_nrmse = build_score(df)
    return {
        "name": name,
        "label": label,
        "color": color,
        "light": light,
        "edge": edge,
        "df": df,
        "observed_std": observed_std,
        "property_nrmse": property_nrmse,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tg-input",
        default="outputs/02_oof_reliability_scores/Tg_K_oof_reliability.csv",
    )
    parser.add_argument(
        "--dielectric-input",
        default="outputs/02_oof_reliability_scores/dielectric_constant_oof_reliability.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/14_score_error_validation_cross_property",
    )
    parser.add_argument(
        "--mapping-docx",
        default=r"C:\Users\ASUS\Desktop\补充材料.docx",
    )
    parser.add_argument(
        "--external-figure",
        default=r"D:\nexpoly\new visualization\output fig\52_external_red30_psi_quartiles\external_red30_parity_and_psi_quartiles.png",
    )
    parser.add_argument("--dpi", type=int, default=450)
    args = parser.parse_args()

    tg = load_property(
        Path(args.tg_input), "Tg", r"$T_{\mathrm{g}}$", BLUE, BLUE_LIGHT, BLUE_EDGE
    )
    dc = load_property(
        Path(args.dielectric_input),
        "dielectric_constant",
        r"$D_k$",
        ORANGE,
        ORANGE_LIGHT,
        ORANGE_EDGE,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "axes.labelsize": AXIS_LABEL_SIZE,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(22.5, 12.3),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.85]},
    )
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = axes.flatten()

    # A. Cross-property score-error relationship.
    decile_labels = [f"D{i}" for i in range(1, 11)]
    for offset, prop in [(-0.18, tg), (0.18, dc)]:
        df = prop["df"].copy()
        score_col = "reliability_score"
        df["score_decile"] = assign_rank_quantile(df[score_col], 10, decile_labels)
        rows = []
        for i, lab in enumerate(decile_labels, start=1):
            sub = df[df["score_decile"] == lab]
            med = float(sub["oof_error_normalized"].median())
            lo, hi = bootstrap_ci(sub["oof_error_normalized"].to_numpy(), fn=np.median, seed=1000 + i)
            rows.append(
                {
                    "decile": lab,
                    "n": len(sub),
                    "median_error": med,
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )
        prop["a_df"] = pd.DataFrame(rows)
        x = np.arange(10) + offset
        y = prop["a_df"]["median_error"].to_numpy()
        yerr = np.vstack([y - prop["a_df"]["ci_low"].to_numpy(), prop["a_df"]["ci_high"].to_numpy() - y])
        ax_a.bar(
            x,
            y,
            width=0.32,
            color=prop["light"],
            edgecolor=prop["edge"],
            linewidth=1.0,
            zorder=2,
            label=prop["label"],
        )
        ax_a.errorbar(x, y, yerr=yerr, fmt="o", color=prop["edge"], capsize=3, markersize=4.5, zorder=3)
        for xi, yi in zip(x, y):
            ax_a.text(xi, yi + 0.012, f"{yi:.3f}", ha="center", va="bottom", fontsize=VALUE_SIZE, color="#222222")

    ax_a.set_xticks(np.arange(10))
    ax_a.set_xticklabels(decile_labels)
    ax_a.set_xlabel("Reliability score decile")
    ax_a.set_ylabel("Median normalized absolute error")
    ax_a.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        fontsize=LEGEND_SIZE,
        handlelength=1.5,
        columnspacing=1.5,
    )
    style_axes(ax_a)

    # B. Cross-property selective retention.
    coverages = np.arange(10, 101, 10)
    method_styles = {
        "ranked": "-",
        "random": "--",
    }
    for prop, marker in [(tg, "o"), (dc, "s")]:
        df = prop["df"].copy()
        score_col = "reliability_score"
        ranked = df.sort_values(score_col, ascending=False).reset_index(drop=True)
        rng = np.random.default_rng(20260827)
        random_ranked = df.iloc[rng.permutation(len(df))].reset_index(drop=True)
        prop["b_rows"] = []
        for cov in coverages:
            k = max(1, int(round(len(df) * cov / 100)))
            prop["b_rows"].append(
                {
                    "coverage": cov,
                    "ranked_mae": float(ranked.iloc[:k]["oof_error_normalized"].mean()),
                    "random_mae": float(random_ranked.iloc[:k]["oof_error_normalized"].mean()),
                }
            )
        b_df = pd.DataFrame(prop["b_rows"])
        ax_b.plot(
            coverages,
            b_df["ranked_mae"],
            method_styles["ranked"],
            color=prop["color"],
            marker=marker,
            markersize=5,
            lw=1.9,
            label=f"{prop['name']} top",
        )
        ax_b.plot(
            coverages,
            b_df["random_mae"],
            method_styles["random"],
            color=prop["color"],
            marker=marker,
            markerfacecolor="white",
            markersize=4.5,
            lw=1.5,
            alpha=0.85,
            label=f"{prop['name']} rand",
        )

    ax_b.set_xlabel("Coverage retained (%)")
    ax_b.set_ylabel("MAE on retained predictions")
    ymax_b = max(
        max(line.get_ydata()) for line in ax_b.lines
    ) * 1.22
    ax_b.set_ylim(0, ymax_b)
    # Use compact legend labels so the enlarged legend remains inside panel B.
    legend_handles, legend_labels = ax_b.get_legend_handles_labels()
    compact_labels = ["Tg top", "Tg rand", r"$D_k$ top", r"$D_k$ rand"]
    ax_b.legend(
        legend_handles,
        compact_labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        fontsize=LEGEND_SIZE,
        handlelength=1.4,
        columnspacing=1.0,
    )
    style_axes(ax_b)

    # C. Cross-property PSI-error rank association.
    psi_paths = sorted(Path(args.tg_input).parent.glob("*_oof_reliability.csv"))
    psi_rows = []
    for path in psi_paths:
        data = pd.read_csv(path, usecols=["reliability_score", "oof_error_normalized"], low_memory=False).replace([np.inf, -np.inf], np.nan).dropna()
        if len(data) >= 3 and data["reliability_score"].nunique() > 1 and data["oof_error_normalized"].nunique() > 1:
            rho = float(data["reliability_score"].rank().corr(data["oof_error_normalized"].rank()))
            psi_rows.append({"property_name": path.name.replace("_oof_reliability.csv", ""), "rho_error_psi": rho, "n": len(data)})
    psi_summary = load_property_mapping(Path(args.mapping_docx)).merge(pd.DataFrame(psi_rows), on="property_name", how="inner")
    psi_summary = psi_summary.sort_values(["category", "n"], ascending=[True, False]).groupby("category", sort=False).head(5)
    psi_summary = psi_summary[psi_summary["display_name"] != "E_g"].reset_index(drop=True)
    x_c = np.arange(len(psi_summary), dtype=float) * 1.25
    category_colors = {"Electrical":"#fdcbc8", "Mechanical":"#a3e1f8", "Optical":"#f7f9cd", "Solution":"#8dd7d4", "Stability":"#fdbad1", "Surface":"#aff3be", "Thermal":"#8697c3", "Transport":"#fec6a1"}
    ax_c.bar(x_c, -psi_summary["rho_error_psi"], color=[category_colors.get(c, "#cccccc") for c in psi_summary["category"]], edgecolor="#666666", linewidth=.35, zorder=3)
    ax_c.set_ylim(0, 1.0)
    ax_c.set_xticks(x_c)
    ax_c.set_xticklabels(psi_summary["display_name"], rotation=58, ha="right", fontsize=TICK_SIZE)
    ax_c.set_xlabel("Property")
    ax_c.set_ylabel(r"$-\mathrm{Spearman}\ \rho(e,\mathrm{PSI})$")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="#666666", linewidth=.35, label=cat) for cat, color in category_colors.items() if cat in set(psi_summary["category"])]
    ax_c.legend(handles=handles, frameon=False, loc="lower center", bbox_to_anchor=(.5, 1.02), ncol=4, fontsize=LEGEND_SIZE, columnspacing=1.2)
    style_axes(ax_c)

    # D. Structural-distance stress test for Tg.
    # This is the original four-quartile panel; the legend describes PSI
    # strata explicitly instead of calling them generic reliability groups.
    cdf = tg["df"].copy()
    q_labels = ["Q1\n(closest)", "Q2", "Q3", "Q4\n(most distant)"]
    cdf["distance_quartile"] = assign_rank_quantile(cdf["tanimoto_distance"], 4, q_labels)
    ranks = cdf["reliability_score"].rank(method="first") / len(cdf)
    cdf["psi_group"] = pd.cut(
        ranks, bins=[0, 0.2, 0.8, 1.0],
        labels=["Low PSI", "Medium PSI", "High PSI"], include_lowest=True,
    )
    c_rows = []
    for qi, qlab in enumerate(q_labels, start=1):
        sub = cdf[cdf["distance_quartile"] == qlab]
        for group in ["High PSI", "Medium PSI", "Low PSI"]:
            vals = sub.loc[sub["psi_group"] == group, "oof_error_normalized"].to_numpy()
            c_rows.append({"distance_quartile": f"Q{qi}", "psi_group": group,
                           "mean_error": float(np.mean(vals)), "sem": sem(vals), "n": len(vals)})
    c_df = pd.DataFrame(c_rows)
    colors = {"High PSI": "#8697c3", "Medium PSI": "#fec6a1", "Low PSI": "#f3a6bd"}
    offsets = {"High PSI": -0.24, "Medium PSI": 0.0, "Low PSI": 0.24}
    for group in ["High PSI", "Medium PSI", "Low PSI"]:
        sub = c_df[c_df["psi_group"] == group]
        xpos = np.arange(1, 5) + offsets[group]
        ax_d.bar(xpos, sub["mean_error"], width=0.22, color=colors[group],
                 edgecolor="#333333", linewidth=0.35, alpha=0.88,
                 label=group, zorder=2)
        ax_d.errorbar(xpos, sub["mean_error"], yerr=sub["sem"], fmt="none",
                      ecolor="#222222", elinewidth=0.8, capsize=2.5, zorder=3)
        for xi, yi, uncertainty in zip(xpos, sub["mean_error"], sub["sem"]):
            ax_d.text(xi, yi + uncertainty + 0.009, f"{yi:.3f}",
                      ha="center", va="bottom", fontsize=VALUE_SIZE)
    ymax_c = max((c_df["mean_error"] + c_df["sem"]).max() * 1.18, 0.30)
    ax_d.set_ylim(0, ymax_c)
    ax_d.set_xticks(np.arange(1, 5))
    ax_d.set_xticklabels(q_labels)
    ax_d.set_xlabel("Tanimoto distance quartile")
    ax_d.set_ylabel("Mean normalized absolute error")
    ax_d.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02),
                ncol=3, fontsize=LEGEND_SIZE, handlelength=1.4, columnspacing=1.0)
    style_axes(ax_d)
    c_df.to_csv(output_dir / "panel_d_structural_distance_psi_groups.csv", index=False)

    # E/F. Redraw the external validation panels in the same compact style as A-C.
    # Using the tabular results avoids importing titles and explanatory paragraphs
    # from the standalone external-validation figure.
    external_csv = Path(args.external_figure).parent / "all_7874_actual_reports_with_psi_quartiles.csv"
    quartile_csv = Path(args.external_figure).parent / "psi_quartile_report_errors_and_intervals.csv"
    if not external_csv.exists() or not quartile_csv.exists():
        raise FileNotFoundError("External validation CSV results are required for E/F")

    ext = pd.read_csv(external_csv, low_memory=False)
    if len(ext) != 7874 or ext["PSMILES"].nunique() != 7367:
        raise ValueError("External figure requires the complete saved report population")
    expected_labels = {"gold": 492, "yellow": 264, "black": 7088, "red": 30}
    if ext["polymetrix_label"].value_counts().to_dict() != expected_labels:
        raise ValueError("External author-label counts changed")
    report_error = np.abs(ext["Tg_K"] - ext["y_pred_mean"])
    if not np.allclose(report_error, ext["report_absolute_error_K"], rtol=0, atol=1e-9):
        raise ValueError("External error export differs from actual report coordinates")
    label_colors = {
        "gold": "#e8c86c",
        "yellow": "#f7e59a",
        "black": "#c7cdd4",
        "red": "#ef9fa6",
    }
    label_edges = {
        "gold": "#b89b42",
        "yellow": "#c8ae58",
        "black": "#a5adb5",
        "red": "#c56e78",
    }
    for label in ["black", "yellow", "gold", "red"]:
        group = ext.loc[ext["polymetrix_label"].eq(label)]
        ax_e.scatter(
            group["Tg_K"], group["y_pred_mean"],
            s=20 if label == "black" else 90 if label == "red" else 42,
            alpha=.34 if label == "black" else .92,
            marker="*" if label == "red" else "o", color=label_colors[label],
            edgecolors=label_edges[label], linewidths=.45,
            zorder=7 if label == "red" else 2 if label == "black" else 4,
            rasterized=True,
        )
    ax_e.plot([0, 800], [0, 800], color="#555555", linestyle="--", linewidth=1.1, zorder=1)
    # Use a common round range so the external panel is immediately readable.
    ax_e.set_xlim(0, 800)
    ax_e.set_ylim(0, 800)
    ax_e.set_xticks(np.arange(0, 801, 200))
    ax_e.set_yticks(np.arange(0, 801, 200))
    ax_e.set_xlabel(r"Reported $T_{\mathrm{g}}$ (K)")
    ax_e.set_ylabel(r"Predicted $T_{\mathrm{g}}$ (K)")
    external_mae = float(report_error.mean())
    external_rmse = float(np.sqrt(np.mean(report_error**2)))
    external_r2 = float(1 - np.sum(report_error**2) / np.sum((ext["Tg_K"]-ext["Tg_K"].mean())**2))
    metrics_text = (
        f"n = {len(ext):,} reports\n"
        f"MAE = {external_mae:.2f} K\n"
        f"RMSE = {external_rmse:.2f} K\n"
        f"$R^2$ = {external_r2:.3f}"
    )
    ax_e.text(0.04, 0.96, metrics_text, transform=ax_e.transAxes,
              ha="left", va="top", fontsize=TICK_SIZE, color="#333333",
              bbox=dict(facecolor="white", edgecolor="none", alpha=0.94, pad=2.5))
    from matplotlib.lines import Line2D
    author_handles = [Line2D([], [], linestyle="none", markerfacecolor=label_colors[label],
        markeredgecolor=label_edges[label], markeredgewidth=.7,
        marker="*" if label == "red" else "o", markersize=12 if label == "red" else 9,
        label=f"{label.title()} ({expected_labels[label]:,})")
        for label in ["gold", "yellow", "black", "red"]]
    ax_e.legend(handles=author_handles, frameon=True, facecolor="white", edgecolor="none",
                framealpha=.94, loc="lower right", bbox_to_anchor=(.985, .025),
                ncol=2, fontsize=LEGEND_SIZE, handletextpad=0.35, columnspacing=1.1,
                handlelength=1.0, borderaxespad=.2)
    ax_e.tick_params(axis="both", labelsize=TICK_SIZE)
    style_axes(ax_e)

    qdf = pd.read_csv(quartile_csv)
    qdf = qdf.iloc[:4].copy()
    if qdf["n_reports"].sum() != len(ext) or qdf["n_structures"].sum() != ext["PSMILES"].nunique():
        raise ValueError("External quartile summary differs from E population")
    if not np.isclose(np.average(qdf["mae_K"], weights=qdf["n_reports"]), external_mae, rtol=0, atol=1e-9):
        raise ValueError("External quartile weighted MAE differs from E")
    x_f = np.arange(len(qdf))
    ax_f.bar(x_f, qdf["mae_K"], width=0.62,
             color=["#d9edf5", "#a3e1f8", "#9eb8d2", "#8697c3"],
             edgecolor="#667da9", linewidth=0.5, zorder=3)
    ax_f.errorbar(x_f, qdf["mae_K"],
                  yerr=[qdf["mae_K"] - qdf["ci95_low_K"],
                        qdf["ci95_high_K"] - qdf["mae_K"]],
                  fmt="none", color="#35404a", capsize=4, linewidth=1.2, zorder=4)
    overall_mae = external_mae
    ax_f.axhline(overall_mae, color="#777777", linestyle="--", linewidth=1.1,
                 label=f"All reports: {overall_mae:.2f} K")
    for xi, yi, upper, n in zip(x_f, qdf["mae_K"], qdf["ci95_high_K"], qdf["n_reports"]):
        ax_f.text(xi, upper + 0.85, f"{yi:.2f}\nn={int(n):,}",
                  ha="center", va="bottom", fontsize=VALUE_SIZE, zorder=6,
                  bbox=dict(facecolor="white", edgecolor="none", alpha=.95, pad=.2))
    ax_f.set_xticks(x_f)
    ax_f.set_xticklabels(["0–25%", "25–50%", "50–75%", "75–100%"])
    ax_f.set_xlabel("PSI percentile")
    ax_f.set_ylabel("Mean absolute error (K)")
    ax_f.set_ylim(0, max(float(qdf["ci95_high_K"].max()) + 7.5, overall_mae * 1.18))
    ax_f.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02),
                fontsize=LEGEND_SIZE, handlelength=1.3)
    style_axes(ax_f)

    fig.tight_layout(rect=[0.02, 0.035, 0.995, 0.98], h_pad=2.4, w_pad=0.35)

    # Independent row widths: D is 90% as wide as C, and E/F fill the remainder.
    # Do not constrain bottom box aspects: that would shrink wide subplot cells
    # and reintroduce the gaps the user asked to remove.
    left = ax_a.get_position().x0
    right = ax_c.get_position().x1
    bottom_y = ax_d.get_position().y0
    bottom_height = ax_a.get_position().height
    d_width = .90 * ax_c.get_position().width
    gap = .052
    remaining_width = right-left-d_width-2*gap
    e_width = remaining_width * .52
    f_width = remaining_width * .48
    ax_d.set_position([left, bottom_y, d_width, bottom_height])
    ax_e.set_position([left+d_width+gap, bottom_y, e_width, bottom_height])
    ax_f.set_position([right-f_width, bottom_y, f_width, bottom_height])
    for letter, ax in zip(["A", "B", "C", "D", "E", "F"], axes.flatten()):
        ax.text(-0.14, 1.08, letter, transform=ax.transAxes,
                fontsize=18, fontweight="bold", clip_on=False)

    out_png = output_dir / "psi_validation_abcdef.png"
    out_pdf = output_dir / "psi_validation_abcdef.pdf"
    fig.savefig(out_png, dpi=args.dpi)
    fig.savefig(out_pdf)
    fig.savefig(output_dir / "psi_validation_abcdef.svg")
    panel_dir = output_dir / "individual_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    panel_names = {
        "A": "panel_A_score_error_relationship",
        "B": "panel_B_selective_retention",
        "C": "panel_C_cross_property_psi_error",
        "D": "panel_D_structural_distance_psi",
        "E": "panel_E_external_prediction",
        "F": "panel_F_external_psi_quartiles",
    }
    for letter, ax in zip("ABCDEF", axes.flatten()):
        export_panel(fig, ax, panel_dir, panel_names[letter], args.dpi)
    # Retain the previous conversation's links as byte-identical aliases.
    shutil.copyfile(out_png.resolve(), (output_dir / "cross_property_score_error_validation_abc.png").resolve())
    shutil.copyfile(out_pdf.resolve(), (output_dir / "cross_property_score_error_validation_abc.pdf").resolve())
    layout = {letter: [float(x) for x in ax.get_position().bounds]
              for letter, ax in zip("ABCDEF", axes.flatten())}
    report = dict(source_thread="01a05335-da17-7452-8955-0372ece243f4",
        script=str(Path(__file__).resolve()), tg_input=str(Path(args.tg_input).resolve()),
        dielectric_input=str(Path(args.dielectric_input).resolve()),
        mapping_docx=str(Path(args.mapping_docx).resolve()), external_reports=str(external_csv.resolve()),
        external_quartile_summary=str(quartile_csv.resolve()),
        figure_size_inches=[22.5, 12.3], dpi=args.dpi, axes_bounds=layout,
        d_to_c_plot_width_ratio=float(ax_d.get_position().width/ax_c.get_position().width),
        external_report_count=len(ext), external_structure_count=int(ext["PSMILES"].nunique()),
        author_report_counts=expected_labels, external_mae_K=external_mae,
        external_rmse_K=external_rmse, external_r2=external_r2,
        external_quartile_mae_K=qdf["mae_K"].tolist(),
        external_quartile_report_counts=qdf["n_reports"].astype(int).tolist(),
        external_psi_cutoffs=json.loads((external_csv.parent / "figure_report.json").read_text(encoding="utf-8"))["psi_quartile_cutoffs"],
        external_bootstrap_unit="Whole structure, 2000 resamples; existing model outputs",
        includes_shared_training_graphs=True, new_model_training=False,
        panel_c_metric="Negative Spearman correlation between normalized OOF error and PSI",
        panel_c_property_count=len(psi_summary),
        panel_d_groups="Global Tg PSI highest 20%, middle 60%, lowest 20%, within distance quartiles",
        individual_panel_dir=str(panel_dir.resolve()),
        individual_panel_files=list(panel_names.values()))
    (output_dir / "six_panel_figure_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    plt.close(fig)

    tg["a_df"].to_csv(output_dir / "tg_panel_a_score_decile_error.csv", index=False)
    dc["a_df"].to_csv(output_dir / "dielectric_constant_panel_a_score_decile_error.csv", index=False)
    pd.DataFrame(tg["b_rows"]).to_csv(output_dir / "tg_panel_b_selective_retention.csv", index=False)
    pd.DataFrame(dc["b_rows"]).to_csv(output_dir / "dielectric_constant_panel_b_selective_retention.csv", index=False)
    psi_summary.to_csv(output_dir / "panel_c_property_psi_error_correlations.csv", index=False, encoding="utf-8-sig")
    qdf.to_csv(output_dir / "panel_f_external_psi_quartile_errors.csv", index=False)

    print(
        {
            "png": str(out_png),
            "pdf": str(out_pdf),
            "tg_score_error_spearman": float(
                tg["df"][["reliability_score", "oof_error_normalized"]].corr(method="spearman").iloc[0, 1]
            ),
            "dielectric_score_error_spearman": float(
                dc["df"][["reliability_score", "oof_error_normalized"]].corr(method="spearman").iloc[0, 1]
            ),
            "panel_c_metric": report["panel_c_metric"],
            "panel_d_metric": "Mean normalized OOF error by Tanimoto-distance quartile and PSI group",
            "d_to_c_width_ratio": report["d_to_c_plot_width_ratio"],
            "external_mae_K": external_mae,
            "external_quartile_mae_K": report["external_quartile_mae_K"],
            "mapping_docx": str(Path(args.mapping_docx).resolve()),
            "formula": "R=1-[NRMSE/(1+NRMSE)+sigma/(1+sigma)+D*sigma/(1+D*sigma)]/3",
        }
    )


if __name__ == "__main__":
    main()
