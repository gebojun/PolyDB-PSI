from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


GRID = "#d8dce2"
BLUE = "#173f7a"
BLUE_LIGHT = "#dce9fb"
BLUE_EDGE = "#5279ad"
ORANGE = "#f4a340"
ORANGE_LIGHT = "#fde4bf"
ORANGE_EDGE = "#c9801e"
TEAL = "#4f9f95"
TEAL_LIGHT = "#d7f0ed"
TEAL_EDGE = "#3d7f76"
RED = "#e64b4b"
GRAY = "#6c6c6c"


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
    ax.tick_params(axis="both", labelsize=14)


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
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(22.5, 10.8),
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
            ax_a.text(xi, yi + 0.012, f"{yi:.3f}", ha="center", va="bottom", fontsize=9, color="#222222")

    ax_a.set_xticks(np.arange(10))
    ax_a.set_xticklabels(decile_labels)
    ax_a.set_xlabel("Reliability score decile")
    ax_a.set_ylabel("Median normalized absolute error")
    ax_a.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        fontsize=15,
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
        fontsize=15,
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
    ax_c.set_xticklabels(psi_summary["display_name"], rotation=58, ha="right", fontsize=8)
    ax_c.set_xlabel("Property")
    ax_c.set_ylabel(r"$-\mathrm{Spearman}\ \rho(e,\mathrm{PSI})$")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="#666666", linewidth=.35, label=cat) for cat, color in category_colors.items() if cat in set(psi_summary["category"])]
    ax_c.legend(handles=handles, frameon=False, loc="lower center", bbox_to_anchor=(.5, 1.02), ncol=4, fontsize=15, columnspacing=1.2)
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
    colors = {"High PSI": "#356db4", "Medium PSI": ORANGE, "Low PSI": RED}
    offsets = {"High PSI": -0.24, "Medium PSI": 0.0, "Low PSI": 0.24}
    for group in ["High PSI", "Medium PSI", "Low PSI"]:
        sub = c_df[c_df["psi_group"] == group]
        xpos = np.arange(1, 5) + offsets[group]
        ax_d.bar(xpos, sub["mean_error"], width=0.22, color=colors[group],
                 edgecolor="#333333", linewidth=0.35, alpha=0.88,
                 label=group, zorder=2)
        ax_d.errorbar(xpos, sub["mean_error"], yerr=sub["sem"], fmt="none",
                      ecolor="#222222", elinewidth=0.8, capsize=2.5, zorder=3)
        for xi, yi in zip(xpos, sub["mean_error"]):
            ax_d.text(xi, yi + 0.012, f"{yi:.3f}", ha="center", va="bottom", fontsize=9)
    ymax_c = max((c_df["mean_error"] + c_df["sem"]).max() * 1.18, 0.30)
    ax_d.set_ylim(0, ymax_c)
    ax_d.set_xticks(np.arange(1, 5))
    ax_d.set_xticklabels(q_labels)
    ax_d.set_xlabel("Tanimoto distance quartile")
    ax_d.set_ylabel("Mean normalized absolute error")
    ax_d.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02),
                ncol=3, fontsize=15, handlelength=1.4, columnspacing=1.0)
    style_axes(ax_d)
    c_df.to_csv(output_dir / "panel_d_structural_distance_psi_groups.csv", index=False)

    # D is populated above; the remaining empty slot is retained as a spacer.

    # E/F. Redraw the external validation panels in the same compact style as A-C.
    # Using the tabular results avoids importing titles and explanatory paragraphs
    # from the standalone external-validation figure.
    external_csv = Path(args.external_figure).parent / "all_7874_actual_reports_with_psi_quartiles.csv"
    quartile_csv = Path(args.external_figure).parent / "psi_quartile_report_errors_and_intervals.csv"
    if not external_csv.exists() or not quartile_csv.exists():
        raise FileNotFoundError("External validation CSV results are required for E/F")

    ext = pd.read_csv(external_csv, low_memory=False)
    label_colors = {
        "gold": "#173f7a",
        "yellow": "#f4a340",
        "black": "#8b929b",
        "red": "#e64b4b",
    }
    for label, group in ext.groupby("polymetrix_label", sort=False):
        ax_e.scatter(
            group["Tg_K"], group["y_pred_mean"], s=25, alpha=0.48,
            color=label_colors.get(label, "#777777"), edgecolors="none",
            label=label.capitalize(), rasterized=True,
        )
    lims = [
        float(min(ext["Tg_K"].min(), ext["y_pred_mean"].min())),
        float(max(ext["Tg_K"].max(), ext["y_pred_mean"].max())),
    ]
    pad = (lims[1] - lims[0]) * 0.025
    lims = [lims[0] - pad, lims[1] + pad]
    ax_e.plot(lims, lims, color="#555555", linestyle="--", linewidth=1.1, zorder=1)
    # Use a common round range so the external panel is immediately readable.
    ax_e.set_xlim(0, 800)
    ax_e.set_ylim(0, 800)
    ax_e.set_xlabel(r"Reported $T_{\mathrm{g}}$ (K)")
    ax_e.set_ylabel(r"Predicted $T_{\mathrm{g}}$ (K)")
    metrics_text = (
        f"n = {len(ext):,} reports\n"
        f"MAE = {ext['report_absolute_error_K'].mean():.2f} K\n"
        f"$R^2$ = {1 - np.sum((ext['Tg_K'] - ext['y_pred_mean'])**2) / np.sum((ext['Tg_K'] - ext['Tg_K'].mean())**2):.3f}"
    )
    ax_e.text(0.04, 0.96, metrics_text, transform=ax_e.transAxes,
              ha="left", va="top", fontsize=12, color="#333333",
              bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=2.5))
    ax_e.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02),
                ncol=4, fontsize=16, handletextpad=0.35, columnspacing=0.7,
                markerscale=1.15)
    ax_e.tick_params(axis="both", labelsize=14)
    ax_e.set_box_aspect(0.88)
    style_axes(ax_e)

    qdf = pd.read_csv(quartile_csv)
    qdf = qdf.iloc[:4].copy()
    x_f = np.arange(len(qdf))
    ax_f.bar(x_f, qdf["mae_K"], width=0.62,
             color=["#bccddd", "#96b4d3", "#638fbf", "#245f9b"],
             edgecolor="#486681", linewidth=0.5, zorder=3)
    ax_f.errorbar(x_f, qdf["mae_K"],
                  yerr=[qdf["mae_K"] - qdf["ci95_low_K"],
                        qdf["ci95_high_K"] - qdf["mae_K"]],
                  fmt="none", color="#35404a", capsize=4, linewidth=1.2, zorder=4)
    overall_mae = float(ext["report_absolute_error_K"].mean())
    ax_f.axhline(overall_mae, color="#777777", linestyle="--", linewidth=1.1,
                 label=f"All reports ({overall_mae:.1f} K)")
    for xi, yi in zip(x_f, qdf["mae_K"]):
        ax_f.text(xi, yi + 1.0, f"{yi:.2f}", ha="center", va="bottom", fontsize=9)
    ax_f.set_xticks(x_f)
    ax_f.set_xticklabels(["0–25%", "25–50%", "50–75%", "75–100%"])
    ax_f.set_xlabel("PSI percentile")
    ax_f.set_ylabel("Mean absolute error (K)")
    ax_f.set_ylim(0, max(float(qdf["ci95_high_K"].max()) * 1.18, overall_mae * 1.18))
    ax_f.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02),
                fontsize=11, handlelength=1.3)
    ax_f.set_box_aspect(0.86)
    style_axes(ax_f)
    ax_d.set_box_aspect(0.78)

    for letter, ax in zip(["A", "B", "C", "D", "E", "F"], [ax_a, ax_b, ax_c, ax_d, ax_e, ax_f]):
        ax.text(-0.14, 1.08, letter, transform=ax.transAxes, fontsize=18, fontweight="bold")

    fig.tight_layout(rect=[0.02, 0.035, 0.995, 0.98], h_pad=2.4, w_pad=0.35)

    out_png = output_dir / "cross_property_score_error_validation_abc.png"
    out_pdf = output_dir / "cross_property_score_error_validation_abc.pdf"
    fig.savefig(out_png, dpi=args.dpi)
    fig.savefig(out_pdf)
    plt.close(fig)

    tg["a_df"].to_csv(output_dir / "tg_panel_a_score_decile_error.csv", index=False)
    dc["a_df"].to_csv(output_dir / "dielectric_constant_panel_a_score_decile_error.csv", index=False)
    pd.DataFrame(tg["b_rows"]).to_csv(output_dir / "tg_panel_b_selective_retention.csv", index=False)
    pd.DataFrame(dc["b_rows"]).to_csv(output_dir / "dielectric_constant_panel_b_selective_retention.csv", index=False)

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
            "panel_c_metric": "mean normalized OOF error by Tanimoto-distance quartile and PSI group",
            "mapping_docx": str(Path(args.mapping_docx).resolve()),
            "formula": "R=1-[NRMSE/(1+NRMSE)+sigma/(1+sigma)+D*sigma/(1+D*sigma)]/3",
        }
    )


if __name__ == "__main__":
    main()
