from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BLUE = "#173f7a"
LIGHT_BLUE = "#dce9fb"
BAR_EDGE = "#5279ad"
ORANGE = "#f4a340"
RED = "#e64b4b"
GRAY = "#6c6c6c"
GRID = "#d8dce2"


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


def assign_rank_quantile(series: pd.Series, q: int, labels: list[str]) -> pd.Series:
    ranks = series.rank(method="first")
    return pd.qcut(ranks, q=q, labels=labels)


def style_axes(ax):
    ax.grid(axis="y", color=GRID, linestyle="--", linewidth=0.8, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")
    ax.tick_params(axis="both", labelsize=11)


def add_reliability_score(df: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="outputs/02_oof_reliability_scores/Tg_K_oof_reliability.csv",
        help="Strict OOF reliability CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/13_score_error_validation_abc",
        help="Directory for ABC figure and summary tables.",
    )
    parser.add_argument("--property-name", default=None)
    parser.add_argument("--dpi", type=int, default=450)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
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
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required].replace([np.inf, -np.inf], np.nan).dropna().copy()
    df = df[df["oof_error_normalized"] >= 0].copy()
    df, observed_std, property_nrmse = add_reliability_score(df)
    n_total = len(df)
    score_col = "reliability_score"

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 9.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(15.2, 10.2))
    ax_a, ax_b, ax_c, ax_blank = axes.flatten()

    decile_labels = [f"D{i}" for i in range(1, 11)]
    df["score_decile"] = assign_rank_quantile(df[score_col], 10, decile_labels)
    decile_rows = []
    for i, lab in enumerate(decile_labels, start=1):
        sub = df[df["score_decile"] == lab]
        med = float(sub["oof_error_normalized"].median())
        mean = float(sub["oof_error_normalized"].mean())
        lo, hi = bootstrap_ci(
            sub["oof_error_normalized"].to_numpy(), fn=np.median, seed=1000 + i
        )
        decile_rows.append(
            {
                "decile": lab,
                "n": len(sub),
                "score_min": sub[score_col].min(),
                "score_max": sub[score_col].max(),
                "median_normalized_error": med,
                "mean_normalized_error": mean,
                "ci_low": lo,
                "ci_high": hi,
            }
        )
    decile_df = pd.DataFrame(decile_rows)
    x = np.arange(10)
    y = decile_df["median_normalized_error"].to_numpy()
    yerr = np.vstack(
        [y - decile_df["ci_low"].to_numpy(), decile_df["ci_high"].to_numpy() - y]
    )
    ax_a.bar(
        x,
        y,
        color=LIGHT_BLUE,
        edgecolor=BAR_EDGE,
        linewidth=1.0,
        width=0.62,
        zorder=2,
    )
    ax_a.errorbar(x, y, yerr=yerr, fmt="o", color=BLUE, capsize=3, markersize=4.8, zorder=3)
    for xi, yi in zip(x, y):
        ax_a.text(xi, yi + 0.014, f"{yi:.3f}", ha="center", va="bottom", fontsize=9)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(decile_labels)
    ax_a.set_ylabel("Median normalized absolute error")
    ax_a.set_xlabel("Reliability score decile")
    style_axes(ax_a)

    coverages = np.arange(10, 101, 10)
    ranked = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    rng = np.random.default_rng(20260827)
    random_ranked = df.iloc[rng.permutation(n_total)].reset_index(drop=True)
    b_rows = []
    for cov in coverages:
        k = max(1, int(round(n_total * cov / 100)))
        b_rows.append(
            {
                "coverage_retained_pct": cov,
                "n_retained": k,
                "confidence_ranked_mae": float(
                    ranked.iloc[:k]["oof_error_normalized"].mean()
                ),
                "random_selection_mae": float(
                    random_ranked.iloc[:k]["oof_error_normalized"].mean()
                ),
            }
        )
    b_df = pd.DataFrame(b_rows)
    ax_b.plot(
        coverages,
        b_df["confidence_ranked_mae"],
        "-o",
        color=BLUE,
        lw=1.8,
        markersize=5,
        label="Reliability-ranked selection",
    )
    ax_b.plot(
        coverages,
        b_df["random_selection_mae"],
        "--o",
        color=GRAY,
        markerfacecolor="white",
        lw=1.5,
        markersize=4.5,
        label="Random-order selection",
    )
    for idx, r in b_df.iterrows():
        offset = 0.010 if idx % 2 == 0 else 0.017
        ax_b.text(
            r.coverage_retained_pct,
            r.confidence_ranked_mae + offset,
            f"{r.confidence_ranked_mae:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
        if r.coverage_retained_pct in [10, 100]:
            ax_b.text(
                r.coverage_retained_pct,
                r.random_selection_mae + 0.012,
                f"{r.random_selection_mae:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=GRAY,
            )
    ax_b.set_xlabel("Coverage retained (%)")
    ax_b.set_ylabel("MAE on retained predictions")
    ymax_b = max(b_df["confidence_ranked_mae"].max(), b_df["random_selection_mae"].max()) * 1.30
    ax_b.set_ylim(0, ymax_b)
    ax_b.legend(frameon=False, loc="upper left")
    style_axes(ax_b)

    q_labels = ["Q1\n(closest)", "Q2", "Q3", "Q4\n(most distant)"]
    df["distance_quartile"] = assign_rank_quantile(df["tanimoto_distance"], 4, q_labels)
    c_rows = []
    for qi, qlab in enumerate(q_labels, start=1):
        sub = df[df["distance_quartile"] == qlab].copy()
        ranks = sub[score_col].rank(method="first") / len(sub)
        sub["confidence_group"] = pd.cut(
            ranks,
            bins=[0, 0.2, 0.8, 1.0],
            labels=["Low", "Medium", "High"],
            include_lowest=True,
        )
        for group in ["High", "Medium", "Low"]:
            vals = sub.loc[sub["confidence_group"] == group, "oof_error_normalized"].to_numpy()
            c_rows.append(
                {
                    "distance_quartile": f"Q{qi}",
                    "confidence_group": group,
                    "n": len(vals),
                    "mean_error": float(np.mean(vals)),
                    "median_error": float(np.median(vals)),
                    "sem": sem(vals),
                }
            )
    c_df = pd.DataFrame(c_rows)
    group_order = ["High", "Medium", "Low"]
    colors = {"High": "#356db4", "Medium": ORANGE, "Low": RED}
    offsets = {"High": -0.24, "Medium": 0.0, "Low": 0.24}
    group_labels = {
        "High": "High PSI (top 20%)",
        "Medium": "Medium PSI (middle 60%)",
        "Low": "Low PSI (bottom 20%)",
    }
    for group in group_order:
        sub = c_df[c_df["confidence_group"] == group]
        xpos = np.arange(1, 5) + offsets[group]
        ax_c.bar(
            xpos,
            sub["mean_error"],
            width=0.22,
            color=colors[group],
            edgecolor="#333333",
            linewidth=0.35,
            alpha=0.88,
            label=group_labels[group],
            zorder=2,
        )
        ax_c.errorbar(
            xpos,
            sub["mean_error"],
            yerr=sub["sem"],
            fmt="none",
            ecolor="#222222",
            elinewidth=0.8,
            capsize=2.5,
            zorder=3,
        )
        for xi, yi in zip(xpos, sub["mean_error"]):
            ax_c.text(xi, yi + 0.012, f"{yi:.3f}", ha="center", va="bottom", fontsize=9)
    ax_c.set_xticks(np.arange(1, 5))
    ax_c.set_xticklabels(q_labels)
    ax_c.set_xlabel("Structural distance quartile")
    ax_c.set_ylabel("Mean normalized absolute error")
    ymax_c = max((c_df["mean_error"] + c_df["sem"]).max() * 1.25, 0.30)
    ax_c.set_ylim(0, ymax_c)
    for qi in range(1, 5):
        n_q = int(
            c_df[
                (c_df["distance_quartile"] == f"Q{qi}")
                & (c_df["confidence_group"] == "High")
            ]["n"].iloc[0]
        )
        ax_c.text(
            qi,
            ymax_c * 0.93,
            f"n = {n_q:,}/group",
            ha="center",
            va="top",
            fontsize=9,
            style="italic",
        )
    ax_c.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=3)
    style_axes(ax_c)

    ax_blank.axis("off")

    for letter, ax in zip(["A", "B", "C"], [ax_a, ax_b, ax_c]):
        ax.text(-0.14, 1.08, letter, transform=ax.transAxes, fontsize=18, fontweight="bold")

    fig.tight_layout(rect=[0.025, 0.035, 0.995, 0.98], h_pad=5.0, w_pad=3.0)

    tag = "abc_canonical_distance_sigma"
    file_stem = (
        args.property_name or input_path.name.replace("_oof_reliability.csv", "")
    ).replace("/", "_")
    out_png = output_dir / f"{file_stem}_score_error_validation_{tag}.png"
    out_pdf = output_dir / f"{file_stem}_score_error_validation_{tag}.pdf"
    fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    decile_df.to_csv(output_dir / f"{file_stem}_panel_a_score_decile_error_{tag}.csv", index=False)
    b_df.to_csv(output_dir / f"{file_stem}_panel_b_selective_retention_{tag}.csv", index=False)
    c_df.to_csv(output_dir / f"{file_stem}_panel_c_shift_confidence_groups_{tag}.csv", index=False)

    print(
        {
            "input": str(input_path),
            "n": n_total,
            "png": str(out_png),
            "pdf": str(out_pdf),
            "panel_a_spearman_score_error": float(
                df[[score_col, "oof_error_normalized"]].corr(method="spearman").iloc[0, 1]
            ),
            "spearman_sigma_error": float(
                df[["ensemble_std_normalized_global", "oof_error_normalized"]]
                .corr(method="spearman")
                .iloc[0, 1]
            ),
            "panel_c_spearman_distance_error": float(
                df[["tanimoto_distance", "oof_error_normalized"]]
                .corr(method="spearman")
                .iloc[0, 1]
            ),
            "score_mode": "abc_canonical_fraction_equal_distance_sigma",
            "observed_std": observed_std,
            "property_nrmse": property_nrmse,
            "formula": (
                "R=1-[NRMSE/(1+NRMSE)+sigma/(1+sigma)+"
                "D*sigma/(1+D*sigma)]/3"
            ),
        }
    )


if __name__ == "__main__":
    main()
