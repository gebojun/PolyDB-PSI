from __future__ import annotations

import argparse
import json
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
TEAL = "#79c7b8"
GRAY = "#6c6c6c"
GRID = "#d8dce2"


def bootstrap_ci(values: np.ndarray, fn=np.median, n_boot: int = 500, seed: int = 20260828):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        v = float(fn(values)) if values.size else np.nan
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
    return pd.qcut(series.rank(method="first"), q=q, labels=labels)


def style_axes(ax):
    ax.grid(axis="y", color=GRID, linestyle="--", linewidth=0.8, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")
    ax.tick_params(axis="both", labelsize=9)


def load_training_summary(logs_dir: Path) -> dict:
    files = sorted(logs_dir.glob("01_training_summary_*.json"))
    if not files:
        raise FileNotFoundError(f"No training summary found in {logs_dir}")
    return json.loads(files[-1].read_text(encoding="utf-8"))


def load_medium_pooled(outputs_dir: Path, min_n: int, max_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = load_training_summary(outputs_dir / "logs")
    meta_rows = []
    frames = []
    for item in summary["results"]:
        if item.get("status") != "success":
            continue
        prop = item["property_name"]
        n = int(item["n_observed"])
        if not (min_n <= n <= max_n):
            continue
        path = outputs_dir / "02_oof_reliability_scores" / f"{prop}_oof_reliability.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path).replace([np.inf, -np.inf], np.nan)
        required = [
            "polymer_id",
            "y_true",
            "y_pred_oof",
            "oof_error_normalized",
            "property_nrmse",
            "ensemble_std",
            "local_mean",
            "tanimoto_distance",
        ]
        if any(col not in df.columns for col in required):
            continue
        df = df[required].dropna().copy()
        if len(df) < min_n:
            continue
        observed_std = float(df["y_true"].std(ddof=1))
        property_nrmse = float(df["property_nrmse"].iloc[0])
        if not np.isfinite(observed_std) or observed_std <= 0:
            continue

        sigma_z = df["ensemble_std"] / observed_std
        risk_model = property_nrmse / (1.0 + property_nrmse)
        risk_sigma = sigma_z / (1.0 + sigma_z)
        distance_sigma = df["tanimoto_distance"] * sigma_z
        risk_domain = distance_sigma / (1.0 + distance_sigma)
        reliability = 1.0 - (risk_model + risk_sigma + risk_domain) / 3.0

        out = pd.DataFrame(
            {
                "property": prop,
                "category": item["category"],
                "polymer_id": df["polymer_id"].to_numpy(),
                "error_normalized": df["oof_error_normalized"].to_numpy(),
                "reliability_score": reliability.to_numpy(),
                "tanimoto_distance": df["tanimoto_distance"].to_numpy(),
                "sigma_ens_z": sigma_z.to_numpy(),
                "distance_sigma_risk": distance_sigma.to_numpy(),
                "property_nrmse": np.full(len(df), property_nrmse),
                "observed_n": np.full(len(df), len(df)),
            }
        ).replace([np.inf, -np.inf], np.nan).dropna()
        frames.append(out)
        tmp = pd.DataFrame({"R": out["reliability_score"], "err": out["error_normalized"]})
        meta_rows.append(
            {
                "property": prop,
                "category": item["category"],
                "observed_n": len(out),
                "oof_r2": item["oof_r2"],
                "oof_nrmse": item["oof_nrmse"],
                "spearman_score_error": tmp.corr(method="spearman").iloc[0, 1],
                "median_score": out["reliability_score"].median(),
                "median_error": out["error_normalized"].median(),
            }
        )
    if not frames:
        raise RuntimeError("No eligible medium-sized properties were loaded.")
    return pd.concat(frames, ignore_index=True), pd.DataFrame(meta_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/08_score_error_validation_medium_pooled"))
    parser.add_argument("--min-n", type=int, default=200)
    parser.add_argument("--max-n", type=int, default=2000)
    parser.add_argument("--dpi", type=int, default=450)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df, prop_summary = load_medium_pooled(args.outputs_dir, args.min_n, args.max_n)
    n_total = len(df)
    n_properties = prop_summary["property"].nunique()

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(15.2, 10.2))
    ax_a, ax_b, ax_c, ax_d = axes.flatten()

    # A. Score-error relationship.
    decile_labels = [f"D{i}" for i in range(1, 11)]
    df["score_decile"] = assign_rank_quantile(df["reliability_score"], 10, decile_labels)
    decile_rows = []
    for i, lab in enumerate(decile_labels, start=1):
        sub = df[df["score_decile"] == lab]
        med = float(sub["error_normalized"].median())
        lo, hi = bootstrap_ci(sub["error_normalized"].to_numpy(), fn=np.median, seed=1000 + i)
        decile_rows.append(
            {
                "decile": lab,
                "n": len(sub),
                "score_min": sub["reliability_score"].min(),
                "score_max": sub["reliability_score"].max(),
                "median_normalized_error": med,
                "mean_normalized_error": sub["error_normalized"].mean(),
                "ci_low": lo,
                "ci_high": hi,
            }
        )
    decile_df = pd.DataFrame(decile_rows)
    x = np.arange(10)
    y = decile_df["median_normalized_error"].to_numpy()
    yerr = np.vstack([y - decile_df["ci_low"].to_numpy(), decile_df["ci_high"].to_numpy() - y])
    ax_a.bar(x, y, color=LIGHT_BLUE, edgecolor=BAR_EDGE, linewidth=1.0, width=0.62, zorder=2)
    ax_a.errorbar(x, y, yerr=yerr, fmt="o", color=BLUE, capsize=3, markersize=4.5, zorder=3)
    for xi, yi in zip(x, y):
        ax_a.text(xi, yi + 0.015, f"{yi:.3f}", ha="center", va="bottom", fontsize=8)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(decile_labels)
    ax_a.set_ylabel("Median normalized absolute error")
    ax_a.set_xlabel("Reliability score decile")
    ax_a.set_title("Score-error relationship", fontweight="bold", pad=12)
    ax_a.text(
        0.5,
        -0.23,
        f"Equal-frequency pooled deciles; {n_properties} properties, total n = {n_total:,}",
        transform=ax_a.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color=BLUE,
    )
    style_axes(ax_a)

    # B. Selective retention performance.
    coverages = np.arange(10, 101, 10)
    ranked = df.sort_values("reliability_score", ascending=False).reset_index(drop=True)
    # Use one deterministic random ordering, matching the single-property
    # validation.  Repeated random subsampling would make the baseline
    # depend on an unrelated Monte Carlo average rather than on a fixed
    # retention rule.
    rng = np.random.default_rng(20260828)
    random_ranked = df.iloc[rng.permutation(n_total)].reset_index(drop=True)
    b_rows = []
    for cov in coverages:
        k = max(1, int(round(n_total * cov / 100)))
        ranked_mae = float(ranked.iloc[:k]["error_normalized"].mean())
        b_rows.append(
            {
                "coverage_retained_pct": cov,
                "n_retained": k,
                "reliability_ranked_mae": ranked_mae,
                "random_selection_mae": float(
                    random_ranked.iloc[:k]["error_normalized"].mean()
                ),
            }
        )
    b_df = pd.DataFrame(b_rows)
    ax_b.plot(coverages, b_df["reliability_ranked_mae"], "-o", color=BLUE, lw=1.8, markersize=5,
              label="Reliability-ranked selection")
    ax_b.plot(coverages, b_df["random_selection_mae"], "--o", color=GRAY, markerfacecolor="white",
              lw=1.5, markersize=4.5, label="Random selection")
    for idx, r in b_df.iterrows():
        ax_b.text(r.coverage_retained_pct, r.reliability_ranked_mae + 0.025,
                  f"{r.reliability_ranked_mae:.3f}", ha="center", va="bottom", fontsize=8)
        if r.coverage_retained_pct in [10, 100]:
            ax_b.text(r.coverage_retained_pct, r.random_selection_mae + 0.025,
                      f"{r.random_selection_mae:.3f}", ha="center", va="bottom", fontsize=8, color=GRAY)
    ymax_b = max(b_df["reliability_ranked_mae"].max(), b_df["random_selection_mae"].max()) * 1.35
    ax_b.set_ylim(0, ymax_b)
    ax_b.set_xlabel("Coverage retained (%)")
    ax_b.set_ylabel("MAE on retained predictions")
    ax_b.set_title("Selective retention performance", fontweight="bold", pad=12)
    ax_b.legend(frameon=False, loc="upper left")
    ax_b.text(0.5, -0.22, f"n full set = {n_total:,}", transform=ax_b.transAxes,
              ha="center", va="top", fontsize=9, color=BLUE)
    style_axes(ax_b)

    # C. Effect of structural shift.
    q_labels = ["Q1\n(closest)", "Q2", "Q3", "Q4\n(most distant)"]
    df["distance_quartile"] = assign_rank_quantile(df["tanimoto_distance"], 4, q_labels)
    c_data = [df.loc[df["distance_quartile"] == lab, "error_normalized"].to_numpy() for lab in q_labels]
    viol = ax_c.violinplot(c_data, positions=np.arange(1, 5), widths=0.72, showextrema=False)
    for body in viol["bodies"]:
        body.set_facecolor(TEAL)
        body.set_edgecolor("#3b8d81")
        body.set_alpha(0.55)
        body.set_linewidth(0.9)
    ax_c.boxplot(
        c_data,
        positions=np.arange(1, 5),
        widths=0.25,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#b33c3c", "linewidth": 1.2},
        boxprops={"facecolor": "white", "edgecolor": "#666666", "linewidth": 0.9},
        whiskerprops={"color": "#666666", "linewidth": 0.8},
        capprops={"color": "#666666", "linewidth": 0.8},
    )
    c_rows = []
    dist_ranges = []
    for i, lab in enumerate(q_labels, start=1):
        sub = df[df["distance_quartile"] == lab]
        sample = sub.sample(min(1600, len(sub)), random_state=20260828 + i)
        jitter = np.random.default_rng(99 + i).normal(0, 0.055, size=len(sample))
        ax_c.scatter(np.full(len(sample), i) + jitter, sample["error_normalized"],
                     s=2, color="#555555", alpha=0.13, linewidths=0, zorder=1)
        ax_c.text(i, 1.025, f"n = {len(sub):,}", ha="center", va="bottom", fontsize=8, style="italic")
        dist_ranges.append(f"Q{i}: [{sub['tanimoto_distance'].min():.3f}-{sub['tanimoto_distance'].max():.3f}]")
        c_rows.append(
            {
                "distance_quartile": f"Q{i}",
                "n": len(sub),
                "distance_min": sub["tanimoto_distance"].min(),
                "distance_max": sub["tanimoto_distance"].max(),
                "mean_error": sub["error_normalized"].mean(),
                "median_error": sub["error_normalized"].median(),
                "mean_reliability": sub["reliability_score"].mean(),
            }
        )
    ax_c.set_ylim(-0.03, 1.08)
    ax_c.set_xticks(np.arange(1, 5))
    ax_c.set_xticklabels(q_labels)
    ax_c.set_xlabel("Structural distance quartile")
    ax_c.set_ylabel("Normalized absolute error")
    ax_c.set_title("Effect of structural shift", fontweight="bold", pad=12)
    ax_c.text(0.5, -0.28, "MACCS-Tanimoto structural distance to observed training domain\n" + "   ".join(dist_ranges),
              transform=ax_c.transAxes, ha="center", va="top", fontsize=8.5, color=BLUE)
    style_axes(ax_c)

    # D. Confidence remains informative under shift.
    d_rows = []
    for qi, qlab in enumerate(q_labels, start=1):
        sub = df[df["distance_quartile"] == qlab].copy()
        ranks = sub["reliability_score"].rank(method="first") / len(sub)
        sub["confidence_group"] = pd.cut(
            ranks,
            bins=[0, 0.2, 0.8, 1.0],
            labels=["Low", "Medium", "High"],
            include_lowest=True,
        )
        for group in ["High", "Medium", "Low"]:
            vals = sub.loc[sub["confidence_group"] == group, "error_normalized"].to_numpy()
            d_rows.append(
                {
                    "distance_quartile": f"Q{qi}",
                    "confidence_group": group,
                    "n": len(vals),
                    "mean_error": float(np.mean(vals)),
                    "median_error": float(np.median(vals)),
                    "sem": sem(vals),
                }
            )
    d_df = pd.DataFrame(d_rows)
    colors = {"High": "#356db4", "Medium": ORANGE, "Low": RED}
    offsets = {"High": -0.24, "Medium": 0.0, "Low": 0.24}
    for group in ["High", "Medium", "Low"]:
        sub = d_df[d_df["confidence_group"] == group]
        xpos = np.arange(1, 5) + offsets[group]
        ax_d.bar(
            xpos,
            sub["mean_error"],
            width=0.22,
            color=colors[group],
            edgecolor="#333333",
            linewidth=0.35,
            alpha=0.88,
            label={
                "High": "High reliability (top 20%)",
                "Medium": "Medium reliability (middle 60%)",
                "Low": "Low reliability (bottom 20%)",
            }[group],
            zorder=2,
        )
        ax_d.errorbar(xpos, sub["mean_error"], yerr=sub["sem"], fmt="none", ecolor="#222222",
                      elinewidth=0.8, capsize=2.5, zorder=3)
        for xi, yi in zip(xpos, sub["mean_error"]):
            ax_d.text(xi, yi + 0.025, f"{yi:.3f}", ha="center", va="bottom", fontsize=8)
    ymax_d = max((d_df["mean_error"] + d_df["sem"]).max() * 1.32, 0.35)
    ax_d.set_ylim(0, ymax_d)
    for qi in range(1, 5):
        n_q = int(d_df[(d_df["distance_quartile"] == f"Q{qi}") & (d_df["confidence_group"] == "High")]["n"].iloc[0])
        ax_d.text(qi, ymax_d * 0.93, f"n = {n_q:,}/group", ha="center", va="top", fontsize=8, style="italic")
    ax_d.set_xticks(np.arange(1, 5))
    ax_d.set_xticklabels(q_labels)
    ax_d.set_xlabel("Structural distance quartile")
    ax_d.set_ylabel("Mean normalized absolute error")
    ax_d.set_title("Confidence remains informative under shift", fontweight="bold", pad=12)
    ax_d.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=3)
    ax_d.text(0.5, -0.26, "Reliability groups are defined within each structural-distance quartile",
              transform=ax_d.transAxes, ha="center", va="top", fontsize=8.5, color=BLUE)
    style_axes(ax_d)

    for letter, ax in zip(["A", "B", "C", "D"], [ax_a, ax_b, ax_c, ax_d]):
        ax.text(-0.14, 1.08, letter, transform=ax.transAxes, fontsize=18, fontweight="bold")

    fig.tight_layout(rect=[0.025, 0.035, 0.995, 0.98], h_pad=5.0, w_pad=3.0)
    out_png = args.output_dir / "medium_property_pooled_score_error_validation_four_panel.png"
    out_pdf = args.output_dir / "medium_property_pooled_score_error_validation_four_panel.pdf"
    fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    prop_summary.to_csv(args.output_dir / "included_medium_properties.csv", index=False)
    decile_df.to_csv(args.output_dir / "panel_a_score_decile_error.csv", index=False)
    b_df.to_csv(args.output_dir / "panel_b_selective_retention.csv", index=False)
    pd.DataFrame(c_rows).to_csv(args.output_dir / "panel_c_structural_shift.csv", index=False)
    d_df.to_csv(args.output_dir / "panel_d_shift_confidence_groups.csv", index=False)

    print(
        {
            "n_properties": int(n_properties),
            "n_oof_cells": int(n_total),
            "png": str(out_png),
            "pdf": str(out_pdf),
            "spearman_score_error": float(
                df[["reliability_score", "error_normalized"]].corr(method="spearman").iloc[0, 1]
            ),
            "spearman_distance_error": float(
                df[["tanimoto_distance", "error_normalized"]].corr(method="spearman").iloc[0, 1]
            ),
            "median_property_spearman": float(prop_summary["spearman_score_error"].median()),
        }
    )


if __name__ == "__main__":
    main()
