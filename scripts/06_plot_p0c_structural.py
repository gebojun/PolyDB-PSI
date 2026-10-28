"""
06_plot_p0c_structural.py

P0-C: 结构距离压力测试

回答核心问题：reliability是否只是Tanimoto距离的复述？

图片包含:
- Panel A: 4个结构距离四分位（Q1-Q4）内NAE的分布
- Panel B: Q1-Q4内的reliability分布
- Panel C: 每个Q内按reliability分High/Medium/Low的NAE对比
- Panel D: reliability vs 结构距离的散点图（颜色=NAE）
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "config"))

from workflow_common import ensure_dir, infer_category
from workflow_config import OOF_RELIABILITY_DIR, P0_FIGURES_DIR


plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.linewidth"] = 1.2


def load_all_reliability_data() -> pd.DataFrame:
    all_dfs = []
    for f in sorted(OOF_RELIABILITY_DIR.glob("*_oof_reliability.csv")):
        prop_name = f.stem.replace("_oof_reliability", "")
        df = pd.read_csv(f)
        df["property_name"] = prop_name
        df["category"] = infer_category(prop_name)
        all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)


def add_quartile_labels(data: pd.DataFrame) -> pd.DataFrame:
    """按tanimoto距离分四分位"""
    data = data.copy()
    valid = np.isfinite(data["tanimoto_distance"])
    q_labels = pd.qcut(
        data.loc[valid, "tanimoto_distance"],
        q=4,
        labels=["Q1 (closest)", "Q2", "Q3", "Q4 (farthest)"],
    )
    data.loc[valid, "distance_quartile"] = q_labels.astype(str)
    return data


def plot_panel_a_distance_quartile_error(ax, data: pd.DataFrame):
    """Panel A: 结构距离四分位 vs 折外误差"""
    quartiles = ["Q1 (closest)", "Q2", "Q3", "Q4 (farthest)"]
    positions = np.arange(len(quartiles))

    box_data = []
    medians = []
    for q in quartiles:
        subset = data[data["distance_quartile"] == q]
        errors = subset["oof_error_normalized"].to_numpy()
        errors = errors[np.isfinite(errors)]
        errors_capped = np.minimum(errors, np.quantile(errors, 0.95))
        box_data.append(errors_capped)
        medians.append(np.median(errors))

    bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True,
                    showfliers=False, medianprops=dict(color="black", linewidth=1.5))

    colors = ["#2CA02C", "#8DC63F", "#FFB84D", "#D62728"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for i, med in enumerate(medians):
        ax.text(i, med, f"  {med:.3f}", ha="left", va="center", fontsize=9, fontweight="bold")

    ax.set_xticks(positions)
    ax.set_xticklabels(["Q1\n(closest)", "Q2", "Q3", "Q4\n(farthest)"])
    ax.set_xlabel("Nearest-neighbor Tanimoto distance quartile", fontsize=11)
    ax.set_ylabel("Normalized absolute error (NAE)", fontsize=11)
    ax.set_title("A. Structural distance stress: error grows with distance",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, axis="y")


def plot_panel_b_distance_quartile_reliability(ax, data: pd.DataFrame):
    """Panel B: 结构距离四分位 vs reliability分布"""
    quartiles = ["Q1 (closest)", "Q2", "Q3", "Q4 (farthest)"]
    positions = np.arange(len(quartiles))

    box_data = []
    medians = []
    for q in quartiles:
        subset = data[data["distance_quartile"] == q]
        rel = subset["reliability_score"].to_numpy()
        rel = rel[np.isfinite(rel)]
        box_data.append(rel)
        medians.append(np.median(rel))

    bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True,
                    showfliers=False, medianprops=dict(color="black", linewidth=1.5))

    colors = ["#2CA02C", "#8DC63F", "#FFB84D", "#D62728"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for i, med in enumerate(medians):
        ax.text(i, med, f"  {med:.3f}", ha="left", va="center", fontsize=9, fontweight="bold")

    ax.set_xticks(positions)
    ax.set_xticklabels(["Q1\n(closest)", "Q2", "Q3", "Q4\n(farthest)"])
    ax.set_xlabel("Nearest-neighbor Tanimoto distance quartile", fontsize=11)
    ax.set_ylabel("Reliability score", fontsize=11)
    ax.set_title("B. Reliability decreases with structural distance",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, axis="y")


def plot_panel_c_within_quartile(ax, data: pd.DataFrame):
    """Panel C: 每个Q内按reliability分组的NAE对比（关键图）"""
    quartiles = ["Q1 (closest)", "Q2", "Q3", "Q4 (farthest)"]

    reliability_groups = {
        "Low": lambda x: x < 0.60,
        "Medium": lambda x: (x >= 0.60) & (x < 0.80),
        "High": lambda x: x >= 0.80,
    }

    group_colors = {
        "Low": "#D62728",
        "Medium": "#FFB84D",
        "High": "#2CA02C",
    }

    x = np.arange(len(quartiles))
    width = 0.25

    stats = {}
    for i, (grp_name, grp_func) in enumerate(reliability_groups.items()):
        medians = []
        counts = []
        for q in quartiles:
            subset = data[data["distance_quartile"] == q]
            group_data = subset[grp_func(subset["reliability_score"])]
            errors = group_data["oof_error_normalized"].to_numpy()
            errors = errors[np.isfinite(errors)]
            if len(errors) > 0:
                medians.append(np.median(errors))
                counts.append(len(errors))
            else:
                medians.append(np.nan)
                counts.append(0)

        stats[grp_name] = (medians, counts)

        offset = (i - 1) * width
        bars = ax.bar(x + offset, medians, width, color=group_colors[grp_name],
                      alpha=0.85, edgecolor="black", linewidth=0.8,
                      label=f"{grp_name} reliability")

        for j, (bar, cnt) in enumerate(zip(bars, counts)):
            if not np.isnan(bar.get_height()) and cnt > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"n={cnt}", ha="center", va="bottom", fontsize=7, rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels(["Q1\n(closest)", "Q2", "Q3", "Q4\n(farthest)"])
    ax.set_xlabel("Structural distance quartile", fontsize=11)
    ax.set_ylabel("Median NAE", fontsize=11)
    ax.set_title("C. Within each distance quartile, reliability still separates errors\n(⭐ proves reliability ≠ Tanimoto distance)",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=0.3, axis="y")


def plot_panel_d_scatter(ax, data: pd.DataFrame):
    """Panel D: reliability vs 距离 散点图（颜色=NAE）"""
    n_sample = min(10000, len(data))
    sample = data.sample(n=n_sample, random_state=42) if len(data) > n_sample else data

    scores = sample["reliability_score"].to_numpy()
    distances = sample["tanimoto_distance"].to_numpy()
    errors = sample["oof_error_normalized"].to_numpy()

    valid = np.isfinite(scores) & np.isfinite(distances) & np.isfinite(errors)
    scores, distances, errors = scores[valid], distances[valid], errors[valid]

    errors_capped = np.minimum(errors, 1.5)

    sc = ax.scatter(distances, scores, c=errors_capped, cmap="RdYlGn_r",
                    s=8, alpha=0.5, vmin=0, vmax=1.5)

    cbar = plt.colorbar(sc, ax=ax, label="NAE (capped at 1.5)")

    ax.set_xlabel("Nearest-neighbor Tanimoto distance", fontsize=11)
    ax.set_ylabel("Reliability score", fontsize=11)
    ax.set_title("D. Reliability vs distance colored by error", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)

    if len(scores) > 100:
        corr = np.corrcoef(scores, distances)[0, 1]
        ax.text(
            0.98, 0.98,
            f"Correlation: r = {corr:.3f}\n(if reliability were just\ndistance, r would be -1)",
            transform=ax.transAxes,
            horizontalalignment="right",
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9),
            fontsize=9,
        )


def main():
    print("=" * 80)
    print("P0-C: 结构距离压力测试")
    print("=" * 80)

    ensure_dir(P0_FIGURES_DIR)

    print("\n加载数据...")
    data = load_all_reliability_data()
    data = add_quartile_labels(data)
    print(f"总样本数: {len(data)}")
    print()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    plot_panel_a_distance_quartile_error(axes[0, 0], data)
    plot_panel_b_distance_quartile_reliability(axes[0, 1], data)
    plot_panel_c_within_quartile(axes[1, 0], data)
    plot_panel_d_scatter(axes[1, 1], data)

    fig.suptitle(
        "Figure P0-C: Structural distance stress test\n(Reliability is not merely a restatement of Tanimoto distance)",
        fontsize=13, fontweight="bold", y=0.998,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_pdf = P0_FIGURES_DIR / "fig_p0c_structural.pdf"
    output_png = P0_FIGURES_DIR / "fig_p0c_structural.png"

    plt.savefig(output_pdf, bbox_inches="tight", dpi=300)
    plt.savefig(output_png, bbox_inches="tight", dpi=200)
    plt.close()

    print(f"\n✓ 已保存:")
    print(f"  {output_pdf}")
    print(f"  {output_png}")

    print("\n=== 关键统计（Panel C：quartile内的reliability分层）===")
    for q in ["Q1 (closest)", "Q2", "Q3", "Q4 (farthest)"]:
        subset = data[data["distance_quartile"] == q]
        low = subset[subset["reliability_score"] < 0.60]
        med = subset[(subset["reliability_score"] >= 0.60) & (subset["reliability_score"] < 0.80)]
        high = subset[subset["reliability_score"] >= 0.80]

        med_low = np.median(low["oof_error_normalized"]) if len(low) > 0 else float("nan")
        med_med = np.median(med["oof_error_normalized"]) if len(med) > 0 else float("nan")
        med_high = np.median(high["oof_error_normalized"]) if len(high) > 0 else float("nan")

        print(f"\n{q}:")
        print(f"  Low reliability  (n={len(low):5d}): median NAE = {med_low:.4f}")
        print(f"  Med reliability  (n={len(med):5d}): median NAE = {med_med:.4f}")
        print(f"  High reliability (n={len(high):5d}): median NAE = {med_high:.4f}")

        if not np.isnan(med_high) and not np.isnan(med_low) and med_low > 0:
            reduction = (med_low - med_high) / med_low * 100
            print(f"  → 组内 Low→High reliability 误差降低: {reduction:.1f}%")


if __name__ == "__main__":
    main()
