"""
09_plot_p1a_property_gap.py

P1-A: 性质级知识空缺排名

回答核心问题：哪些性质最需要新的实验数据？
指标：
- observed_count: 已观测样本数
- missing_count: 缺失样本数
- mean_reliability_missing: 平均imputation reliability
- knowledge_gap_score = missing_count × (1 - mean_reliability_missing)
  → 越大 = 缺失量大 & imputation越不可靠 = 越亟需补数据

Panels:
- A: Top-20 gap-score排名（bar）
- B: observed_count vs mean_reliability_missing 气泡图（气泡大小=缺失量，颜色=category）
- C: 8个类别的平均gap分布
- D: 全性质缺失率 vs OOF reliability
"""

from __future__ import annotations

import io
import json
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
from workflow_config import (
    MISSING_PREDICTIONS_DIR,
    OOF_PREDICTIONS_DIR,
    OOF_RELIABILITY_DIR,
    P0_FIGURES_DIR,
    SOURCE_MERGED_DATASET,
)

P1_FIGURES_DIR = P0_FIGURES_DIR.parent / "06_p1_figures"

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.linewidth"] = 1.2

COLOR_PALETTE = {
    "thermal": "#D62728",
    "mechanical": "#8C564B",
    "electrical": "#FF7F0E",
    "optical": "#9467BD",
    "solution": "#1F77B4",
    "stability": "#2CA02C",
    "surface": "#17BECF",
    "transport": "#E377C2",
}


def build_property_summary() -> pd.DataFrame:
    df = pd.read_csv(SOURCE_MERGED_DATASET)

    rows = []
    for meta_file in sorted(OOF_PREDICTIONS_DIR.glob("*_metadata.json")):
        prop = meta_file.stem.replace("_metadata", "")
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        oof_file = OOF_RELIABILITY_DIR / f"{prop}_oof_reliability.csv"
        miss_file = MISSING_PREDICTIONS_DIR / f"{prop}_missing_reliability.csv"

        if not oof_file.exists() or not miss_file.exists():
            continue

        oof = pd.read_csv(oof_file)
        miss = pd.read_csv(miss_file)

        n_obs = int(oof.shape[0])
        n_miss = int(miss.shape[0])
        total = n_obs + n_miss
        missing_rate = n_miss / total if total > 0 else np.nan

        mean_rel_obs = float(oof["reliability_score"].mean())
        mean_rel_miss = float(miss["reliability_score"].mean())

        gap_score = n_miss * (1.0 - mean_rel_miss)

        rows.append({
            "property": prop,
            "category": infer_category(prop),
            "observed_count": n_obs,
            "missing_count": n_miss,
            "missing_rate": missing_rate,
            "property_nrmse": meta["oof_nrmse"],
            "mean_reliability_observed": mean_rel_obs,
            "mean_reliability_missing": mean_rel_miss,
            "knowledge_gap_score": gap_score,
        })

    return pd.DataFrame(rows)


def plot_panel_a_top20(ax, summary: pd.DataFrame):
    top20 = summary.nlargest(20, "knowledge_gap_score").iloc[::-1]
    colors = [COLOR_PALETTE.get(c, "gray") for c in top20["category"]]
    y = np.arange(len(top20))
    ax.barh(y, top20["knowledge_gap_score"], color=colors, edgecolor="black", linewidth=0.5, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(top20["property"], fontsize=8)
    ax.set_xlabel("Knowledge gap score  =  n_missing × (1 − mean reliability)", fontsize=11)
    ax.set_title("A. Top-20 properties most in need of new experimental data",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, axis="x")

    for i, (v, r) in enumerate(zip(top20["knowledge_gap_score"], top20["mean_reliability_missing"])):
        ax.text(v, i, f"  r̄={r:.2f}", va="center", fontsize=7)


def plot_panel_b_bubble(ax, summary: pd.DataFrame):
    for cat in sorted(summary["category"].unique()):
        sub = summary[summary["category"] == cat]
        color = COLOR_PALETTE.get(cat, "gray")
        sizes = np.sqrt(sub["missing_count"]) * 0.8
        ax.scatter(
            sub["observed_count"], sub["mean_reliability_missing"],
            s=sizes, color=color, alpha=0.65, edgecolor="black", linewidth=0.5,
            label=f"{cat} (n={len(sub)})"
        )

    top5 = summary.nlargest(5, "knowledge_gap_score")
    for _, r in top5.iterrows():
        ax.annotate(
            r["property"],
            (r["observed_count"], r["mean_reliability_missing"]),
            fontsize=7, xytext=(3, 3), textcoords="offset points",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Observed sample count (log)", fontsize=11)
    ax.set_ylabel("Mean reliability on missing cells", fontsize=11)
    ax.set_title("B. Property landscape\n(bubble size ∝ √n_missing)",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", frameon=False, fontsize=7, ncol=2)
    ax.grid(alpha=0.3)


def plot_panel_c_by_category(ax, summary: pd.DataFrame):
    cats = sorted(summary["category"].unique())
    data_by_cat = [summary[summary["category"] == c]["knowledge_gap_score"].to_numpy() for c in cats]
    counts = [len(d) for d in data_by_cat]

    positions = np.arange(len(cats))
    bp = ax.boxplot(data_by_cat, positions=positions, widths=0.6, patch_artist=True,
                    showfliers=True, medianprops=dict(color="black", linewidth=1.5))
    for patch, cat in zip(bp["boxes"], cats):
        patch.set_facecolor(COLOR_PALETTE.get(cat, "gray"))
        patch.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{c}\n(n={n})" for c, n in zip(cats, counts)], fontsize=8)
    ax.set_ylabel("Knowledge gap score", fontsize=11)
    ax.set_title("C. Gap distribution across property categories",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, axis="y")


def plot_panel_d_missing_vs_reliability(ax, summary: pd.DataFrame):
    for cat in sorted(summary["category"].unique()):
        sub = summary[summary["category"] == cat]
        ax.scatter(
            sub["missing_rate"] * 100, sub["mean_reliability_observed"],
            s=40, color=COLOR_PALETTE.get(cat, "gray"), alpha=0.75,
            edgecolor="black", linewidth=0.5, label=cat,
        )

    for _, r in summary.nlargest(3, "knowledge_gap_score").iterrows():
        ax.annotate(
            r["property"],
            (r["missing_rate"] * 100, r["mean_reliability_observed"]),
            fontsize=7, xytext=(3, -8), textcoords="offset points",
        )
    for _, r in summary.nsmallest(3, "mean_reliability_observed").iterrows():
        ax.annotate(
            r["property"],
            (r["missing_rate"] * 100, r["mean_reliability_observed"]),
            fontsize=7, xytext=(3, 3), textcoords="offset points",
        )

    ax.set_xlabel("Missing rate (%)", fontsize=11)
    ax.set_ylabel("Mean OOF reliability (observed cells)", fontsize=11)
    ax.set_title("D. Missing rate vs observed-cell reliability",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="lower left", frameon=False, fontsize=7, ncol=2)
    ax.grid(alpha=0.3)


def main():
    print("=" * 80)
    print("P1-A: 性质级知识空缺排名")
    print("=" * 80)

    ensure_dir(P1_FIGURES_DIR)

    print("\n汇总各性质...")
    summary = build_property_summary()
    print(f"性质数: {len(summary)}")

    csv_out = P1_FIGURES_DIR / "p1a_property_gap_ranking.csv"
    summary.sort_values("knowledge_gap_score", ascending=False).to_csv(csv_out, index=False)
    print(f"CSV 已保存: {csv_out}")

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    plot_panel_a_top20(axes[0, 0], summary)
    plot_panel_b_bubble(axes[0, 1], summary)
    plot_panel_c_by_category(axes[1, 0], summary)
    plot_panel_d_missing_vs_reliability(axes[1, 1], summary)

    fig.suptitle(
        "Figure P1-A: Property-level knowledge gap ranking\n"
        "(what to measure next, guided by reliability × missing count)",
        fontsize=13, fontweight="bold", y=0.998,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_pdf = P1_FIGURES_DIR / "fig_p1a_property_gap.pdf"
    out_png = P1_FIGURES_DIR / "fig_p1a_property_gap.png"
    plt.savefig(out_pdf, bbox_inches="tight", dpi=300)
    plt.savefig(out_png, bbox_inches="tight", dpi=200)
    plt.close()

    print(f"\n✓ 已保存: {out_png}")

    print("\n=== Top-10 gap score ===")
    top = summary.nlargest(10, "knowledge_gap_score")[
        ["property", "category", "missing_count", "mean_reliability_missing", "knowledge_gap_score"]
    ]
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()
