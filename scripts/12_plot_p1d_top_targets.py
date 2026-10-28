"""
12_plot_p1d_top_targets.py

P1-D: Top-N 待测样本清单（active learning target list）

回答核心问题：接下来该做哪些具体的实验？

对每个缺失（polymer, property）单元格计算"实验价值"：
    value = ensemble_std_normalized × (1 - reliability_score)
       = 越大 = 越不可靠 & 越不一致 = 越值得动手测

输出:
- 一张CSV：Top-N 单元格清单（含smiles、性质、imputed value、置信区间、reliability）
- 一张四联图：
  A. 价值分布直方图（分类别）
  B. Top-500 待测样本的性质×类别分布饼图
  C. reliability vs ensemble_std 散点（着色=value）
  D. Top-100样本表格样例可视化
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
from workflow_config import (
    MISSING_PREDICTIONS_DIR,
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

TOP_N = 500


def collect_all_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """扫描所有缺失预测，构建完整候选表"""
    all_rows = []
    files = sorted(MISSING_PREDICTIONS_DIR.glob("*_missing_reliability.csv"))
    for f in files:
        prop = f.stem.replace("_missing_reliability", "")
        d = pd.read_csv(f)
        d["property"] = prop
        d["category"] = infer_category(prop)
        d["smiles"] = df.loc[d["polymer_id"].to_numpy(dtype=int), "smiles"].values
        d["experimental_value"] = d["ensemble_std_normalized"] * (1.0 - d["reliability_score"])
        all_rows.append(d)
    all_df = pd.concat(all_rows, ignore_index=True)

    # 保留有用列，排序
    cols = [
        "polymer_id", "smiles", "property", "category",
        "y_imputed_mean", "y_imputed_std",
        "property_nrmse", "ensemble_std_normalized",
        "tanimoto_distance", "reliability_score",
        "experimental_value",
    ]
    return all_df[cols]


def plot_panel_a_value_hist(ax, all_df: pd.DataFrame):
    for cat in sorted(all_df["category"].unique()):
        sub = all_df[all_df["category"] == cat]
        vals = sub["experimental_value"].to_numpy()
        vals = vals[np.isfinite(vals)]
        vals_capped = np.minimum(vals, np.quantile(vals, 0.99))
        ax.hist(vals_capped, bins=50, alpha=0.55, color=COLOR_PALETTE.get(cat, "gray"),
                label=cat, histtype="stepfilled", edgecolor="black", linewidth=0.3)

    ax.set_xlabel("Experimental value = ens_std_norm × (1 − reliability)", fontsize=11)
    ax.set_ylabel("# candidate cells (log)", fontsize=11)
    ax.set_yscale("log")
    ax.set_title("A. Experimental-value distribution by category",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=2)
    ax.grid(alpha=0.3)


def plot_panel_b_topN_composition(ax, top_n: pd.DataFrame):
    counts = top_n["category"].value_counts().sort_index()
    colors = [COLOR_PALETTE.get(c, "gray") for c in counts.index]
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=counts.index, colors=colors,
        autopct=lambda p: f"{p:.1f}%\n(n={int(p*counts.sum()/100)})",
        startangle=90, textprops={"fontsize": 8},
    )
    ax.set_title(f"B. Top-{TOP_N} experimental targets: category composition",
                 fontsize=12, fontweight="bold")


def plot_panel_c_scatter(ax, all_df: pd.DataFrame):
    sample = all_df.sample(n=min(20000, len(all_df)), random_state=42)
    x = sample["reliability_score"].to_numpy()
    y = sample["ensemble_std_normalized"].to_numpy()
    v = sample["experimental_value"].to_numpy()

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
    x, y, v = x[valid], y[valid], v[valid]
    v_capped = np.minimum(v, np.quantile(v, 0.98))

    sc = ax.scatter(x, y, c=v_capped, cmap="RdYlGn_r", s=4, alpha=0.5,
                    edgecolors="none")
    plt.colorbar(sc, ax=ax, label="Experimental value")

    ax.set_xlabel("Reliability score", fontsize=11)
    ax.set_ylabel("Ensemble std (normalized)", fontsize=11)
    ax.set_yscale("log")
    ax.set_title("C. Reliability × ensemble std, colored by experimental value\n(top-right corner = highest priority)",
                 fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3)


def plot_panel_d_top20_table(ax, top_n: pd.DataFrame):
    ax.axis("off")
    top20 = top_n.head(20).copy()

    def smiles_short(s):
        s = str(s)
        return s if len(s) < 45 else s[:42] + "..."

    display = pd.DataFrame({
        "#": np.arange(1, len(top20) + 1),
        "smiles": top20["smiles"].apply(smiles_short).values,
        "property": top20["property"].values,
        "ŷ (imputed)": [f"{v:.3f}" for v in top20["y_imputed_mean"]],
        "ens.std": [f"{v:.3f}" for v in top20["y_imputed_std"]],
        "rel.": [f"{v:.3f}" for v in top20["reliability_score"]],
        "value": [f"{v:.3f}" for v in top20["experimental_value"]],
    })

    tbl = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        loc="center", cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(6.5)
    tbl.scale(1, 1.2)

    for (i, j), cell in tbl.get_celld().items():
        if i == 0:
            cell.set_facecolor("#333333")
            cell.set_text_props(color="white", weight="bold")

    ax.set_title(f"D. Top-20 targets to measure next  (full Top-{TOP_N} in CSV)",
                 fontsize=12, fontweight="bold", pad=12)


def main():
    print("=" * 80)
    print("P1-D: Top-N 待测样本清单")
    print("=" * 80)

    ensure_dir(P1_FIGURES_DIR)

    df = pd.read_csv(SOURCE_MERGED_DATASET)
    print(f"数据形状: {df.shape}")

    print("\n扫描所有缺失预测...")
    all_df = collect_all_candidates(df)
    print(f"候选(polymer, property) 单元格总数: {len(all_df):,}")

    all_df_sorted = all_df.sort_values("experimental_value", ascending=False)
    top_n = all_df_sorted.head(TOP_N).copy()

    top_csv = P1_FIGURES_DIR / f"p1d_top{TOP_N}_experimental_targets.csv"
    top_n.to_csv(top_csv, index=False)
    print(f"Top-{TOP_N} 清单已保存: {top_csv}")

    full_csv = P1_FIGURES_DIR / "p1d_all_candidates_ranked.csv.gz"
    all_df_sorted.to_csv(full_csv, index=False, compression="gzip")
    print(f"完整排序清单已保存: {full_csv}")

    fig, axes = plt.subplots(2, 2, figsize=(17, 12))
    plot_panel_a_value_hist(axes[0, 0], all_df)
    plot_panel_b_topN_composition(axes[0, 1], top_n)
    plot_panel_c_scatter(axes[1, 0], all_df)
    plot_panel_d_top20_table(axes[1, 1], top_n)

    fig.suptitle(
        "Figure P1-D: Reliability-guided experimental target list\n"
        "(actionable output: which polymer × property to measure next, prioritized by information gain)",
        fontsize=13, fontweight="bold", y=0.998,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_pdf = P1_FIGURES_DIR / "fig_p1d_top_targets.pdf"
    out_png = P1_FIGURES_DIR / "fig_p1d_top_targets.png"
    plt.savefig(out_pdf, bbox_inches="tight", dpi=300)
    plt.savefig(out_png, bbox_inches="tight", dpi=200)
    plt.close()

    print(f"\n✓ 已保存: {out_png}")

    print(f"\n=== Top-{TOP_N} 组成 ===")
    print("按category:")
    print(top_n["category"].value_counts().to_string())
    print("\n按property (top 10):")
    print(top_n["property"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
