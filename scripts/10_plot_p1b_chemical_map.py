"""
10_plot_p1b_chemical_map.py

P1-B: 化学空间可靠性地图

回答核心问题：可靠性在化学空间上如何分布？"暗区"在哪？

将所有polymer的MACCS 167维投影到2D（PCA），每个polymer着色为
其在所有已训练性质上的平均reliability。

Panels:
- A: 2D投影，色标=每个polymer的mean reliability across all obs cells
- B: 2D投影，色标=每个polymer的mean reliability across all missing cells
- C: 2D投影，色标=样本已观测性质数（数据密度）
- D: 观测/缺失 reliability的直方图对比
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "config"))

from workflow_common import compute_maccs_fingerprints, ensure_dir
from workflow_config import (
    MISSING_PREDICTIONS_DIR,
    OOF_RELIABILITY_DIR,
    P0_FIGURES_DIR,
    SMILES_COL,
    SOURCE_MERGED_DATASET,
)

P1_FIGURES_DIR = P0_FIGURES_DIR.parent / "06_p1_figures"

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.linewidth"] = 1.2


def load_maccs(df: pd.DataFrame) -> np.ndarray:
    parquet_path = SOURCE_MERGED_DATASET.parent / "maccs_cache.parquet"
    pickle_path = SOURCE_MERGED_DATASET.parent / "maccs_cache.pkl"
    if parquet_path.exists():
        try:
            fp_df = pd.read_parquet(parquet_path)
            fp_cols = [c for c in fp_df.columns if c.startswith("maccs_")]
            if len(fp_df) == len(df):
                return fp_df[fp_cols].to_numpy(dtype=np.int8)
        except Exception:
            pass
    if pickle_path.exists():
        fp_df = pd.read_pickle(pickle_path)
        fp_cols = [c for c in fp_df.columns if c.startswith("maccs_")]
        return fp_df[fp_cols].to_numpy(dtype=np.int8)
    fp_df = compute_maccs_fingerprints(df, SMILES_COL)
    fp_cols = [c for c in fp_df.columns if c.startswith("maccs_")]
    return fp_df[fp_cols].to_numpy(dtype=np.int8)


def build_per_polymer_stats(n_polymers: int) -> pd.DataFrame:
    """聚合每个polymer的平均reliability（观测&缺失）以及观测性质数"""
    sum_obs = np.zeros(n_polymers, dtype=np.float64)
    cnt_obs = np.zeros(n_polymers, dtype=np.int32)
    sum_miss = np.zeros(n_polymers, dtype=np.float64)
    cnt_miss = np.zeros(n_polymers, dtype=np.int32)

    for f in sorted(OOF_RELIABILITY_DIR.glob("*_oof_reliability.csv")):
        d = pd.read_csv(f, usecols=["polymer_id", "reliability_score"])
        ids = d["polymer_id"].to_numpy(dtype=int)
        rel = d["reliability_score"].to_numpy(dtype=np.float64)
        np.add.at(sum_obs, ids, rel)
        np.add.at(cnt_obs, ids, 1)

    for f in sorted(MISSING_PREDICTIONS_DIR.glob("*_missing_reliability.csv")):
        d = pd.read_csv(f, usecols=["polymer_id", "reliability_score"])
        ids = d["polymer_id"].to_numpy(dtype=int)
        rel = d["reliability_score"].to_numpy(dtype=np.float64)
        np.add.at(sum_miss, ids, rel)
        np.add.at(cnt_miss, ids, 1)

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_obs = np.where(cnt_obs > 0, sum_obs / cnt_obs, np.nan)
        mean_miss = np.where(cnt_miss > 0, sum_miss / cnt_miss, np.nan)

    return pd.DataFrame({
        "polymer_id": np.arange(n_polymers),
        "n_observed_props": cnt_obs,
        "n_missing_props": cnt_miss,
        "mean_reliability_observed": mean_obs,
        "mean_reliability_missing": mean_miss,
    })


def plot_map(ax, coords: np.ndarray, values: np.ndarray, title: str,
             cbar_label: str, cmap: str = "RdYlGn", vmin=None, vmax=None):
    valid = np.isfinite(values)
    sc = ax.scatter(
        coords[valid, 0], coords[valid, 1],
        c=values[valid], cmap=cmap,
        s=3, alpha=0.55, vmin=vmin, vmax=vmax, edgecolors="none",
    )
    plt.colorbar(sc, ax=ax, label=cbar_label)
    ax.set_xlabel("PC1", fontsize=11)
    ax.set_ylabel("PC2", fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(alpha=0.25)


def plot_panel_d_hist(ax, stats: pd.DataFrame):
    obs = stats["mean_reliability_observed"].dropna().to_numpy()
    miss = stats["mean_reliability_missing"].dropna().to_numpy()

    ax.hist(obs, bins=40, alpha=0.6, color="#2CA02C", edgecolor="black",
            label=f"Observed cells (per-polymer mean, n={len(obs):,})", linewidth=0.4)
    ax.hist(miss, bins=40, alpha=0.6, color="#D62728", edgecolor="black",
            label=f"Missing cells (per-polymer mean, n={len(miss):,})", linewidth=0.4)
    ax.axvline(np.median(obs), color="#2CA02C", linestyle="--", linewidth=1.5,
               label=f"Obs median = {np.median(obs):.3f}")
    ax.axvline(np.median(miss), color="#D62728", linestyle="--", linewidth=1.5,
               label=f"Miss median = {np.median(miss):.3f}")

    ax.set_xlabel("Per-polymer mean reliability", fontsize=11)
    ax.set_ylabel("# polymers", fontsize=11)
    ax.set_title("D. Distribution: observed vs missing reliability",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.grid(alpha=0.3)


def main():
    print("=" * 80)
    print("P1-B: 化学空间可靠性地图")
    print("=" * 80)

    ensure_dir(P1_FIGURES_DIR)

    df = pd.read_csv(SOURCE_MERGED_DATASET)
    n = len(df)
    print(f"数据形状: {df.shape}")

    print("\n加载MACCS指纹...")
    fp = load_maccs(df)
    print(f"MACCS: {fp.shape}")

    print("\n聚合每个polymer的可靠性...")
    stats = build_per_polymer_stats(n)
    print(f"  polymers with observed reliability: {stats['n_observed_props'].gt(0).sum():,}")
    print(f"  polymers with missing reliability: {stats['n_missing_props'].gt(0).sum():,}")

    print("\nPCA 投影到 2D...")
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(fp.astype(np.float32))
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")

    stats["pc1"] = coords[:, 0]
    stats["pc2"] = coords[:, 1]
    stats.to_csv(P1_FIGURES_DIR / "p1b_polymer_stats.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    plot_map(
        axes[0, 0], coords, stats["mean_reliability_observed"].to_numpy(),
        title="A. Chemical space colored by mean reliability of\nOBSERVED measurements",
        cbar_label="Mean reliability (observed)",
        cmap="RdYlGn", vmin=0.5, vmax=1.0,
    )
    plot_map(
        axes[0, 1], coords, stats["mean_reliability_missing"].to_numpy(),
        title="B. Chemical space colored by mean reliability of\nIMPUTED (missing) predictions",
        cbar_label="Mean reliability (imputed)",
        cmap="RdYlGn", vmin=0.5, vmax=1.0,
    )
    plot_map(
        axes[1, 0], coords, stats["n_observed_props"].to_numpy(),
        title="C. Chemical space colored by DATA DENSITY\n(# observed properties per polymer)",
        cbar_label="# observed properties",
        cmap="viridis", vmin=0, vmax=None,
    )
    plot_panel_d_hist(axes[1, 1], stats)

    fig.suptitle(
        "Figure P1-B: Reliability map over MACCS-PCA chemical space\n"
        "(red / green regions expose exactly where new experiments will have highest information gain)",
        fontsize=13, fontweight="bold", y=0.998,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_pdf = P1_FIGURES_DIR / "fig_p1b_chemical_map.pdf"
    out_png = P1_FIGURES_DIR / "fig_p1b_chemical_map.png"
    plt.savefig(out_pdf, bbox_inches="tight", dpi=300)
    plt.savefig(out_png, bbox_inches="tight", dpi=200)
    plt.close()

    print(f"\n✓ 已保存: {out_png}")

    print("\n=== 统计 ===")
    print(f"Observed mean reliability (polymers avg): {stats['mean_reliability_observed'].mean():.4f}")
    print(f"Missing  mean reliability (polymers avg): {stats['mean_reliability_missing'].mean():.4f}")


if __name__ == "__main__":
    main()
