"""
11_plot_p1c_cluster_heatmap.py

P1-C: 化学聚类 × 性质 可靠性热力图

回答核心问题：哪些"化学家族×性质"组合系统性地不可靠？
方法：
1. 在MACCS指纹上做KMeans（K=16）
2. 对每个（cluster, property）组合，计算：
   - 观测样本的平均reliability
   - 缺失样本的平均reliability
3. 用热力图展示"哪些格子是暗区"

Panels:
- A: cluster × property 观测reliability 热力图
- B: cluster × property 缺失(imputation) reliability 热力图
- C: 每个cluster的样本数 & 平均缺失率
- D: 每个property在各cluster上reliability的变化系数（结构敏感性）
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import KMeans

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "config"))

from workflow_common import compute_maccs_fingerprints, ensure_dir, infer_category
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

N_CLUSTERS = 16


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


def build_cluster_x_property_matrix(
    labels: np.ndarray,
    source_dir: Path,
    file_suffix: str,
    n_clusters: int,
    property_order: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    返回:
        matrix_reliability: (n_clusters, n_properties) 中位reliability
        matrix_count: (n_clusters, n_properties) 样本数
    """
    n_props = len(property_order)
    reliability = np.full((n_clusters, n_props), np.nan)
    counts = np.zeros((n_clusters, n_props), dtype=int)

    for j, prop in enumerate(property_order):
        f = source_dir / f"{prop}{file_suffix}"
        if not f.exists():
            continue
        d = pd.read_csv(f, usecols=["polymer_id", "reliability_score"])
        d_labels = labels[d["polymer_id"].to_numpy(dtype=int)]
        for c in range(n_clusters):
            mask = d_labels == c
            if mask.sum() > 0:
                reliability[c, j] = np.median(d.loc[mask, "reliability_score"])
                counts[c, j] = mask.sum()
    return reliability, counts


def get_property_order() -> list[str]:
    props = []
    for f in sorted(OOF_RELIABILITY_DIR.glob("*_oof_reliability.csv")):
        props.append(f.stem.replace("_oof_reliability", ""))
    props.sort(key=lambda p: (infer_category(p), p))
    return props


def plot_heatmap(ax, mat: np.ndarray, xticklabels: list[str], title: str,
                 cbar_label: str, vmin: float = 0.5, vmax: float = 1.0):
    cmap = LinearSegmentedColormap.from_list(
        "rel", ["#D62728", "#FFEB84", "#2CA02C"], N=256,
    )
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    ax.set_xticks(np.arange(len(xticklabels)))
    ax.set_xticklabels(xticklabels, rotation=90, fontsize=6)
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels([f"C{c}" for c in range(mat.shape[0])], fontsize=8)
    ax.set_xlabel("Property (grouped by category)", fontsize=11)
    ax.set_ylabel("Chemical cluster (KMeans on MACCS)", fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, label=cbar_label)


def plot_panel_c_cluster_info(ax, labels: np.ndarray, missing_matrix: np.ndarray):
    cluster_sizes = np.bincount(labels, minlength=N_CLUSTERS)
    per_cluster_mean_miss = np.nanmean(missing_matrix, axis=1)

    x = np.arange(N_CLUSTERS)
    bars = ax.bar(x, cluster_sizes, color="#5C635D", alpha=0.8,
                  edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Chemical cluster", fontsize=11)
    ax.set_ylabel("# polymers in cluster", fontsize=11, color="#5C635D")
    ax.tick_params(axis="y", labelcolor="#5C635D")
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{c}" for c in range(N_CLUSTERS)], fontsize=8)

    ax2 = ax.twinx()
    ax2.plot(x, per_cluster_mean_miss, "o-", color="#D62728", linewidth=2, markersize=7,
             label="Mean imputation reliability")
    ax2.set_ylabel("Mean imputation reliability", fontsize=11, color="#D62728")
    ax2.tick_params(axis="y", labelcolor="#D62728")
    ax2.set_ylim(0.5, 1.0)
    ax2.axhline(0.7, color="#D62728", linestyle="--", alpha=0.4)

    ax.set_title("C. Cluster size & average imputation reliability",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.25, axis="y")


def plot_panel_d_property_sensitivity(ax, matrix_obs: np.ndarray, property_order: list[str]):
    stds = np.nanstd(matrix_obs, axis=0)
    means = np.nanmean(matrix_obs, axis=0)
    cv = stds / (means + 1e-6)

    idx = np.argsort(cv)[::-1][:20]
    cats = [infer_category(property_order[i]) for i in idx]
    color_map = {
        "thermal": "#D62728", "mechanical": "#8C564B", "electrical": "#FF7F0E",
        "optical": "#9467BD", "solution": "#1F77B4", "stability": "#2CA02C",
        "surface": "#17BECF", "transport": "#E377C2",
    }
    colors = [color_map.get(c, "gray") for c in cats]

    y = np.arange(len(idx))
    ax.barh(y, cv[idx][::-1], color=colors[::-1], alpha=0.85,
            edgecolor="black", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels([property_order[i] for i in idx[::-1]], fontsize=7)
    ax.set_xlabel("Coefficient of variation (std/mean of cluster-level reliability)", fontsize=10)
    ax.set_title("D. Top-20 properties MOST structure-sensitive\n(reliability varies most across chemical clusters)",
                 fontsize=10, fontweight="bold")
    ax.grid(alpha=0.3, axis="x")


def main():
    print("=" * 80)
    print("P1-C: 化学聚类 × 性质 可靠性热力图")
    print("=" * 80)

    ensure_dir(P1_FIGURES_DIR)

    df = pd.read_csv(SOURCE_MERGED_DATASET)
    n = len(df)
    print(f"数据形状: {df.shape}")

    print("\n加载MACCS指纹...")
    fp = load_maccs(df).astype(np.float32)

    print(f"\nKMeans聚类（K={N_CLUSTERS}）...")
    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    labels = km.fit_predict(fp)
    print(f"cluster sizes: {np.bincount(labels)}")

    print("\n性质排序...")
    props = get_property_order()
    print(f"共 {len(props)} 个性质")

    print("\n构建 cluster × property 矩阵（观测）...")
    mat_obs, cnt_obs = build_cluster_x_property_matrix(
        labels, OOF_RELIABILITY_DIR, "_oof_reliability.csv", N_CLUSTERS, props
    )
    print("构建 cluster × property 矩阵（缺失）...")
    mat_miss, cnt_miss = build_cluster_x_property_matrix(
        labels, MISSING_PREDICTIONS_DIR, "_missing_reliability.csv", N_CLUSTERS, props
    )

    out_df = pd.DataFrame(mat_miss, columns=props)
    out_df.index = [f"C{c}" for c in range(N_CLUSTERS)]
    out_df.to_csv(P1_FIGURES_DIR / "p1c_cluster_x_property_missing.csv")
    print(f"CSV 已保存")

    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    plot_heatmap(
        axes[0, 0], mat_obs, props,
        title="A. OBSERVED reliability (median) by chemical cluster × property",
        cbar_label="Median reliability",
    )
    plot_heatmap(
        axes[0, 1], mat_miss, props,
        title="B. IMPUTATION reliability (median) by chemical cluster × property\n(red cells = knowledge dark spots)",
        cbar_label="Median reliability",
    )
    plot_panel_c_cluster_info(axes[1, 0], labels, mat_miss)
    plot_panel_d_property_sensitivity(axes[1, 1], mat_obs, props)

    fig.suptitle(
        "Figure P1-C: Chemical-cluster × property reliability heatmap\n"
        "(pinpoints which chemical family × property combinations are systematically low-support)",
        fontsize=13, fontweight="bold", y=0.998,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_pdf = P1_FIGURES_DIR / "fig_p1c_cluster_heatmap.pdf"
    out_png = P1_FIGURES_DIR / "fig_p1c_cluster_heatmap.png"
    plt.savefig(out_pdf, bbox_inches="tight", dpi=300)
    plt.savefig(out_png, bbox_inches="tight", dpi=200)
    plt.close()

    print(f"\n✓ 已保存: {out_png}")

    print("\n=== 最不可靠的 (cluster, property) 组合 Top-10 ===")
    flat = []
    for c in range(N_CLUSTERS):
        for j, p in enumerate(props):
            r = mat_miss[c, j]
            if np.isfinite(r):
                flat.append((f"C{c}", p, r, cnt_miss[c, j]))
    flat.sort(key=lambda x: x[2])
    print(f"{'cluster':<8} {'property':<32} {'median rel':<10} {'n_miss':<8}")
    for c, p, r, n in flat[:10]:
        print(f"{c:<8} {p:<32} {r:<10.3f} {n:<8}")


if __name__ == "__main__":
    main()
