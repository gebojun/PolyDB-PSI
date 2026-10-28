"""
07_plot_p0d_extrapolation.py

P0-D: 结构支持距离梯度诊断（Distance-gradient calibration）

回答核心问题：在已有实验观测分布内，结构支持逐渐减弱时，可靠性分数
是否仍能排序真实OOF误差？该图刻画有限结构偏移下的适用边界，不把全部
真实缺失格等同于外推，也不声称随机OOF已覆盖所有远域缺失空间。

图片包含:
- Panel A: 距离×可靠性 热力图（median NAE）
- Panel B: 分距离段的校准曲线（近端、中端、远端）
- Panel C: 观测样本中结构支持最弱子集的校准
- Panel D: 各距离段中 High vs Low reliability 的误差降低幅度
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

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

N_DISTANCE_BINS = 10
N_RELIABILITY_BINS = 10


def load_all_reliability_data() -> pd.DataFrame:
    all_dfs = []
    for f in sorted(OOF_RELIABILITY_DIR.glob("*_oof_reliability.csv")):
        prop_name = f.stem.replace("_oof_reliability", "")
        df = pd.read_csv(f)
        df["property_name"] = prop_name
        df["category"] = infer_category(prop_name)
        all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)


def compute_2d_calibration_matrix(
    distances: np.ndarray,
    scores: np.ndarray,
    errors: np.ndarray,
    n_dist_bins: int = N_DISTANCE_BINS,
    n_rel_bins: int = N_RELIABILITY_BINS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """计算距离×可靠性的中位NAE矩阵及bin中心"""
    valid = np.isfinite(distances) & np.isfinite(scores) & np.isfinite(errors)
    d = distances[valid]
    s = scores[valid]
    e = errors[valid]

    dist_edges = np.quantile(d, np.linspace(0, 1, n_dist_bins + 1))
    dist_edges[0] -= 1e-9
    dist_edges[-1] += 1e-9

    matrix = np.full((n_dist_bins, n_rel_bins), np.nan)
    counts = np.zeros((n_dist_bins, n_rel_bins), dtype=int)
    dist_centers = np.zeros(n_dist_bins)
    rel_centers_global = np.zeros(n_rel_bins)

    for i in range(n_dist_bins):
        dmask = (d > dist_edges[i]) & (d <= dist_edges[i + 1])
        if dmask.sum() == 0:
            continue
        dist_centers[i] = d[dmask].mean()

        s_bin = s[dmask]
        e_bin = e[dmask]

        rel_edges = np.quantile(s_bin, np.linspace(0, 1, n_rel_bins + 1))
        rel_edges[0] -= 1e-9
        rel_edges[-1] += 1e-9

        for j in range(n_rel_bins):
            rmask = (s_bin > rel_edges[j]) & (s_bin <= rel_edges[j + 1])
            if rmask.sum() > 0:
                matrix[i, j] = np.median(e_bin[rmask])
                counts[i, j] = rmask.sum()
                if i == 0:
                    rel_centers_global[j] = s_bin[rmask].mean()

    return matrix, dist_centers, rel_centers_global, counts


def plot_panel_a_heatmap(ax, data: pd.DataFrame):
    """Panel A: 距离×可靠性 中位NAE 热力图"""
    matrix, dist_c, rel_c, _ = compute_2d_calibration_matrix(
        data["tanimoto_distance"].to_numpy(),
        data["reliability_score"].to_numpy(),
        data["oof_error_normalized"].to_numpy(),
    )

    vmax = np.nanquantile(matrix, 0.95)

    cmap = LinearSegmentedColormap.from_list(
        "cal", ["#2CA02C", "#FFEB84", "#D62728"], N=256,
    )
    im = ax.imshow(
        matrix, aspect="auto", origin="lower", cmap=cmap,
        vmin=0, vmax=vmax, interpolation="nearest",
    )

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if np.isfinite(v):
                color = "white" if v > vmax * 0.55 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color=color)

    ax.set_xticks(np.arange(N_RELIABILITY_BINS))
    ax.set_xticklabels([f"R{i+1}" for i in range(N_RELIABILITY_BINS)], fontsize=8)
    ax.set_yticks(np.arange(N_DISTANCE_BINS))
    ax.set_yticklabels([f"D{i+1}" for i in range(N_DISTANCE_BINS)], fontsize=8)
    ax.set_xlabel("Reliability decile (low→high, per distance-band)", fontsize=11)
    ax.set_ylabel("Tanimoto distance decile (near→far)", fontsize=11)
    ax.set_title("A. Distance × reliability median NAE heatmap",
                 fontsize=12, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, label="Median NAE (95%-capped)")


def plot_panel_b_curves(ax, data: pd.DataFrame):
    """Panel B: 分距离段的校准曲线（近/中/远）"""
    d = data["tanimoto_distance"].to_numpy()
    s = data["reliability_score"].to_numpy()
    e = data["oof_error_normalized"].to_numpy()
    valid = np.isfinite(d) & np.isfinite(s) & np.isfinite(e)
    d, s, e = d[valid], s[valid], e[valid]

    edges = np.quantile(d, np.linspace(0, 1, N_DISTANCE_BINS + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9

    highlight = {
        0: ("Near (D1, closest 10%)", "#2CA02C"),
        4: ("Mid  (D5, middle 40-50%)", "#FFB84D"),
        8: ("Far  (D9, 80-90%)", "#D62728"),
        9: ("Extreme (D10, farthest 10%)", "#7F0E0E"),
    }

    for band_idx, (label, color) in highlight.items():
        dmask = (d > edges[band_idx]) & (d <= edges[band_idx + 1])
        s_bin = s[dmask]
        e_bin = e[dmask]
        if len(s_bin) < 30:
            continue

        rel_edges = np.quantile(s_bin, np.linspace(0, 1, N_RELIABILITY_BINS + 1))
        rel_edges[0] -= 1e-9
        rel_edges[-1] += 1e-9
        centers = []
        medians = []
        for j in range(N_RELIABILITY_BINS):
            rmask = (s_bin > rel_edges[j]) & (s_bin <= rel_edges[j + 1])
            if rmask.sum() > 0:
                centers.append(s_bin[rmask].mean())
                medians.append(np.median(e_bin[rmask]))

        centers = np.array(centers)
        medians = np.array(medians)
        ax.plot(centers, medians, "o-", color=color, linewidth=2,
                markersize=6, label=f"{label} (n={dmask.sum()})", alpha=0.9)

    ax.set_xlabel("Reliability score (within-band decile)", fontsize=11)
    ax.set_ylabel("Median NAE", fontsize=11)
    ax.set_title("B. Calibration holds across distance bands\n(low reliability → high error at every distance)",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.grid(alpha=0.3)


def plot_panel_c_extrapolation(ax, data: pd.DataFrame):
    """Panel C: 观测样本中结构支持最弱10%子集的校准。"""
    d = data["tanimoto_distance"].to_numpy()
    s = data["reliability_score"].to_numpy()
    e = data["oof_error_normalized"].to_numpy()
    valid = np.isfinite(d) & np.isfinite(s) & np.isfinite(e)
    d, s, e = d[valid], s[valid], e[valid]

    cutoff = np.quantile(d, 0.90)
    ext_mask = d >= cutoff
    s_ext = s[ext_mask]
    e_ext = e[ext_mask]

    rel_edges = np.quantile(s_ext, np.linspace(0, 1, N_RELIABILITY_BINS + 1))
    rel_edges[0] -= 1e-9
    rel_edges[-1] += 1e-9

    centers, medians, q25, q75, counts = [], [], [], [], []
    for j in range(N_RELIABILITY_BINS):
        m = (s_ext > rel_edges[j]) & (s_ext <= rel_edges[j + 1])
        if m.sum() > 0:
            centers.append(s_ext[m].mean())
            medians.append(np.median(e_ext[m]))
            q25.append(np.quantile(e_ext[m], 0.25))
            q75.append(np.quantile(e_ext[m], 0.75))
            counts.append(m.sum())

    centers = np.array(centers)
    medians = np.array(medians)
    q25 = np.array(q25)
    q75 = np.array(q75)

    ax.fill_between(centers, q25, q75, alpha=0.25, color="#7F0E0E",
                    label="IQR (Q25-Q75)")
    ax.plot(centers, medians, "o-", color="#7F0E0E", linewidth=2.5,
            markersize=8, label="Median NAE")

    for c, m, n in zip(centers, medians, counts):
        ax.text(c, m, f"  n={n}", fontsize=7, va="bottom")

    ax.set_xlabel("Reliability score (weak-support subset decile)", fontsize=11)
    ax.set_ylabel("NAE (weak-support subset, d ≥ P90)", fontsize=11)
    ax.set_title(f"C. Weakest structural-support subset (top-10% farthest, d ≥ {cutoff:.2f})\nreliability-error ordering within observed chemical space",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(alpha=0.3)

    if len(medians) >= 2 and medians[0] > 0:
        red = (medians[0] - medians[-1]) / medians[0] * 100
        ax.text(
            0.05, 0.95,
            f"Weak-support subset:\nlowest-decile NAE={medians[0]:.3f}\nhighest-decile NAE={medians[-1]:.3f}\nreduction={red:.1f}%\nn = {ext_mask.sum():,}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9),
            fontsize=9,
        )


def plot_panel_d_reduction(ax, data: pd.DataFrame):
    """Panel D: 每个距离段内 Low vs High reliability 的误差降低"""
    d = data["tanimoto_distance"].to_numpy()
    s = data["reliability_score"].to_numpy()
    e = data["oof_error_normalized"].to_numpy()
    valid = np.isfinite(d) & np.isfinite(s) & np.isfinite(e)
    d, s, e = d[valid], s[valid], e[valid]

    edges = np.quantile(d, np.linspace(0, 1, N_DISTANCE_BINS + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9

    reductions = []
    low_medians = []
    high_medians = []
    bin_labels = []

    for i in range(N_DISTANCE_BINS):
        dmask = (d > edges[i]) & (d <= edges[i + 1])
        s_bin = s[dmask]
        e_bin = e[dmask]
        if len(s_bin) < 30:
            reductions.append(np.nan)
            low_medians.append(np.nan)
            high_medians.append(np.nan)
            bin_labels.append(f"D{i+1}")
            continue

        thr_low = np.quantile(s_bin, 0.10)
        thr_high = np.quantile(s_bin, 0.90)
        low_e = e_bin[s_bin <= thr_low]
        high_e = e_bin[s_bin >= thr_high]

        low_med = np.median(low_e) if len(low_e) > 0 else np.nan
        high_med = np.median(high_e) if len(high_e) > 0 else np.nan

        low_medians.append(low_med)
        high_medians.append(high_med)

        if np.isfinite(low_med) and np.isfinite(high_med) and low_med > 0:
            reductions.append((low_med - high_med) / low_med * 100)
        else:
            reductions.append(np.nan)
        bin_labels.append(f"D{i+1}")

    x = np.arange(N_DISTANCE_BINS)
    bar_width = 0.35

    ax.bar(x - bar_width / 2, low_medians, bar_width, color="#D62728",
           alpha=0.85, edgecolor="black", linewidth=0.6, label="Low reliability (bottom 10%)")
    ax.bar(x + bar_width / 2, high_medians, bar_width, color="#2CA02C",
           alpha=0.85, edgecolor="black", linewidth=0.6, label="High reliability (top 10%)")

    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels)
    ax.set_xlabel("Tanimoto distance decile (near→far)", fontsize=11)
    ax.set_ylabel("Median NAE", fontsize=11)
    ax.set_title("D. Low-vs-high reliability contrast at every distance band",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=0.3, axis="y")

    ax2 = ax.twinx()
    ax2.plot(x, reductions, "o-", color="#1F77B4", linewidth=2,
             markersize=7, label="Reduction (%)")
    ax2.set_ylabel("Error reduction low→high (%)", fontsize=10, color="#1F77B4")
    ax2.tick_params(axis="y", labelcolor="#1F77B4")
    ax2.set_ylim(0, 100)


def main():
    print("=" * 80)
    print("P0-D: 结构支持距离梯度诊断（distance-gradient calibration）")
    print("=" * 80)

    ensure_dir(P0_FIGURES_DIR)

    print("\n加载数据...")
    data = load_all_reliability_data()
    print(f"总样本数: {len(data):,}")
    print(f"性质数: {data['property_name'].nunique()}")
    print()

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    plot_panel_a_heatmap(axes[0, 0], data)
    plot_panel_b_curves(axes[0, 1], data)
    plot_panel_c_extrapolation(axes[1, 0], data)
    plot_panel_d_reduction(axes[1, 1], data)

    fig.suptitle(
        "Figure P0-D: Reliability calibration across structural-support distances\n"
        "(random OOF evidence from well-supported to weakly supported observed regions)",
        fontsize=13, fontweight="bold", y=0.998,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_pdf = P0_FIGURES_DIR / "fig_p0d_extrapolation.pdf"
    output_png = P0_FIGURES_DIR / "fig_p0d_extrapolation.png"

    plt.savefig(output_pdf, bbox_inches="tight", dpi=300)
    plt.savefig(output_png, bbox_inches="tight", dpi=200)
    plt.close()

    print(f"\n✓ 已保存:")
    print(f"  {output_pdf}")
    print(f"  {output_png}")

    print("\n=== 关键统计（各距离段内的可靠性分层）===")
    d = data["tanimoto_distance"].to_numpy()
    s = data["reliability_score"].to_numpy()
    e = data["oof_error_normalized"].to_numpy()
    valid = np.isfinite(d) & np.isfinite(s) & np.isfinite(e)
    d, s, e = d[valid], s[valid], e[valid]

    edges = np.quantile(d, np.linspace(0, 1, N_DISTANCE_BINS + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9

    print("\nDistBand | d_range          |  n     | Low-NAE | High-NAE | Reduction")
    for i in range(N_DISTANCE_BINS):
        dmask = (d > edges[i]) & (d <= edges[i + 1])
        s_bin = s[dmask]
        e_bin = e[dmask]
        if len(s_bin) < 30:
            print(f"  D{i+1:<2d}   | too few samples")
            continue
        thr_low = np.quantile(s_bin, 0.10)
        thr_high = np.quantile(s_bin, 0.90)
        low_med = np.median(e_bin[s_bin <= thr_low])
        high_med = np.median(e_bin[s_bin >= thr_high])
        red = (low_med - high_med) / low_med * 100 if low_med > 0 else 0
        print(f"  D{i+1:<2d}   | [{edges[i]:.3f}, {edges[i+1]:.3f}] | {dmask.sum():5d} | "
              f"{low_med:6.3f}  | {high_med:6.3f}   | {red:6.1f}%")

    print("\n=== 观测数据中结构支持最弱子集（d ≥ P90）的校准 ===")
    cutoff = np.quantile(d, 0.90)
    ext_mask = d >= cutoff
    s_ext = s[ext_mask]
    e_ext = e[ext_mask]
    if len(s_ext) >= 100:
        rel_edges = np.quantile(s_ext, np.linspace(0, 1, N_RELIABILITY_BINS + 1))
        rel_edges[0] -= 1e-9
        rel_edges[-1] += 1e-9
        print(f"弱结构支持子集大小: n={ext_mask.sum():,} (cutoff d={cutoff:.3f})")
        print("Decile | reliability center | median NAE | n")
        for j in range(N_RELIABILITY_BINS):
            m = (s_ext > rel_edges[j]) & (s_ext <= rel_edges[j + 1])
            if m.sum() > 0:
                c = s_ext[m].mean()
                med = np.median(e_ext[m])
                print(f"  R{j+1:<2d}  |      {c:.3f}         |   {med:6.3f}   | {m.sum()}")


if __name__ == "__main__":
    main()
