"""
04_plot_p0a_calibration.py

P0-A: Reliability score vs 折外真实误差校准图

回答核心问题：可靠性分数能否识别高错误的样本？

图片包含:
- Panel A: 全局校准图（所有性质pooled）- reliability decile vs 中位NAE
- Panel B: 分类别校准图（8个类别分别显示）
- Panel C: 代表性性质校准图（6-8个）
- Panel D: 三个可靠性组件的独立校准（NRMSE only vs ensemble only vs distance only）
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
    OOF_RELIABILITY_DIR,
    P0_FIGURES_DIR,
)


plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["xtick.major.width"] = 1.2
plt.rcParams["ytick.major.width"] = 1.2

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


def load_all_reliability_data() -> pd.DataFrame:
    """加载所有性质的reliability CSV，合并为一个大DataFrame"""
    all_dfs = []
    files = sorted(OOF_RELIABILITY_DIR.glob("*_oof_reliability.csv"))

    for f in files:
        prop_name = f.stem.replace("_oof_reliability", "")
        df = pd.read_csv(f)
        df["property_name"] = prop_name
        df["category"] = infer_category(prop_name)
        all_dfs.append(df)

    if not all_dfs:
        raise RuntimeError(f"没有找到可靠性数据: {OOF_RELIABILITY_DIR}")

    return pd.concat(all_dfs, ignore_index=True)


def compute_calibration_curve(
    scores: np.ndarray,
    errors: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    计算校准曲线：按reliability分位数分bin，返回每个bin的中位误差

    Returns:
        bin_centers: 每个bin的中心reliability
        median_errors: 每个bin的中位NAE
        q25_errors: 25%分位数
        q75_errors: 75%分位数
    """
    valid_mask = np.isfinite(scores) & np.isfinite(errors)
    scores = scores[valid_mask]
    errors = errors[valid_mask]

    if len(scores) < n_bins:
        return np.array([]), np.array([]), np.array([]), np.array([])

    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(scores, quantiles)
    bin_edges[0] -= 1e-9
    bin_edges[-1] += 1e-9

    bin_centers = np.zeros(n_bins)
    median_errors = np.zeros(n_bins)
    q25 = np.zeros(n_bins)
    q75 = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (scores > bin_edges[i]) & (scores <= bin_edges[i + 1])
        if mask.sum() > 0:
            bin_centers[i] = scores[mask].mean()
            median_errors[i] = np.median(errors[mask])
            q25[i] = np.quantile(errors[mask], 0.25)
            q75[i] = np.quantile(errors[mask], 0.75)
        else:
            bin_centers[i] = (bin_edges[i] + bin_edges[i + 1]) / 2
            median_errors[i] = np.nan
            q25[i] = np.nan
            q75[i] = np.nan

    return bin_centers, median_errors, q25, q75


def plot_panel_a_global(ax, data: pd.DataFrame):
    """Panel A: 全局校准图"""
    bin_c, med, q25, q75 = compute_calibration_curve(
        data["reliability_score"].to_numpy(),
        data["oof_error_normalized"].to_numpy(),
        n_bins=10,
    )

    ax.fill_between(bin_c, q25, q75, alpha=0.25, color="#C4612F", label="IQR (Q25-Q75)")
    ax.plot(bin_c, med, "o-", color="#C4612F", linewidth=2, markersize=8, label="Median NAE")

    ax.set_xlabel("Reliability score (decile bins)", fontsize=11)
    ax.set_ylabel("Normalized absolute error (NAE)", fontsize=11)
    ax.set_title("A. Global calibration: all properties pooled", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)

    if len(bin_c) >= 3 and np.all(np.isfinite(med)):
        r_first_bin = med[0]
        r_last_bin = med[-1]
        reduction = (r_first_bin - r_last_bin) / r_first_bin * 100 if r_first_bin > 0 else 0
        ax.text(
            0.05, 0.95,
            f"Error reduction: {reduction:.1f}%\n(low→high reliability)",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9),
            fontsize=9,
        )


def plot_panel_b_by_category(ax, data: pd.DataFrame):
    """Panel B: 按类别校准"""
    categories = sorted(data["category"].unique())

    for cat in categories:
        cat_data = data[data["category"] == cat]
        if len(cat_data) < 100:
            continue

        bin_c, med, _, _ = compute_calibration_curve(
            cat_data["reliability_score"].to_numpy(),
            cat_data["oof_error_normalized"].to_numpy(),
            n_bins=10,
        )

        color = COLOR_PALETTE.get(cat, "gray")
        ax.plot(bin_c, med, "o-", color=color, linewidth=1.8, markersize=6,
                label=f"{cat} (n={len(cat_data)})", alpha=0.85)

    ax.set_xlabel("Reliability score (decile bins)", fontsize=11)
    ax.set_ylabel("Median NAE", fontsize=11)
    ax.set_title("B. Calibration by property category", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)


def plot_panel_c_representative(ax, data: pd.DataFrame):
    """Panel C: 代表性性质"""
    representative = [
        "Tg_K", "Td_K", "dielectric_constant",
        "refractive_index", "surface_tension_mN_m",
        "tensile_break_GPa", "density_g_cm3", "O2_perm_Barrer",
    ]

    available = [p for p in representative if p in data["property_name"].unique()]

    for prop in available:
        prop_data = data[data["property_name"] == prop]
        cat = infer_category(prop)
        color = COLOR_PALETTE.get(cat, "gray")

        bin_c, med, _, _ = compute_calibration_curve(
            prop_data["reliability_score"].to_numpy(),
            prop_data["oof_error_normalized"].to_numpy(),
            n_bins=10,
        )

        ax.plot(bin_c, med, "o-", color=color, linewidth=1.5, markersize=5,
                label=f"{prop}", alpha=0.75)

    ax.set_xlabel("Reliability score (decile bins)", fontsize=11)
    ax.set_ylabel("Median NAE", fontsize=11)
    ax.set_title("C. Representative properties", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)


def plot_panel_d_components(ax, data: pd.DataFrame):
    """Panel D: 三个可靠性组件独立校准（component ablation）"""

    components = {
        "Full reliability score": ("reliability_score", "#C4612F"),
        "NRMSE only (inverted)": (None, "#1F77B4"),
        "Ensemble std only (inverted)": (None, "#2CA02C"),
        "Distance-sigma interaction only": (None, "#9467BD"),
    }

    nrmse = data["property_nrmse"].to_numpy()
    sigma = data["ensemble_std_normalized"].to_numpy()
    distance_sigma = data["tanimoto_distance"].to_numpy() * sigma
    nrmse_inv = 1.0 - nrmse / (1.0 + nrmse)
    ens_inv = 1.0 - sigma / (1.0 + sigma)
    distance_sigma_inv = 1.0 - distance_sigma / (1.0 + distance_sigma)

    scores_dict = {
        "Full reliability score": data["reliability_score"].to_numpy(),
        "NRMSE only (inverted)": nrmse_inv,
        "Ensemble std only (inverted)": ens_inv,
        "Distance-sigma interaction only": distance_sigma_inv,
    }

    errors = data["oof_error_normalized"].to_numpy()

    for label, (_, color) in components.items():
        scores = scores_dict[label]
        bin_c, med, _, _ = compute_calibration_curve(scores, errors, n_bins=10)

        lw = 2.5 if "Full" in label else 1.5
        alpha = 1.0 if "Full" in label else 0.75

        ax.plot(bin_c, med, "o-", color=color, linewidth=lw, markersize=6,
                label=label, alpha=alpha)

    ax.set_xlabel("Component score (decile bins)", fontsize=11)
    ax.set_ylabel("Median NAE", fontsize=11)
    ax.set_title("D. Component ablation", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)


def main():
    print("=" * 80)
    print("P0-A: Reliability校准图")
    print("=" * 80)

    ensure_dir(P0_FIGURES_DIR)

    print("\n加载所有可靠性数据...")
    data = load_all_reliability_data()
    print(f"总样本数: {len(data)}")
    print(f"性质数: {data['property_name'].nunique()}")
    print(f"类别数: {data['category'].nunique()}")
    print()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    plot_panel_a_global(axes[0, 0], data)
    plot_panel_b_by_category(axes[0, 1], data)
    plot_panel_c_representative(axes[1, 0], data)
    plot_panel_d_components(axes[1, 1], data)

    fig.suptitle(
        "Figure P0-A: Reliability score calibration against out-of-fold error",
        fontsize=14, fontweight="bold", y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    output_pdf = P0_FIGURES_DIR / "fig_p0a_calibration.pdf"
    output_png = P0_FIGURES_DIR / "fig_p0a_calibration.png"

    plt.savefig(output_pdf, bbox_inches="tight", dpi=300)
    plt.savefig(output_png, bbox_inches="tight", dpi=200)
    plt.close()

    print(f"\n✓ 已保存:")
    print(f"  {output_pdf}")
    print(f"  {output_png}")

    print("\n=== 关键统计 ===")
    for cat in sorted(data["category"].unique()):
        cat_data = data[data["category"] == cat]
        bin_c, med, _, _ = compute_calibration_curve(
            cat_data["reliability_score"].to_numpy(),
            cat_data["oof_error_normalized"].to_numpy(),
            n_bins=10,
        )
        if len(med) > 0 and np.isfinite(med[0]) and np.isfinite(med[-1]) and med[0] > 0:
            reduction = (med[0] - med[-1]) / med[0] * 100
            print(f"  {cat:12s} (n={len(cat_data):6d}): "
                  f"lowest-decile NAE={med[0]:.3f}, "
                  f"highest-decile NAE={med[-1]:.3f}, "
                  f"reduction={reduction:5.1f}%")


if __name__ == "__main__":
    main()
