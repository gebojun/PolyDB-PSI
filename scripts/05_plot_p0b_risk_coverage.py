"""
05_plot_p0b_risk_coverage.py

P0-B: Risk-coverage 选择性保留曲线

回答核心问题：reliability能否用于数据库筛选？

图片包含:
- Panel A: 全局risk-coverage（reliability-ranked vs random）
- Panel B: 分类别risk-coverage
- Panel C: 高错误样本捕获率（低reliability捕获高错误）
- Panel D: 富集因子（enrichment factor）
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
    all_dfs = []
    for f in sorted(OOF_RELIABILITY_DIR.glob("*_oof_reliability.csv")):
        prop_name = f.stem.replace("_oof_reliability", "")
        df = pd.read_csv(f)
        df["property_name"] = prop_name
        df["category"] = infer_category(prop_name)
        all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)


def compute_risk_coverage_curve(
    scores: np.ndarray,
    errors: np.ndarray,
    coverages: np.ndarray,
    ascending: bool = False,
) -> np.ndarray:
    """
    计算risk-coverage曲线

    Args:
        scores: reliability分数
        errors: 归一化误差
        coverages: 保留比例数组 (0-1)
        ascending: False=按reliability从高到低保留（正常）
                   True=按reliability从低到高保留（用于错误检测）

    Returns:
        每个coverage下被保留样本的平均误差
    """
    n = len(scores)
    order = np.argsort(scores) if ascending else np.argsort(-scores)
    sorted_errors = errors[order]

    result = np.zeros(len(coverages))
    for i, cov in enumerate(coverages):
        n_keep = max(1, int(n * cov))
        result[i] = sorted_errors[:n_keep].mean()

    return result


def compute_random_baseline(errors: np.ndarray, coverages: np.ndarray, n_repeats: int = 50) -> np.ndarray:
    """随机选择的baseline"""
    n = len(errors)
    rng = np.random.default_rng(42)
    means = np.zeros((n_repeats, len(coverages)))

    for i in range(n_repeats):
        shuffled = errors[rng.permutation(n)]
        for j, cov in enumerate(coverages):
            n_keep = max(1, int(n * cov))
            means[i, j] = shuffled[:n_keep].mean()

    return means.mean(axis=0)


def compute_enrichment(
    scores: np.ndarray,
    errors: np.ndarray,
    budgets: np.ndarray,
    error_threshold: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    计算低reliability选择的富集因子

    Args:
        error_threshold: NAE > 该阈值视为"高错误样本"

    Returns:
        capture_rate: 各budget下低reliability选择捕获高错误的比例
        enrichment: 相对随机的富集因子
    """
    n = len(scores)
    is_high_error = errors > error_threshold
    total_high_error = is_high_error.sum()

    if total_high_error == 0:
        return np.zeros(len(budgets)), np.ones(len(budgets))

    order_asc = np.argsort(scores)
    sorted_is_high = is_high_error[order_asc]

    capture_rate = np.zeros(len(budgets))
    enrichment = np.zeros(len(budgets))

    for i, budget in enumerate(budgets):
        n_select = max(1, int(n * budget))
        captured = sorted_is_high[:n_select].sum()
        capture_rate[i] = captured / total_high_error
        random_capture = budget
        enrichment[i] = capture_rate[i] / random_capture if random_capture > 0 else 0

    return capture_rate, enrichment


def plot_panel_a_global(ax, data: pd.DataFrame):
    """Panel A: 全局risk-coverage曲线"""
    coverages = np.linspace(0.05, 1.0, 20)
    scores = data["reliability_score"].to_numpy()
    errors = data["oof_error_normalized"].to_numpy()

    valid = np.isfinite(scores) & np.isfinite(errors)
    scores, errors = scores[valid], errors[valid]

    rel_curve = compute_risk_coverage_curve(scores, errors, coverages)
    random_curve = compute_random_baseline(errors, coverages)

    ax.plot(coverages * 100, rel_curve, "o-", color="#C4612F", linewidth=2.5,
            markersize=7, label="Reliability-ranked (high→low)")
    ax.plot(coverages * 100, random_curve, "s--", color="#5C635D", linewidth=2,
            markersize=6, label="Random selection", alpha=0.7)

    ax.set_xlabel("Coverage retained (%)", fontsize=11)
    ax.set_ylabel("Mean NAE of retained samples", fontsize=11)
    ax.set_title("A. Risk-coverage: all properties pooled", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=0.3)

    error_at_50 = rel_curve[9]
    random_at_50 = random_curve[9]
    reduction = (random_at_50 - error_at_50) / random_at_50 * 100 if random_at_50 > 0 else 0

    ax.text(
        0.98, 0.02,
        f"At 50% coverage:\nreliability={error_at_50:.3f}\nrandom={random_at_50:.3f}\nreduction={reduction:.1f}%",
        transform=ax.transAxes,
        horizontalalignment="right",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9),
        fontsize=9,
    )


def plot_panel_b_by_category(ax, data: pd.DataFrame):
    """Panel B: 分类别risk-coverage"""
    coverages = np.linspace(0.05, 1.0, 20)

    for cat in sorted(data["category"].unique()):
        cat_data = data[data["category"] == cat]
        if len(cat_data) < 100:
            continue

        scores = cat_data["reliability_score"].to_numpy()
        errors = cat_data["oof_error_normalized"].to_numpy()
        valid = np.isfinite(scores) & np.isfinite(errors)
        scores, errors = scores[valid], errors[valid]

        rel_curve = compute_risk_coverage_curve(scores, errors, coverages)

        color = COLOR_PALETTE.get(cat, "gray")
        ax.plot(coverages * 100, rel_curve, "o-", color=color, linewidth=1.8,
                markersize=5, label=f"{cat} (n={len(cat_data)})", alpha=0.85)

    ax.set_xlabel("Coverage retained (%)", fontsize=11)
    ax.set_ylabel("Mean NAE of retained samples", fontsize=11)
    ax.set_title("B. Risk-coverage by category (reliability-ranked)", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", frameon=False, fontsize=8, ncol=2)
    ax.grid(alpha=0.3)


def plot_panel_c_capture_rate(ax, data: pd.DataFrame):
    """Panel C: 高错误样本捕获率"""
    budgets = np.linspace(0.05, 0.5, 10)
    scores = data["reliability_score"].to_numpy()
    errors = data["oof_error_normalized"].to_numpy()

    valid = np.isfinite(scores) & np.isfinite(errors)
    scores, errors = scores[valid], errors[valid]

    capture_rate, _ = compute_enrichment(scores, errors, budgets, error_threshold=1.0)
    random_capture = budgets

    ax.plot(budgets * 100, capture_rate * 100, "o-", color="#C4612F",
            linewidth=2.5, markersize=7, label="Low-reliability selection")
    ax.plot(budgets * 100, random_capture * 100, "--", color="#5C635D",
            linewidth=2, label="Random baseline (y=x)")

    ax.fill_between(budgets * 100, random_capture * 100, capture_rate * 100,
                    alpha=0.2, color="#C4612F")

    ax.set_xlabel("Verification budget (%)", fontsize=11)
    ax.set_ylabel("High-error samples captured (%)", fontsize=11)
    ax.set_title("C. Virtual audit: capture high-error samples\n(NAE > 1 stddev)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=0.3)

    if len(capture_rate) > 1:
        c_at_10 = capture_rate[1]
        ax.text(
            0.98, 0.02,
            f"At 10% budget:\ncaptures {c_at_10*100:.1f}% of\nhigh-error samples\n(random: 10%)",
            transform=ax.transAxes,
            horizontalalignment="right",
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9),
            fontsize=9,
        )


def plot_panel_d_enrichment(ax, data: pd.DataFrame):
    """Panel D: 富集因子（分类别）"""
    budgets = np.array([0.05, 0.10, 0.15, 0.20, 0.30, 0.50])

    cats = sorted(data["category"].unique())
    x_pos = np.arange(len(budgets))
    bar_width = 0.1

    global_scores = data["reliability_score"].to_numpy()
    global_errors = data["oof_error_normalized"].to_numpy()
    valid = np.isfinite(global_scores) & np.isfinite(global_errors)
    _, global_enrichment = compute_enrichment(
        global_scores[valid], global_errors[valid], budgets, error_threshold=1.0
    )

    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="No enrichment (=1.0)")
    ax.plot(x_pos, global_enrichment, "o-", color="black", linewidth=2.5,
            markersize=10, label="Global (all properties)", zorder=10)

    for i, cat in enumerate(cats):
        cat_data = data[data["category"] == cat]
        if len(cat_data) < 100:
            continue
        scores = cat_data["reliability_score"].to_numpy()
        errors = cat_data["oof_error_normalized"].to_numpy()
        valid = np.isfinite(scores) & np.isfinite(errors)
        _, cat_enrichment = compute_enrichment(
            scores[valid], errors[valid], budgets, error_threshold=1.0
        )
        color = COLOR_PALETTE.get(cat, "gray")
        ax.plot(x_pos, cat_enrichment, "s-", color=color, linewidth=1.2,
                markersize=5, label=cat, alpha=0.7)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{int(b*100)}%" for b in budgets])
    ax.set_xlabel("Verification budget", fontsize=11)
    ax.set_ylabel("Enrichment factor\n(low-reliability / random)", fontsize=11)
    ax.set_title("D. Enrichment factor by category", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=2)
    ax.grid(alpha=0.3)


def main():
    print("=" * 80)
    print("P0-B: Risk-coverage 选择性保留曲线")
    print("=" * 80)

    ensure_dir(P0_FIGURES_DIR)

    print("\n加载所有可靠性数据...")
    data = load_all_reliability_data()
    print(f"总样本数: {len(data)}")
    print(f"性质数: {data['property_name'].nunique()}")
    print()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    plot_panel_a_global(axes[0, 0], data)
    plot_panel_b_by_category(axes[0, 1], data)
    plot_panel_c_capture_rate(axes[1, 0], data)
    plot_panel_d_enrichment(axes[1, 1], data)

    fig.suptitle(
        "Figure P0-B: Reliability-guided selective retention and verification",
        fontsize=14, fontweight="bold", y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    output_pdf = P0_FIGURES_DIR / "fig_p0b_risk_coverage.pdf"
    output_png = P0_FIGURES_DIR / "fig_p0b_risk_coverage.png"

    plt.savefig(output_pdf, bbox_inches="tight", dpi=300)
    plt.savefig(output_png, bbox_inches="tight", dpi=200)
    plt.close()

    print(f"\n✓ 已保存:")
    print(f"  {output_pdf}")
    print(f"  {output_png}")

    print("\n=== 关键统计 ===")

    coverages_report = np.array([0.10, 0.25, 0.50, 0.75])
    scores = data["reliability_score"].to_numpy()
    errors = data["oof_error_normalized"].to_numpy()
    valid = np.isfinite(scores) & np.isfinite(errors)
    scores, errors = scores[valid], errors[valid]

    rel_curve = compute_risk_coverage_curve(scores, errors, coverages_report)
    random_curve = compute_random_baseline(errors, coverages_report)

    print("\nCoverage | Reliability-ranked NAE | Random NAE | Reduction")
    for i, cov in enumerate(coverages_report):
        reduction = (random_curve[i] - rel_curve[i]) / random_curve[i] * 100 if random_curve[i] > 0 else 0
        print(f"  {int(cov*100):3d}%   |        {rel_curve[i]:.4f}         |   {random_curve[i]:.4f}   |  {reduction:5.1f}%")

    budgets = np.array([0.05, 0.10, 0.20, 0.30])
    capture, enrichment = compute_enrichment(scores, errors, budgets, error_threshold=1.0)
    print("\nBudget  | High-error captured | Random | Enrichment")
    for i, b in enumerate(budgets):
        print(f"  {int(b*100):3d}%   |      {capture[i]*100:5.1f}%         |  {b*100:5.1f}%  |    {enrichment[i]:4.2f}x")


if __name__ == "__main__":
    main()
