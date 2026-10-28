"""Local Tkinter tuner for the flexural-storage-modulus reliability UMAP.

The tool is intentionally a Python desktop utility, not an HTML application.
Rendering controls update immediately. UMAP layout controls recompute the
coordinates in a background worker, while keeping polymer rows,
flexural-storage-modulus values, and cell-level reliability scores unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import re
import threading
import time
from pathlib import Path
from tkinter import colorchooser, messagebox, ttk
import tkinter as tk

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams.update({"font.family": "Arial", "font.sans-serif": ["Arial", "DejaVu Sans"]})

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd
import umap
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
# This file supplies the global polymer IDs, class labels, and a ready-made
# initial coordinate layout. Property values and PSI are loaded separately.
DEFAULT_ASSIGNMENTS = (
    PROJECT_ROOT
    / "outputs/21_representative_property_reliability_umaps_05_10/"
    / "electrical_dielectric_constant/"
    / "polymer_class_and_dielectric_constant_reliability_assignments.csv.gz"
)
DEFAULT_DATASET = (
    PROJECT_ROOT
    / "input_data/merged dataset/dataset_v10_maccs_drop30_invalidpsmiles.csv"
)
DEFAULT_RELIABILITY = (
    PROJECT_ROOT
    / "outputs/03_missing_predictions/flexural_storage_GPa_missing_reliability.csv"
)
DEFAULT_COORDINATE_DIR = PROJECT_ROOT / "outputs/14_structural_embedding_comparison"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/32_flexural_storage_GPa_umap_interactive_widget"
PROPERTY_COLUMN = "flexural_storage_GPa"
PROPERTY_SYMBOL = r"$F_{x,S}$"
PROPERTY_UNIT = "GPa"
PROPERTY_LABEL = f"{PROPERTY_SYMBOL} ({PROPERTY_UNIT})"
DEFAULT_FIGURE_HEIGHT = 6.8
DEFAULT_FIGURE_WIDTH = DEFAULT_FIGURE_HEIGHT * 23.0 / 11.0

# These existing coordinate files provide a fast initial preview and reference
# presets. Custom combinations are recomputed by the tuner itself.
LAYOUT_SPECS = [
    {
        "fingerprint": "MACCS 167-bit",
        "n_neighbors": "25",
        "min_dist": "0.08",
        "variant": "spread 0.5",
        "file": "coordinates_maccs_k_25_min_dist_0_08_spread_0_5.csv",
    },
    {
        "fingerprint": "MACCS 167-bit",
        "n_neighbors": "25",
        "min_dist": "0.08",
        "variant": "standard",
        "file": "coordinates_maccs_k_25_min_dist_0_08.csv",
    },
    {
        "fingerprint": "MACCS 167-bit",
        "n_neighbors": "25",
        "min_dist": "0.12",
        "variant": "standard",
        "file": "coordinates_maccs_k_25_min_dist_0_12_current.csv",
    },
    {
        "fingerprint": "MACCS 167-bit",
        "n_neighbors": "100",
        "min_dist": "0.25",
        "variant": "standard",
        "file": "coordinates_maccs_k_100_min_dist_0_25.csv",
    },
    {
        "fingerprint": "MACCS 167-bit",
        "n_neighbors": "200",
        "min_dist": "0.35",
        "variant": "PCA initialization",
        "file": "coordinates_maccs_k_200_min_dist_0_35_pca_init.csv",
    },
    {
        "fingerprint": "Morgan 1024-bit",
        "n_neighbors": "100",
        "min_dist": "0.25",
        "variant": "standard",
        "file": "coordinates_morgan_1024_k_100_min_dist_0_25.csv",
    },
]

# Kept as a small compatibility map for exported settings and external users.
LAYOUTS = {
    f"{item['fingerprint']} · k{item['n_neighbors']} · min_dist {item['min_dist']} · {item['variant']}": item["file"]
    for item in LAYOUT_SPECS
}

CLASS_ORDER = [
    "Polyimide", "Polyurethane", "Polyamide", "Polyester", "Polycarbonate",
    "Polyether", "Polysulfone", "Polysiloxane", "Polyolefin", "Fluoropolymer",
    "Polyacrylate", "Polystyrene", "Mixed/unclear",
]
DEFAULT_CLASS_COLORS = {
    "Polyimide": "#6F2DA8", "Polyurethane": "#E66101", "Polyamide": "#1F77B4",
    "Polyester": "#2CA02C", "Polycarbonate": "#D4A017", "Polyether": "#8C564B",
    "Polysulfone": "#B2188B", "Polysiloxane": "#00A6A6", "Polyolefin": "#7A8B00",
    "Fluoropolymer": "#003F5C", "Polyacrylate": "#E41A1C", "Polystyrene": "#FF7FAD",
    "Mixed/unclear": "#A7ABB0",
}


def read_assignment_table(
    path: Path,
    dataset_path: Path,
    reliability_path: Path,
) -> pd.DataFrame:
    """Combine class/coordinate metadata with this property's values and PSI."""
    required = {"polymer_id", "smiles", "display_class", "umap_1", "umap_2"}
    table = pd.read_csv(path, compression="infer")
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"类别与坐标文件缺少字段: {sorted(missing)}")
    table = table.copy()
    table["polymer_id"] = pd.to_numeric(table["polymer_id"], errors="raise").astype(int)
    if table["polymer_id"].duplicated().any():
        raise ValueError(f"类别与坐标文件包含重复 polymer_id: {path}")

    dataset = pd.read_csv(dataset_path, usecols=["smiles", PROPERTY_COLUMN])
    if len(dataset) != len(table) or not dataset["smiles"].astype(str).equals(
        table["smiles"].astype(str)
    ):
        raise ValueError("原始数据集与类别/坐标文件中的聚合物行不一致")
    table[PROPERTY_COLUMN] = pd.to_numeric(dataset[PROPERTY_COLUMN], errors="coerce")

    reliability = pd.read_csv(
        reliability_path,
        usecols=["polymer_id", "reliability_score"],
    )
    reliability["polymer_id"] = pd.to_numeric(
        reliability["polymer_id"], errors="raise"
    ).astype(int)
    reliability["reliability_score"] = pd.to_numeric(
        reliability["reliability_score"], errors="coerce"
    )
    if reliability["polymer_id"].duplicated().any():
        raise ValueError(f"可靠性文件包含重复 polymer_id: {reliability_path}")
    if not reliability["polymer_id"].between(0, len(table) - 1).all():
        raise ValueError(f"可靠性文件包含越界 polymer_id: {reliability_path}")
    score_by_id = reliability.set_index("polymer_id")["reliability_score"]
    table["property_reliability_score"] = table["polymer_id"].map(score_by_id)

    for col in ["property_reliability_score", "umap_1", "umap_2"]:
        table[col] = pd.to_numeric(table[col], errors="coerce")
    return table


def load_layout(table: pd.DataFrame, path: Path) -> pd.DataFrame:
    coords = pd.read_csv(path)
    if {"dimension_1", "dimension_2"}.issubset(coords.columns):
        coords = coords.rename(columns={"dimension_1": "umap_1", "dimension_2": "umap_2"})
    required = {"polymer_id", "smiles", "umap_1", "umap_2"}
    if not required.issubset(coords.columns):
        raise ValueError(f"坐标文件缺少字段: {sorted(required.difference(coords.columns))}")
    coords = coords.sort_values("polymer_id").reset_index(drop=True)
    base = table.sort_values("polymer_id").reset_index(drop=True)
    if len(coords) != len(base) or not coords["smiles"].astype(str).equals(base["smiles"].astype(str)):
        raise ValueError(f"坐标文件与 {PROPERTY_COLUMN} 聚合物表不匹配: {path}")
    result = table.copy()
    by_id = coords.set_index("polymer_id")
    result["umap_1"] = result["polymer_id"].map(by_id["umap_1"]).astype(float)
    result["umap_2"] = result["polymer_id"].map(by_id["umap_2"]).astype(float)
    return result


def normalize_attachment_placeholders(smiles: str) -> str:
    """Make the attachment labels used by pSMILES parseable by RDKit."""
    return re.sub(r"\[R\d*\]", "[*]", str(smiles))


def fingerprint_matrix(smiles: pd.Series, fingerprint_name: str, progress_callback=None) -> np.ndarray:
    """Generate the selected binary fingerprint once for the full polymer table."""
    if fingerprint_name == "MACCS 167-bit":
        n_bits = 167
    elif fingerprint_name == "Morgan 1024-bit":
        n_bits = 1024
    else:
        raise ValueError(f"未知指纹类型: {fingerprint_name}")
    matrix = np.zeros((len(smiles), n_bits), dtype=np.uint8)
    morgan_generator = (
        rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
        if fingerprint_name == "Morgan 1024-bit"
        else None
    )
    invalid = 0
    total = len(smiles)
    for row, value in enumerate(smiles.astype(str)):
        mol = Chem.MolFromSmiles(normalize_attachment_placeholders(value))
        if mol is None:
            invalid += 1
            continue
        if fingerprint_name == "MACCS 167-bit":
            fp = MACCSkeys.GenMACCSKeys(mol)
        else:
            fp = morgan_generator.GetFingerprint(mol)
        DataStructs.ConvertToNumpyArray(fp, matrix[row])
        completed = row + 1
        if progress_callback is not None and (completed == total or completed % 250 == 0):
            progress_callback(completed, total)
    if invalid:
        raise ValueError(f"共有 {invalid:,} 个 pSMILES 无法生成 {fingerprint_name} 指纹")
    return matrix


def robust_limits(table: pd.DataFrame, trim: float) -> tuple[tuple[float, float], tuple[float, float]]:
    x = table["umap_1"].to_numpy(float)
    y = table["umap_2"].to_numpy(float)
    xl, xh = np.quantile(x, [trim, 1.0 - trim])
    yl, yh = np.quantile(y, [trim, 1.0 - trim])
    return (float(xl - (xh - xl) * 0.018), float(xh + (xh - xl) * 0.018)), (
        float(yl - (yh - yl) * 0.022), float(yh + (yh - yl) * 0.022)
    )


def style_axis(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("UMAP 1", fontsize=11, fontweight="bold")
    ax.set_ylabel("UMAP 2", fontsize=11, fontweight="bold")
    ax.tick_params(axis="both", labelsize=9, length=3, width=0.8)
    ax.set_facecolor("#FCFCFD")
    ax.grid(color="#C9CDD3", linewidth=0.55, alpha=0.30, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#5E6268")
        spine.set_linewidth(0.8)


def draw_figure(table: pd.DataFrame, state: dict, for_preview: bool = False) -> Figure:
    shown_classes = {name for name, var in state["class_vars"].items() if var.get()}
    trim = float(state["trim"].get())
    xlim, ylim = robust_limits(table, trim)
    x_raw = table["umap_1"].to_numpy(float)
    y_raw = table["umap_2"].to_numpy(float)
    finite = np.isfinite(x_raw) & np.isfinite(y_raw)
    # Keep every finite polymer visible while using a compact central window.
    # Outlying coordinates are placed on the corresponding plot boundary.
    x = x_raw
    y = y_raw
    # Remove edge/outlier points from the plotted window instead of pinning
    # them to the axes boundary, which creates distracting edge stripes.
    in_window = (
        finite
        & (x >= xlim[0])
        & (x <= xlim[1])
        & (y >= ylim[0])
        & (y <= ylim[1])
    )
    fig = Figure(figsize=(float(state["figure_width"].get()), float(state["figure_height"].get())), dpi=125 if for_preview else 300)
    ax_b, ax_c = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.0, 1.04]})
    point = float(state["point_size"].get())

    # Panel B: one point is one polymer; class colours are user-editable.
    class_mask = in_window & table["display_class"].isin(shown_classes).to_numpy()
    for name in CLASS_ORDER:
        mask = class_mask & table["display_class"].eq(name).to_numpy()
        if np.any(mask):
            ax_b.scatter(x[mask], y[mask], s=point, color=state["colors"][name],
                         alpha=float(state["class_alpha"].get()), linewidths=0, rasterized=True, zorder=2)
    style_axis(ax_b, xlim, ylim)
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=state["colors"][name],
                      markeredgecolor="none", markersize=7, label=name)
               for name in CLASS_ORDER if name in shown_classes]
    ax_b.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.08),
                frameon=False, ncol=4, fontsize=9, handletextpad=0.25,
                columnspacing=0.9, labelspacing=0.3)

    # Panel C: observed values are neutral; every prediction keeps its own score.
    observed = in_window & table[PROPERTY_COLUMN].notna().to_numpy()
    predicted = in_window & table[PROPERTY_COLUMN].isna().to_numpy() & table["property_reliability_score"].notna().to_numpy()
    unscored = in_window & table[PROPERTY_COLUMN].isna().to_numpy() & ~table["property_reliability_score"].notna().to_numpy()
    if state["show_observed"].get() and np.any(observed):
        ax_c.scatter(x[observed], y[observed], s=point, color="#B7BCC3",
                     alpha=float(state["observed_alpha"].get()), linewidths=0, rasterized=True, zorder=1)
    if state["show_unscored"].get() and np.any(unscored):
        ax_c.scatter(x[unscored], y[unscored], s=max(point * 0.75, 0.1), color="#777D85",
                     alpha=0.45, marker="x", linewidths=0.65, rasterized=True, zorder=2)
    if state["show_prediction"].get() and np.any(predicted):
        scores = table["property_reliability_score"].to_numpy(float)
        order = np.where(predicted)[0]
        # Draw low-PSI predictions last so potentially unreliable regions stay
        # visible instead of being covered by high-PSI points.
        order = order[np.argsort(scores[order])[::-1]]
        scatter = ax_c.scatter(x[order], y[order], c=scores[order], cmap="RdYlGn",
                               norm=Normalize(vmin=float(state["score_min"].get()),
                                              vmax=float(state["score_max"].get()), clip=True),
                               s=point, alpha=float(state["prediction_alpha"].get()),
                               linewidths=0, rasterized=True, zorder=3)
        cbar = fig.colorbar(scatter, ax=ax_c, fraction=0.043, pad=0.025)
        cbar.set_label("PSI", fontsize=9.5, labelpad=6)
        cbar.ax.tick_params(labelsize=7.5)
        cbar.outline.set_linewidth(0.6)
    style_axis(ax_c, xlim, ylim)
    handles = []
    if state["show_observed"].get():
        handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor="#B7BCC3",
                              markeredgecolor="none", alpha=0.7, markersize=8, label=f"Observed {PROPERTY_LABEL}"))
    if state["show_prediction"].get():
        handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor="#55A868",
                              markeredgecolor="none", alpha=0.85, markersize=8, label=f"Prediction {PROPERTY_LABEL}"))
    if handles:
        ax_c.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.08),
                    frameon=False, ncol=len(handles), fontsize=9, handletextpad=0.3,
                    columnspacing=1.0, labelspacing=0.35)
    fig.subplots_adjust(left=0.065, right=0.955, top=0.95, bottom=0.16, wspace=0.26)
    return fig


def figure_to_photoimage(fig: Figure) -> tk.PhotoImage:
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    width, height = canvas.get_width_height()
    rgb = np.ascontiguousarray(np.asarray(canvas.buffer_rgba(), dtype=np.uint8)[:, :, :3])
    ppm = f"P6\n{width} {height}\n255\n".encode("ascii") + rgb.tobytes()
    return tk.PhotoImage(data=ppm, format="PPM")


def save_figure(fig: Figure, png: Path, pdf: Path, dpi: int = 450) -> None:
    """Keep the selected canvas aspect ratio when trimming export margins."""
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    bounds = fig.get_tightbbox(canvas.get_renderer()).padded(0.1)
    target_ratio = float(fig.get_figwidth() / fig.get_figheight())
    width, height = bounds.width, bounds.height
    if width / height < target_ratio:
        width = height * target_ratio
    else:
        height = width / target_ratio
    center_x = (bounds.x0 + bounds.x1) / 2.0
    center_y = (bounds.y0 + bounds.y1) / 2.0
    export_bounds = Bbox.from_bounds(center_x-width/2.0, center_y-height/2.0, width, height)
    for path in (png, pdf):
        fig.savefig(path, dpi=dpi, bbox_inches=export_bounds, facecolor="white")


class FlexuralStorageUmapTuner:
    def __init__(self, root: tk.Tk, table: pd.DataFrame, coordinate_dir: Path, output_dir: Path):
        self.root = root
        self.base_table = table.copy()
        self.table = table.copy()
        self.coordinate_dir = coordinate_dir
        self.output_dir = output_dir
        self.layout_cache_dir = output_dir / "layout_cache"
        self.layout_cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.root.title("modulus UMAP 调参器 | Polymer class + reliability")
        self.root.geometry("1580x960")
        self.root.minsize(1200, 760)
        self.render_after = None
        self.layout_after = None
        self.photo = None
        self.rendering = False
        self.layout_running = False
        self.layout_generation = 0
        self.controls_ready = False
        self.progress_queue = queue.Queue()
        self.progress_value = tk.DoubleVar(value=0.0)
        self.progress_text = tk.StringVar(value="等待参数调整")
        self.progress_base_message = "等待参数调整"
        self.progress_is_indeterminate = False
        self.layout_started_at = None
        self.fingerprint_cache = {}
        self.colors = dict(DEFAULT_CLASS_COLORS)
        self.class_vars = {}
        counts = table["display_class"].value_counts()
        counts = counts.drop(labels="Mixed/unclear", errors="ignore")
        top = set(counts.sort_values(ascending=False, kind="stable").head(12).index)
        first_spec = next(
            item for item in LAYOUT_SPECS
            if item["n_neighbors"] == "200" and item["min_dist"] == "0.35"
        )
        self.state = {
            "fingerprint": tk.StringVar(value=first_spec["fingerprint"]),
            "metric": tk.StringVar(value="Jaccard / binary Tanimoto"),
            "n_neighbors": tk.IntVar(value=int(first_spec["n_neighbors"])),
            "min_dist": tk.DoubleVar(value=float(first_spec["min_dist"])),
            "spread": tk.DoubleVar(value=1.00),
            "init": tk.StringVar(value="pca"),
            "random_seed": tk.IntVar(value=42),
            "set_op_mix_ratio": tk.DoubleVar(value=1.00),
            "local_connectivity": tk.DoubleVar(value=1.00),
            "repulsion_strength": tk.DoubleVar(value=1.00),
            "negative_sample_rate": tk.IntVar(value=5),
            "learning_rate": tk.DoubleVar(value=1.00),
            "n_epochs": tk.StringVar(value="auto"),
            "densmap": tk.BooleanVar(value=False),
            "trim": tk.DoubleVar(value=0.05), "point_size": tk.DoubleVar(value=6.25),
            "class_alpha": tk.DoubleVar(value=0.62), "observed_alpha": tk.DoubleVar(value=0.32),
            "prediction_alpha": tk.DoubleVar(value=0.70), "score_min": tk.DoubleVar(value=0.50),
            "score_max": tk.DoubleVar(value=1.00), "figure_width": tk.DoubleVar(value=DEFAULT_FIGURE_WIDTH),
            "figure_height": tk.DoubleVar(value=DEFAULT_FIGURE_HEIGHT), "show_observed": tk.BooleanVar(value=True),
            "show_prediction": tk.BooleanVar(value=True), "show_unscored": tk.BooleanVar(value=False),
            "class_vars": self.class_vars, "colors": self.colors,
        }
        for name in CLASS_ORDER:
            self.class_vars[name] = tk.BooleanVar(value=name in top)
        self.status = tk.StringVar(value="正在读取初始布局...")
        self._build_ui()
        self.controls_ready = True
        self._load_initial_layout()
        self.root.after(150, self._poll_progress)

    def _build_ui(self):
        outer = ttk.Panedwindow(self.root, orient="horizontal")
        outer.pack(fill="both", expand=True)
        left = ttk.Frame(outer, padding=10); right = ttk.Frame(outer, padding=(4, 8, 10, 8))
        outer.add(left, weight=0); outer.add(right, weight=1)
        canvas = tk.Canvas(left, highlightthickness=0, width=390)
        scroll = ttk.Scrollbar(left, orient="vertical", command=canvas.yview)
        controls = ttk.Frame(canvas, padding=(0, 0, 8, 0))
        controls.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=controls, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set); canvas.pack(side="left", fill="both", expand=True); scroll.pack(side="right", fill="y")
        ttk.Label(controls, text="modulus UMAP 可视化调参", font=("Arial", 15, "bold")).pack(anchor="w", pady=(0, 5))
        ttk.Label(
            controls,
            text="每个点对应一个聚合物。布局参数会重新计算 UMAP 坐标，但不会修改性质值或可靠性分数。",
            wraplength=360,
        ).pack(anchor="w", pady=(0, 8))
        frame = ttk.LabelFrame(controls, text="结构表征", padding=8); frame.pack(fill="x", pady=4)
        self._add_combo(frame, "指纹类型", self.state["fingerprint"], ["MACCS 167-bit", "Morgan 1024-bit"], self._layout_param_changed)
        self._add_combo(frame, "距离度量", self.state["metric"], ["Jaccard / binary Tanimoto", "Dice", "Hamming", "Cosine"], self._layout_param_changed)
        ttk.Label(frame, text="Jaccard 是二值 MACCS/Morgan 指纹对应的 Tanimoto 距离。", wraplength=340).pack(anchor="w", pady=(3, 2))
        frame = ttk.LabelFrame(controls, text="UMAP 局部与整体结构", padding=8); frame.pack(fill="x", pady=4)
        self._add_layout_scale(frame, "邻居数", self.state["n_neighbors"], 5, 500, 5)
        self._add_layout_scale(frame, "min_dist", self.state["min_dist"], 0.00, 0.99, 0.01)
        self._add_layout_scale(frame, "spread", self.state["spread"], 0.50, 3.00, 0.05)
        self._add_combo(frame, "初始化", self.state["init"], ["spectral", "pca", "random"], self._layout_param_changed)
        self._add_layout_scale(frame, "随机种子", self.state["random_seed"], 0, 999, 1)
        frame = ttk.LabelFrame(controls, text="图结构与优化变体", padding=8); frame.pack(fill="x", pady=4)
        self._add_layout_scale(frame, "邻域混合", self.state["set_op_mix_ratio"], 0.00, 1.00, 0.05)
        self._add_layout_scale(frame, "局部连通", self.state["local_connectivity"], 1.00, 5.00, 0.25)
        self._add_layout_scale(frame, "排斥强度", self.state["repulsion_strength"], 0.10, 3.00, 0.10)
        self._add_layout_scale(frame, "负采样率", self.state["negative_sample_rate"], 1, 20, 1)
        self._add_layout_scale(frame, "学习率", self.state["learning_rate"], 0.10, 3.00, 0.10)
        self._add_combo(frame, "训练轮数", self.state["n_epochs"], ["auto", "200", "500", "1000"], self._layout_param_changed)
        ttk.Checkbutton(frame, text="启用 densMAP（保持密度关系，计算更慢）", variable=self.state["densmap"], command=self._layout_param_changed).pack(anchor="w", pady=(3, 0))
        ttk.Button(frame, text="立即重新计算 UMAP", command=self._start_umap_now).pack(fill="x", pady=(6, 1))
        frame = ttk.LabelFrame(controls, text="显示范围与尺寸", padding=8); frame.pack(fill="x", pady=4)
        self._add_scale(frame, "轴向裁剪比例", self.state["trim"], 0.00, 0.20, 0.01)
        self._add_scale(frame, "点大小", self.state["point_size"], 0.5, 15.0, 0.5)
        self._add_scale(frame, "画布宽度", self.state["figure_width"], 10.0, 24.0, 0.01)
        self._add_scale(frame, "画布高度", self.state["figure_height"], 5.5, 14.0, 0.1)
        frame = ttk.LabelFrame(controls, text="透明度", padding=8); frame.pack(fill="x", pady=4)
        self._add_scale(frame, "类别图", self.state["class_alpha"], 0.10, 1.00, 0.05)
        self._add_scale(frame, "观测性质", self.state["observed_alpha"], 0.05, 1.00, 0.05)
        self._add_scale(frame, "预测性质", self.state["prediction_alpha"], 0.10, 1.00, 0.05)
        frame = ttk.LabelFrame(controls, text="性质图层", padding=8); frame.pack(fill="x", pady=4)
        for label, key in [("显示观测性质", "show_observed"), ("显示预测性质", "show_prediction"), ("显示未评分缺失值", "show_unscored")]:
            ttk.Checkbutton(frame, text=label, variable=self.state[key], command=self.schedule_render).pack(anchor="w")
        self._add_scale(frame, "色标下限", self.state["score_min"], 0.00, 0.90, 0.05)
        self._add_scale(frame, "色标上限", self.state["score_max"], 0.60, 1.00, 0.05)
        frame = ttk.LabelFrame(controls, text="类别显示与颜色", padding=8); frame.pack(fill="x", pady=4)
        for name in CLASS_ORDER:
            row = ttk.Frame(frame); row.pack(fill="x", pady=1)
            ttk.Checkbutton(row, text=name, variable=self.class_vars[name], command=self.schedule_render).pack(side="left", fill="x", expand=True)
            button = tk.Button(row, text=self.colors[name], width=9, relief="flat", bg=self.colors[name], command=lambda n=name: self.choose_color(n))
            button.pack(side="right"); setattr(self, f"color_button_{name}", button)
        frame = ttk.Frame(controls); frame.pack(fill="x", pady=(10, 4))
        ttk.Button(frame, text="恢复默认", command=self.reset).pack(fill="x", pady=2)
        ttk.Button(frame, text="保存当前 PNG + PDF", command=self.save_current).pack(fill="x", pady=2)
        ttk.Button(frame, text="保存参数 JSON", command=self.save_settings).pack(fill="x", pady=2)
        progress_frame = ttk.LabelFrame(controls, text="运行进度", padding=8)
        progress_frame.pack(fill="x", pady=(8, 4))
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            mode="determinate",
            maximum=100.0,
            variable=self.progress_value,
        )
        self.progress_bar.pack(fill="x", pady=(0, 5))
        ttk.Label(
            progress_frame,
            textvariable=self.progress_text,
            wraplength=350,
            justify="left",
        ).pack(anchor="w")
        ttk.Label(controls, textvariable=self.status, wraplength=360, justify="left", foreground="#4D6472").pack(anchor="w", pady=(8, 0))
        self.preview = ttk.Label(right, anchor="center"); self.preview.pack(fill="both", expand=True)

    def _add_scale(self, parent, title, variable, low, high, resolution):
        row = ttk.Frame(parent); row.pack(fill="x", pady=1)
        ttk.Label(row, text=title, width=13).pack(side="left")
        value = ttk.Label(row, text=f"{variable.get():.2f}", width=6, anchor="e"); value.pack(side="right")
        tk.Scale(row, variable=variable, from_=low, to=high, resolution=resolution, orient="horizontal", showvalue=False, length=190, command=lambda v: (value.configure(text=f"{float(v):.2f}"), self.schedule_render())).pack(side="left", fill="x", expand=True)

    def _add_layout_scale(self, parent, title, variable, low, high, resolution):
        row = ttk.Frame(parent); row.pack(fill="x", pady=1)
        ttk.Label(row, text=title, width=13).pack(side="left")
        value = ttk.Label(row, text=self._format_parameter(variable.get()), width=6, anchor="e")
        value.pack(side="right")
        tk.Scale(
            row, variable=variable, from_=low, to=high, resolution=resolution,
            orient="horizontal", showvalue=False, length=190,
            command=lambda v: (
                value.configure(text=self._format_parameter(v)),
                self._layout_param_changed(),
            ),
        ).pack(side="left", fill="x", expand=True)

    @staticmethod
    def _format_parameter(value):
        number = float(value)
        return f"{number:.0f}" if number >= 5 and number.is_integer() else f"{number:.2f}"

    def _add_combo(self, parent, title, variable, values, callback):
        row = ttk.Frame(parent); row.pack(fill="x", pady=2)
        ttk.Label(row, text=title, width=13).pack(side="left")
        combo = ttk.Combobox(row, textvariable=variable, state="readonly", width=27)
        combo["values"] = values
        combo.pack(side="left", fill="x", expand=True)
        combo.bind("<<ComboboxSelected>>", lambda event: callback())
        return combo

    def _layout_snapshot(self):
        return {
            "fingerprint": self.state["fingerprint"].get(),
            "metric": self.state["metric"].get(),
            "n_neighbors": int(self.state["n_neighbors"].get()),
            "min_dist": float(self.state["min_dist"].get()),
            "spread": float(self.state["spread"].get()),
            "init": self.state["init"].get(),
            "random_seed": int(self.state["random_seed"].get()),
            "set_op_mix_ratio": float(self.state["set_op_mix_ratio"].get()),
            "local_connectivity": float(self.state["local_connectivity"].get()),
            "repulsion_strength": float(self.state["repulsion_strength"].get()),
            "negative_sample_rate": int(self.state["negative_sample_rate"].get()),
            "learning_rate": float(self.state["learning_rate"].get()),
            "n_epochs": self.state["n_epochs"].get(),
            "densmap": bool(self.state["densmap"].get()),
        }

    @staticmethod
    def _layout_key(snapshot):
        raw = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]

    def _emit_progress(self, mode, message, percent=None):
        self.progress_queue.put(
            {"mode": mode, "message": message, "percent": percent}
        )

    def _poll_progress(self):
        try:
            while True:
                event = self.progress_queue.get_nowait()
                self.progress_base_message = event["message"]
                if event["mode"] == "indeterminate":
                    if not self.progress_is_indeterminate:
                        self.progress_bar.configure(mode="indeterminate")
                        self.progress_bar.start(12)
                        self.progress_is_indeterminate = True
                else:
                    if self.progress_is_indeterminate:
                        self.progress_bar.stop()
                        self.progress_bar.configure(mode="determinate")
                        self.progress_is_indeterminate = False
                    if event["percent"] is not None:
                        self.progress_value.set(float(event["percent"]))
        except queue.Empty:
            pass
        message = self.progress_base_message
        if self.layout_running and self.layout_started_at is not None:
            elapsed = max(0, int(time.monotonic() - self.layout_started_at))
            minutes, seconds = divmod(elapsed, 60)
            message = f"{message}\n已运行 {minutes:02d}:{seconds:02d}"
        self.progress_text.set(message)
        self.root.after(150, self._poll_progress)

    def _load_initial_layout(self):
        """Show the existing V11 layout immediately while keeping the UI responsive."""
        initial_path = self.coordinate_dir / "coordinates_maccs_k_200_min_dist_0_35_pca_init.csv"
        try:
            if initial_path.exists():
                self.table = load_layout(self.base_table, initial_path)
                self.active_layout_key = self._layout_key(self._layout_snapshot())
                self.status.set(f"已加载初始布局：{initial_path.name}")
            else:
                self.table = self.base_table.copy()
                self.active_layout_key = None
                self.status.set("未找到初始坐标，将在首次计算后显示 UMAP。")
            self.schedule_render()
        except Exception as exc:
            self.table = self.base_table.copy()
            self.active_layout_key = None
            self.status.set(f"初始布局加载失败：{exc}")
            self.schedule_render()

    def _layout_param_changed(self):
        """Debounce layout edits so dragging a slider starts only one computation."""
        if not self.controls_ready:
            return
        if self.layout_after is not None:
            self.root.after_cancel(self.layout_after)
        self.layout_after = self.root.after(500, self._request_umap)

    def _start_umap_now(self):
        if self.layout_after is not None:
            self.root.after_cancel(self.layout_after)
            self.layout_after = None
        self._request_umap()

    def _request_umap(self):
        self.layout_after = None
        snapshot = self._layout_snapshot()
        if snapshot["min_dist"] > snapshot["spread"]:
            self.status.set("参数无效：min_dist 必须小于或等于 spread。")
            self._emit_progress("determinate", "参数无效：请调整 min_dist 或 spread。", 0)
            return
        key = self._layout_key(snapshot)
        if key == getattr(self, "active_layout_key", None):
            self.schedule_render()
            return
        cache_path = self.layout_cache_dir / f"umap_{key}.npz"
        if cache_path.exists():
            try:
                cached = np.load(cache_path)
                if len(cached["umap_1"]) == len(self.base_table):
                    self.table = self.base_table.copy()
                    self.table["umap_1"] = cached["umap_1"]
                    self.table["umap_2"] = cached["umap_2"]
                    self.active_layout_key = key
                    self.status.set(f"已读取缓存布局：{key}")
                    self._emit_progress("determinate", f"已从缓存读取布局 {key}", 100)
                    self.schedule_render()
                    return
            except Exception:
                pass
        self.pending_layout = (snapshot, key)
        if self.layout_running:
            self.status.set("当前布局仍在计算；已记录最新参数，完成后将自动切换。")
            return
        self._launch_layout_worker(snapshot, key)

    def _launch_layout_worker(self, snapshot, key):
        self.layout_running = True
        self.layout_started_at = time.monotonic()
        self._emit_progress("indeterminate", "阶段 1/3：准备结构指纹")
        self.status.set(
            "正在后台重新计算 UMAP（33,142 个聚合物），窗口仍可操作...\n"
            f"指纹={snapshot['fingerprint']}，k={snapshot['n_neighbors']}，"
            f"min_dist={snapshot['min_dist']:.2f}，seed={snapshot['random_seed']}"
        )
        thread = threading.Thread(
            target=self._compute_layout_worker,
            args=(snapshot, key),
            daemon=True,
        )
        thread.start()

    def _compute_layout_worker(self, snapshot, key):
        try:
            fingerprint_name = snapshot["fingerprint"]
            if fingerprint_name not in self.fingerprint_cache:
                self._emit_progress(
                    "determinate",
                    f"阶段 1/3：生成 {fingerprint_name} 指纹 0/{len(self.base_table):,}",
                    0,
                )

                def report_fingerprints(completed, total):
                    self._emit_progress(
                        "determinate",
                        f"阶段 1/3：生成 {fingerprint_name} 指纹 {completed:,}/{total:,}",
                        completed / total * 100.0,
                    )

                matrix = fingerprint_matrix(
                    self.base_table["smiles"],
                    fingerprint_name,
                    progress_callback=report_fingerprints,
                )
                self.fingerprint_cache[fingerprint_name] = matrix
            else:
                self._emit_progress(
                    "determinate",
                    f"阶段 1/3：{fingerprint_name} 指纹已在内存缓存中",
                    100,
                )
            matrix = self.fingerprint_cache[fingerprint_name]
            metric_map = {
                "Jaccard / binary Tanimoto": "jaccard",
                "Dice": "dice",
                "Hamming": "hamming",
                "Cosine": "cosine",
            }
            n_epochs = None if snapshot["n_epochs"] == "auto" else int(snapshot["n_epochs"])
            reducer = umap.UMAP(
                n_neighbors=max(2, min(snapshot["n_neighbors"], len(matrix) - 1)),
                min_dist=snapshot["min_dist"],
                spread=snapshot["spread"],
                n_components=2,
                metric=metric_map[snapshot["metric"]],
                init=snapshot["init"],
                random_state=snapshot["random_seed"],
                set_op_mix_ratio=snapshot["set_op_mix_ratio"],
                local_connectivity=snapshot["local_connectivity"],
                repulsion_strength=snapshot["repulsion_strength"],
                negative_sample_rate=snapshot["negative_sample_rate"],
                learning_rate=snapshot["learning_rate"],
                n_epochs=n_epochs,
                densmap=snapshot["densmap"],
                n_jobs=1,
                low_memory=True,
                verbose=False,
            )
            self._emit_progress(
                "indeterminate",
                "阶段 2/3：构建近邻图并优化二维 UMAP（此阶段耗时最长）",
            )
            coordinates = reducer.fit_transform(matrix.astype(bool))
            self._emit_progress("determinate", "阶段 3/3：保存坐标并刷新图像", 96)
            np.savez_compressed(
                self.layout_cache_dir / f"umap_{key}.npz",
                umap_1=coordinates[:, 0], umap_2=coordinates[:, 1],
            )
            cache_csv = self.layout_cache_dir / f"umap_{key}.csv"
            pd.DataFrame({
                "polymer_id": self.base_table["polymer_id"].to_numpy(),
                "smiles": self.base_table["smiles"].to_numpy(),
                "umap_1": coordinates[:, 0], "umap_2": coordinates[:, 1],
            }).to_csv(cache_csv, index=False, encoding="utf-8-sig")
            self.root.after(0, lambda: self._layout_finished(snapshot, key, coordinates, None))
        except Exception as exc:
            self.root.after(
                0,
                lambda caught=exc: self._layout_finished(snapshot, key, None, caught),
            )

    def _layout_finished(self, snapshot, key, coordinates, error):
        self.layout_running = False
        self.layout_started_at = None
        current_key = self._layout_key(self._layout_snapshot())
        if error is not None:
            self.status.set(f"UMAP 计算失败：{type(error).__name__}: {error}")
            self._emit_progress("determinate", f"计算失败：{type(error).__name__}", 0)
            messagebox.showerror("modulus UMAP 调参器", f"UMAP 计算失败：\n{error}")
        elif key == current_key:
            self.table = self.base_table.copy()
            self.table["umap_1"] = coordinates[:, 0]
            self.table["umap_2"] = coordinates[:, 1]
            self.active_layout_key = key
            self.status.set(f"UMAP 已更新：{key}；可以继续调整参数。")
            self._emit_progress("determinate", f"计算完成：布局 {key}", 100)
            self.schedule_render()
        else:
            self.status.set("旧参数的计算已完成，正在启动最新参数布局...")
            self._emit_progress("indeterminate", "参数已变化：准备计算最新布局")
            self._request_umap()

    def choose_color(self, name):
        selected = colorchooser.askcolor(color=self.colors[name], title=f"选择 {name} 颜色")
        if selected and selected[1]:
            self.colors[name] = selected[1].upper()
            button = getattr(self, f"color_button_{name}"); button.configure(text=self.colors[name], bg=self.colors[name], activebackground=self.colors[name])
            self.schedule_render()

    def schedule_render(self):
        if self.render_after is not None:
            self.root.after_cancel(self.render_after)
        self.render_after = self.root.after(180, self.render_preview)

    def render_preview(self):
        if self.rendering: return
        self.rendering = True
        try:
            fig = draw_figure(self.table, self.state, for_preview=True)
            self.photo = figure_to_photoimage(fig)
            self.preview.configure(image=self.photo)
            fig.clear()
            self.status.set(f"预览已更新：{len(self.table):,} 个聚合物；使用逐聚合物 modulus 可靠性分数。")
        except Exception as exc:
            self.status.set(f"绘图失败：{exc}")
        finally:
            self.rendering = False

    def reset(self):
        defaults = {
            "fingerprint": "MACCS 167-bit", "metric": "Jaccard / binary Tanimoto",
            "n_neighbors": 200, "min_dist": 0.35, "spread": 1.00, "init": "pca",
            "random_seed": 42, "set_op_mix_ratio": 1.00, "local_connectivity": 1.00,
            "repulsion_strength": 1.00, "negative_sample_rate": 5,
            "learning_rate": 1.00, "n_epochs": "auto", "densmap": False,
            "trim": 0.05, "point_size": 6.25, "class_alpha": 0.62,
            "observed_alpha": 0.32, "prediction_alpha": 0.70, "score_min": 0.50,
            "score_max": 1.00, "figure_width": DEFAULT_FIGURE_WIDTH, "figure_height": DEFAULT_FIGURE_HEIGHT,
        }
        for key, value in defaults.items(): self.state[key].set(value)
        for name in CLASS_ORDER: self.class_vars[name].set(name != "Mixed/unclear")
        self.colors.update(DEFAULT_CLASS_COLORS)
        for name in CLASS_ORDER:
            button = getattr(self, f"color_button_{name}"); button.configure(text=self.colors[name], bg=self.colors[name], activebackground=self.colors[name])
        self._load_initial_layout()

    def current_settings(self):
        layout = self._layout_snapshot()
        return {
            "layout_parameters": layout,
            "active_layout_cache_key": getattr(self, "active_layout_key", None),
            "trim": self.state["trim"].get(),
            "point_size": self.state["point_size"].get(),
            "class_alpha": self.state["class_alpha"].get(),
            "observed_alpha": self.state["observed_alpha"].get(),
            "prediction_alpha": self.state["prediction_alpha"].get(),
            "score_min": self.state["score_min"].get(),
            "score_max": self.state["score_max"].get(),
            "figure_width": self.state["figure_width"].get(),
            "figure_height": self.state["figure_height"].get(),
            "show_observed": self.state["show_observed"].get(),
            "show_prediction": self.state["show_prediction"].get(),
            "show_unscored": self.state["show_unscored"].get(),
            "classes": {name: self.class_vars[name].get() for name in CLASS_ORDER},
            "colors": self.colors,
            "source_assignment": str(DEFAULT_ASSIGNMENTS),
        }

    def save_settings(self):
        path = self.output_dir / "flexural_storage_GPa_umap_tuner_settings.json"
        path.write_text(json.dumps(self.current_settings(), ensure_ascii=False, indent=2), encoding="utf-8")
        self.status.set(f"参数已保存：{path}")

    def save_current(self):
        desired_key = self._layout_key(self._layout_snapshot())
        if self.layout_running or desired_key != getattr(self, "active_layout_key", None):
            messagebox.showwarning(
                "modulus UMAP 调参器",
                "当前参数对应的 UMAP 仍未完成。请等待状态栏显示 UMAP 已更新后再保存。",
            )
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        fig = draw_figure(self.table, self.state, for_preview=False)
        png = self.output_dir / "flexural_storage_GPa_umap_tuned.png"
        pdf = self.output_dir / "flexural_storage_GPa_umap_tuned.pdf"
        save_figure(fig, png, pdf)
        fig.clear(); self.save_settings()
        self.status.set(f"图片已保存：\n{png}\n{pdf}")


def main():
    parser = argparse.ArgumentParser(description="modulus UMAP 本地 Tkinter 调参器")
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--reliability", type=Path, default=DEFAULT_RELIABILITY)
    parser.add_argument("--coordinate-dir", type=Path, default=DEFAULT_COORDINATE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    RDLogger.DisableLog("rdApp.*")
    table = read_assignment_table(
        args.assignments.resolve(),
        args.dataset.resolve(),
        args.reliability.resolve(),
    )
    root = tk.Tk()
    FlexuralStorageUmapTuner(
        root,
        table,
        args.coordinate_dir.resolve(),
        args.output_dir.resolve(),
    )
    root.mainloop()


if __name__ == "__main__":
    main()
