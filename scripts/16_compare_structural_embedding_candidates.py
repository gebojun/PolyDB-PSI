"""Compare unsupervised structural embeddings for the B/C knowledge-gap map.

Class labels are used only after embedding to quantify visual neighborhood
coherence. They are never passed to UMAP.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs" / "13_polymer_class_knowledge_gap_umap" / "polymer_class_and_knowledge_gap_assignments.csv.gz"
CURRENT_COORDS = ROOT / "outputs" / "13_polymer_class_knowledge_gap_umap" / "shared_maccs_jaccard_umap_coordinates.csv"
OUT = ROOT / "outputs" / "14_structural_embedding_comparison"

DISPLAY_ORDER = [
    "Polyimide", "Polyurethane", "Polyamide", "Polyester", "Polycarbonate",
    "Polyether", "Polysulfone", "Polysiloxane", "Polyolefin",
    "Fluoropolymer", "Polyacrylate", "Polystyrene", "Mixed/unclear",
]
COLORS = {
    "Polyimide": "#7B3FA1", "Polyurethane": "#F06B21", "Polyamide": "#1976C9",
    "Polyester": "#63A735", "Polycarbonate": "#E9B72F", "Polyether": "#A65C2A",
    "Polysulfone": "#E86BA5", "Polysiloxane": "#18A7A8", "Polyolefin": "#7D8727",
    "Fluoropolymer": "#9063CD", "Polyacrylate": "#5D82A8", "Polystyrene": "#C58B00",
    "Mixed/unclear": "#A9AAAC",
}


def normalize_smiles(value: str) -> str:
    return re.sub(r"\[R\d*\]", "[*]", value)


def fingerprints(smiles: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    maccs = np.zeros((len(smiles), 167), dtype=np.uint8)
    morgan = np.zeros((len(smiles), 1024), dtype=np.uint8)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    for row, value in enumerate(smiles.astype(str)):
        mol = Chem.MolFromSmiles(normalize_smiles(value))
        if mol is None:
            raise ValueError(f"Invalid structure at row {row}: {value}")
        DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), maccs[row])
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), morgan[row])
    return maccs, morgan


def embed(
    matrix: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    init: str,
    spread: float = 1.0,
) -> np.ndarray:
    return umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        spread=spread,
        n_components=2,
        metric="jaccard",
        init=init,
        random_state=42,
        n_jobs=1,
        low_memory=True,
        verbose=True,
    ).fit_transform(matrix.astype(bool))


def metrics(coords: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    idx = NearestNeighbors(n_neighbors=11).fit(coords).kneighbors(return_distance=False)[:, 1:]
    purity_rows = (labels[idx] == labels[:, None]).mean(axis=1)
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(coords), size=min(6000, len(coords)), replace=False)
    return {
        "mean_10nn_same_class": float(purity_rows.mean()),
        "median_10nn_same_class": float(np.median(purity_rows)),
        "fraction_all_10_neighbors_same_class": float(np.mean(purity_rows == 1.0)),
        "silhouette_6000": float(silhouette_score(coords[sample_idx], labels[sample_idx])),
    }


def draw(candidates: dict[str, np.ndarray], frame: pd.DataFrame) -> None:
    plt.rcParams.update({"font.family": "Arial", "font.sans-serif": ["Arial", "DejaVu Sans"]})
    fig, axes = plt.subplots(2, 3, figsize=(17.8, 11.0), dpi=220)
    for letter, (title, coords), ax in zip("ABCD", candidates.items(), axes.ravel()):
        for name in reversed(DISPLAY_ORDER):
            mask = frame["display_class"].eq(name).to_numpy()
            ax.scatter(
                coords[mask, 0], coords[mask, 1], s=2.1, c=COLORS[name],
                alpha=0.62 if name != "Mixed/unclear" else 0.25,
                linewidths=0, rasterized=True,
            )
        ax.set_title(f"{letter}. {title}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout(h_pad=2.4, w_pad=1.8)
    fig.savefig(OUT / "structural_embedding_candidates.png", dpi=350, bbox_inches="tight")
    fig.savefig(OUT / "structural_embedding_candidates.pdf", dpi=350, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    RDLogger.DisableLog("rdApp.*")
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(INPUT, usecols=["polymer_id", "smiles", "display_class"])
    old = pd.read_csv(CURRENT_COORDS)
    if not old["smiles"].astype(str).equals(frame["smiles"].astype(str)):
        raise ValueError("Current coordinates do not match the assignment table")

    print("Computing MACCS and Morgan fingerprints...", flush=True)
    maccs, morgan = fingerprints(frame["smiles"])
    candidates = {
        "MACCS, k=25, min_dist=0.12 (current)": old[["umap_1", "umap_2"]].to_numpy(),
        "MACCS, k=25, min_dist=0.08": embed(maccs, 25, 0.08, "spectral"),
        "MACCS, k=25, min_dist=0.08, spread=0.5": embed(
            maccs, 25, 0.08, "spectral", spread=0.5
        ),
        "MACCS, k=100, min_dist=0.25": embed(maccs, 100, 0.25, "spectral"),
        "MACCS, k=200, min_dist=0.35, PCA init": embed(maccs, 200, 0.35, "pca"),
        "Morgan-1024, k=100, min_dist=0.25": embed(morgan, 100, 0.25, "spectral"),
    }

    labels = frame["display_class"].to_numpy()
    rows = []
    for name, coords in candidates.items():
        values = metrics(coords, labels)
        rows.append({"embedding": name, **values})
        stem = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        pd.DataFrame(
            {
                "polymer_id": frame["polymer_id"],
                "smiles": frame["smiles"],
                "dimension_1": coords[:, 0],
                "dimension_2": coords[:, 1],
            }
        ).to_csv(OUT / f"coordinates_{stem}.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(rows).sort_values("mean_10nn_same_class", ascending=False)
    summary.to_csv(OUT / "embedding_metrics.csv", index=False)
    draw(candidates, frame)
    print(json.dumps(summary.to_dict(orient="records"), indent=2), flush=True)


if __name__ == "__main__":
    main()
