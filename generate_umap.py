"""
generate_umap.py — TrendLens Pipeline Step 3
=============================================
Reduces 512-d CLIP embeddings to:
  • 2D  (umap_2d.npy)   — scatter-plot visualisation
  • 10D (umap_10d.npy)  — HDBSCAN clustering input

Run after generate_embeddings.py, before HDBSCAN clustering.

Usage:
    python generate_umap.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                    # headless-safe backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from umap import UMAP

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUTS_DIR     = os.path.join(os.path.dirname(__file__), "trendlens_outputs")
EMBEDDINGS_PATH = os.path.join(OUTPUTS_DIR, "embeddings.npy")
METADATA_PATH   = os.path.join(OUTPUTS_DIR, "metadata.csv")
OUT_2D_PATH     = os.path.join(OUTPUTS_DIR, "umap_2d.npy")
OUT_10D_PATH    = os.path.join(OUTPUTS_DIR, "umap_10d.npy")
SCATTER_PATH    = os.path.join(OUTPUTS_DIR, "umap_scatter.png")

# ── UMAP hyperparameters ───────────────────────────────────────────────────────
UMAP_COMMON = dict(
    n_neighbors  = 30,
    metric       = "cosine",
    random_state = 42,
    low_memory   = False,
    verbose      = True,
)

UMAP_2D_PARAMS  = {**UMAP_COMMON, "n_components": 2,  "min_dist": 0.1}
UMAP_10D_PARAMS = {**UMAP_COMMON, "n_components": 10, "min_dist": 0.0}


# ── Load & verify embeddings ────────────────────────────────────────────────────────────

def load_and_verify() -> tuple[np.ndarray, pd.DataFrame]:
    print("\n" + "=" * 60)
    print("STEP 1 — Loading embeddings & metadata")
    print("=" * 60)

    if not os.path.exists(EMBEDDINGS_PATH):
        sys.exit(f"[ERROR] embeddings.npy not found at: {EMBEDDINGS_PATH}")
    if not os.path.exists(METADATA_PATH):
        sys.exit(f"[ERROR] metadata.csv not found at: {METADATA_PATH}")

    embs = np.load(EMBEDDINGS_PATH)
    meta = pd.read_csv(METADATA_PATH)

    print(f"  embeddings shape : {embs.shape}")
    print(f"  embeddings dtype : {embs.dtype}")
    print(f"  metadata rows    : {len(meta)}")

    # Assertions
    assert embs.ndim == 2,          "Expected 2-D array"
    assert embs.shape[1] == 512,    f"Expected 512 dims, got {embs.shape[1]}"
    assert embs.dtype == np.float32, f"Expected float32, got {embs.dtype}"
    assert embs.shape[0] == len(meta), (
        f"Row-count mismatch: embeddings {embs.shape[0]} vs metadata {len(meta)}"
    )
    print("  ✓ All assertions passed")
    return embs, meta


# ── UMAP reduction helpers ────────────────────────────────────────────────────────────

def run_umap(embs: np.ndarray, params: dict, label: str) -> np.ndarray:
    print(f"\n{'=' * 60}")
    print(f"STEP — Running UMAP ({label})")
    print(f"  params: {params}")
    print(f"{'=' * 60}")

    reducer = UMAP(**params)
    result  = reducer.fit_transform(embs).astype(np.float32)

    print(f"  output shape : {result.shape}")
    print(f"  output dtype : {result.dtype}")
    print(f"  min / max    : {result.min():.4f} / {result.max():.4f}")
    return result


# ── Scatter-plot coloured by category ────────────────────────────────────────────────────────────

def save_scatter(coords_2d: np.ndarray, meta: pd.DataFrame) -> None:
    print(f"\n{'=' * 60}")
    print("STEP — Saving umap_scatter.png")
    print(f"{'=' * 60}")

    categories  = sorted(meta["category"].unique())
    n_cats      = len(categories)
    cmap        = plt.get_cmap("tab20")
    cat_to_idx  = {c: i for i, c in enumerate(categories)}
    colours     = [cmap(cat_to_idx[c] / max(n_cats - 1, 1)) for c in meta["category"]]

    fig, ax = plt.subplots(figsize=(14, 10), dpi=150)
    ax.set_facecolor("#0f0f12")
    fig.patch.set_facecolor("#0f0f12")

    # Draw points in random order to avoid occlusion bias
    rng  = np.random.default_rng(42)
    idx  = rng.permutation(len(coords_2d))
    xs   = coords_2d[idx, 0]
    ys   = coords_2d[idx, 1]
    cols = [colours[i] for i in idx]

    ax.scatter(xs, ys, c=cols, s=1.5, alpha=0.55, linewidths=0, rasterized=True)

    # Legend
    handles = [
        mpatches.Patch(color=cmap(cat_to_idx[c] / max(n_cats - 1, 1)), label=c)
        for c in categories
    ]
    legend = ax.legend(
        handles         = handles,
        title           = "Category",
        fontsize        = 7,
        title_fontsize  = 8,
        loc             = "upper left",
        framealpha      = 0.25,
        labelcolor      = "white",
    )
    legend.get_title().set_color("white")

    ax.set_title(
        "TrendLens — UMAP 2D projection of CLIP embeddings (69 226 images)",
        color    = "white",
        fontsize = 13,
        pad      = 12,
    )
    ax.set_xlabel("UMAP-1", color="#aaaaaa", fontsize=9)
    ax.set_ylabel("UMAP-2", color="#aaaaaa", fontsize=9)
    ax.tick_params(colors="#666666", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    plt.tight_layout()
    plt.savefig(SCATTER_PATH, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved → {SCATTER_PATH}")


# ── Summary printout ────────────────────────────────────────────────────────────

def print_summary(emb2d: np.ndarray, emb10d: np.ndarray, meta: pd.DataFrame) -> None:
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    for name, arr in [("umap_2d",  emb2d), ("umap_10d", emb10d)]:
        print(f"\n  {name}:")
        print(f"    shape  = {arr.shape}")
        print(f"    dtype  = {arr.dtype}")
        print(f"    min    = {arr.min():.6f}")
        print(f"    max    = {arr.max():.6f}")

    # Row-count assertions against metadata
    assert emb2d.shape[0]  == len(meta), "umap_2d row count mismatch with metadata.csv"
    assert emb10d.shape[0] == len(meta), "umap_10d row count mismatch with metadata.csv"
    print(f"\n  ✓ Both outputs align with metadata.csv ({len(meta)} rows)")
    print(f"\n  Output files written to: {OUTPUTS_DIR}")
    print(f"    • umap_2d.npy      {emb2d.shape}  float32")
    print(f"    • umap_10d.npy     {emb10d.shape} float32")
    print(f"    • umap_scatter.png")
    print(f"\n{'=' * 60}")
    print("  generate_umap.py — DONE  ✓")
    print(f"{'=' * 60}\n")


# ── Main ────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # 1. Load
    embs, meta = load_and_verify()

    # 2. UMAP 2D — for visualisation
    emb2d = run_umap(embs, UMAP_2D_PARAMS,  label="2D / visualisation")
    np.save(OUT_2D_PATH, emb2d)
    print(f"  Saved → {OUT_2D_PATH}")

    # 3. UMAP 10D — for HDBSCAN
    emb10d = run_umap(embs, UMAP_10D_PARAMS, label="10D / HDBSCAN input")
    np.save(OUT_10D_PATH, emb10d)
    print(f"  Saved → {OUT_10D_PATH}")

    # 4. Scatter plot
    save_scatter(emb2d, meta)

    # 5. Summary + assertions
    print_summary(emb2d, emb10d, meta)


if __name__ == "__main__":
    main()
