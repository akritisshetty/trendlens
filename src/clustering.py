"""
clustering.py
-------------
Dimensionality reduction + HDBSCAN clustering + representative images.

Implements the Stage 5–7 requirements:
  * no hard-coded embedding dimensionality (inspect the actual shape)
  * compare HDBSCAN on raw embeddings vs on a reduced representation
  * configurable HDBSCAN parameters + parameter sweep experiments
  * report clusters / noise / cluster sizes / persistence / silhouette
  * representative images per cluster (membership prob + medoid distance)
  * contact sheets (PNG grids) per cluster
  * deterministic + cacheable (UMAP/PCA outputs cached to .npy)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import config


# ──────────────────────────────────────────────────────────────────────────
# Dimensionality reduction
# ──────────────────────────────────────────────────────────────────────────
def embedding_fingerprint(embeddings: np.ndarray) -> str:
    """
    Cheap content hash of an embedding matrix, used to key caches.

    Row count alone is NOT a sufficient cache key. Every encoder in
    ``scripts/select_models.py`` produces the same number of rows for the same
    corpus, so a cache keyed on ``(rows, n_components)`` silently returns the
    previous checkpoint's UMAP projection when the encoder changes — and
    HDBSCAN then clusters 768-d image vectors against a projection built from
    512-d ones. Hashing the bytes makes that a cache miss instead.
    """
    import hashlib

    arr = np.ascontiguousarray(np.asarray(embeddings, dtype="float32"))
    h = hashlib.sha256()
    h.update(str(arr.shape).encode())
    # Sample rather than hash every element: this is a cache key, not a
    # correctness check, and full hashing of a large matrix costs more than the
    # UMAP fit it would save.
    flat = arr.reshape(-1)
    step = max(1, flat.size // 4096)
    h.update(flat[::step].tobytes())
    h.update(str(float(flat[0])).encode())
    h.update(str(float(flat[-1])).encode())
    return h.hexdigest()[:16]


def reduce_dimensions(
    embeddings: np.ndarray,
    method: str = "umap",
    n_components: int = 10,
    seed: int | None = None,
    cache_path: Path | None = None,
    force: bool = False,
) -> np.ndarray:
    """
    Project ``embeddings`` (N, D) to (N, n_components) float32.

    * ``method="pca"``  → sklearn PCA (deterministic)
    * ``method="umap"`` → umap-learn, seeded for reproducibility

    Results are cached to ``cache_path`` (or ``artifacts/embeddings/``) so the
    expensive UMAP step runs once per config. The default cache filename embeds
    a fingerprint of the input matrix, so changing the encoder or the corpus
    invalidates it automatically.
    """
    emb = np.asarray(embeddings, dtype="float32")
    if cache_path is None:
        fp = embedding_fingerprint(emb)
        cache_path = (
            config.ARTIFACTS_DIR
            / "embeddings"
            / f"{method}_{n_components}d_s{seed if seed is not None else config.RANDOM_SEED}_{fp}.npy"
        )
    cache_path = Path(cache_path)

    if not force and cache_path.exists():
        cached = np.load(cache_path)
        if cached.shape == (emb.shape[0], n_components):
            print(f"Loading cached {method} ({n_components}d): {cache_path.name}")
            return cached.astype("float32")
        print(f"Cache shape mismatch ({cached.shape}), recomputing.")

    seed = int(seed) if seed is not None else int(config.RANDOM_SEED)
    t0 = time.time()

    if method == "pca":
        from sklearn.decomposition import PCA

        reduced = PCA(n_components=n_components, random_state=seed).fit_transform(emb)
    elif method == "umap":
        import umap

        reducer = umap.UMAP(
            n_components=n_components,
            random_state=seed,
            n_neighbors=15,
            min_dist=0.0,
            metric="cosine",
            verbose=False,
        )
        reduced = reducer.fit_transform(emb)
    else:
        raise ValueError(f"Unknown reduction method: {method!r}")

    reduced = np.asarray(reduced, dtype="float32")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, reduced)
    print(f"{method} {n_components}d: {emb.shape[1]} -> {reduced.shape[1]} "
          f"({time.time() - t0:.1f}s) -> {cache_path.name}")
    return reduced


# ──────────────────────────────────────────────────────────────────────────
# HDBSCAN
# ──────────────────────────────────────────────────────────────────────────
DEFAULT_HDBSCAN_PARAMS = dict(
    min_cluster_size=50,
    min_samples=10,
    cluster_selection_method="eom",
)


def run_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int | None = None,
    min_samples: int | None = None,
    cluster_selection_method: str | None = None,
    save_model: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, Any]:
    """
    Run HDBSCAN. Returns (labels, probabilities, clusterer).

    Noise points get label ``-1`` and are never forced into clusters.
    """
    from hdbscan import HDBSCAN

    params = dict(DEFAULT_HDBSCAN_PARAMS)
    if min_cluster_size is not None:
        params["min_cluster_size"] = int(min_cluster_size)
    if min_samples is not None:
        params["min_samples"] = int(min_samples)
    if cluster_selection_method is not None:
        params["cluster_selection_method"] = cluster_selection_method

    clusterer = HDBSCAN(
        **params,
        prediction_data=True,  # for future consistency of new points
        metric="euclidean",
    )
    t0 = time.time()
    clusterer.fit(np.asarray(embeddings))
    labels = np.asarray(clusterer.labels_, dtype=np.int64)
    probabilities = np.asarray(clusterer.probabilities_, dtype="float32")
    print(
        f"HDBSCAN(min_cluster_size={params['min_cluster_size']}, "
        f"min_samples={params['min_samples']}, "
        f"method={params['cluster_selection_method']}) "
        f"in {time.time() - t0:.1f}s: "
        f"{labels.max() + 1 if labels.max() >= 0 else 0} clusters, "
        f"{int((labels == -1).sum())} noise"
    )

    if save_model is not None:
        import pickle

        save_model = Path(save_model)
        save_model.parent.mkdir(parents=True, exist_ok=True)
        with open(save_model, "wb") as fh:
            pickle.dump({"labels": labels, "params": params}, fh)
        print(f"Saved model manifest -> {save_model}")
    return labels, probabilities, clusterer


# ──────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────
def cluster_report(
    embeddings: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    clusterer: Any = None,
    silhouette_sample: int | None = 1000,
) -> dict[str, Any]:
    """
    Build + print a cluster report: counts, noise %, cluster sizes,
    persistence, silhouette score (only where meaningful).
    """
    labels = np.asarray(labels)
    n = len(labels)
    cluster_ids = np.unique(labels[labels >= 0])
    n_clusters = len(cluster_ids)
    noise = int((labels == -1).sum())

    stats: dict[str, Any] = {
        "n_points": n,
        "n_clusters": n_clusters,
        "n_noise": noise,
        "noise_pct": float(noise / n),
        "clustered_pct": float((n - noise) / n),
        "cluster_sizes": {
            int(c): int((labels == c).sum()) for c in sorted(cluster_ids)
        },
        "cluster_probability_mean": {
            int(c): float(probabilities[labels == c].mean())
            if (labels == c).any()
            else None
            for c in sorted(cluster_ids)
        },
    }

    if clusterer is not None:
        persistence = getattr(clusterer, "cluster_persistence_", None)
        if persistence is not None:
            stats["cluster_persistence"] = {
                int(c): float(persistence[c])
                for c in sorted(cluster_ids)
                if int(c) < len(persistence)
            }

    stats["silhouette"] = _silhouette_score(embeddings, labels, sample=silhouette_sample)

    # ---- print ----
    print("=" * 60)
    print("CLUSTER REPORT")
    print("=" * 60)
    print(f"points      : {n:,}")
    print(f"clusters    : {n_clusters}")
    print(f"noise       : {noise:,} ({stats['noise_pct']:.1%})")
    print(f"clustered   : {stats['clustered_pct']:.1%}")
    sizes = sorted(stats["cluster_sizes"].items(), key=lambda kv: -kv[1])
    print("cluster sizes (desc):", ", ".join(f"c{c}={s}" for c, s in sizes))
    if stats.get("cluster_persistence"):
        pers = sorted(stats["cluster_persistence"].items(), key=lambda kv: -kv[1])
        print("persistence (top):",
              ", ".join(f"c{c}={p:.3f}" for c, p in pers[:10]))
    sil = stats["silhouette"]
    print(f"silhouette  : {sil if sil is not None else 'N/A (need 2+ clusters, excl. noise)'}")
    print("=" * 60)
    return stats


def _silhouette_score(
    embeddings: np.ndarray, labels: np.ndarray, sample: int | None = 1000
) -> float | None:
    from sklearn.metrics import silhouette_score

    mask = labels >= 0
    uniq = np.unique(labels[mask])
    if len(uniq) < 2:
        return None
    if mask.sum() > (sample or np.inf):
        rng = np.random.default_rng(config.RANDOM_SEED)
        idx = np.where(mask)[0]
        pick = rng.choice(idx, size=sample, replace=False)
    else:
        pick = np.where(mask)[0]
    if len(np.unique(labels[pick])) < 2:
        return None
    return float(silhouette_score(embeddings[pick], labels[pick], metric="euclidean"))


def cluster_summary(
    df: pd.DataFrame,
    labels: np.ndarray,
    probabilities: np.ndarray,
    out_path: Path | None = None,
) -> pd.DataFrame:
    """
    Per-cluster engagement/size summary tied to the metadata frame.

    ``out_path`` defaults to the real artifact location. Tests MUST pass a
    tmp path so unit fixtures never overwrite the real pipeline artifacts.
    """
    rows = []
    for c in sorted(np.unique(labels[labels >= 0])):
        m = labels == c
        sub = df[m]
        eng = sub["likes"].fillna(0) + sub["comments"].fillna(0)
        rows.append(
            {
                "cluster_id": int(c),
                "size": int(m.sum()),
                "percentage": float(m.mean()),
                "mean_probability": float(probabilities[m].mean()),
                "median_engagement": float(eng.median()),
                "mean_engagement": float(eng.mean()),
            }
        )
    out = pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)
    out_path = Path(out_path) if out_path else (
        config.ARTIFACTS_DIR / "cluster_metadata" / "cluster_summary.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out


def parameter_sweep(
    embeddings: np.ndarray,
    min_cluster_sizes: list[int] | None = None,
    min_samples_list: list[int] | None = None,
    methods: list[str] | None = None,
    silhouette_sample: int | None = 800,
) -> pd.DataFrame:
    """
    Grid search over HDBSCAN parameters. Returns a comparison DataFrame.
    """
    rows = []
    for mcs in min_cluster_sizes or [50, 100, 200]:
        for ms in min_samples_list or [10, 25]:
            for method in methods or ["eom"]:
                labels, probs, clusterer = run_hdbscan(
                    embeddings,
                    min_cluster_size=mcs,
                    min_samples=ms,
                    cluster_selection_method=method,
                )
                n_clusters = int(labels.max() + 1) if labels.max() >= 0 else 0
                sil = _silhouette_score(embeddings, labels, sample=silhouette_sample)
                rows.append(
                    {
                        "min_cluster_size": mcs,
                        "min_samples": ms,
                        "method": method,
                        "n_clusters": n_clusters,
                        "noise_pct": float((labels == -1).mean()),
                        "clustered_pct": float((labels >= 0).mean()),
                        "silhouette": sil,
                    }
                )
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────
# Representative images
# ──────────────────────────────────────────────────────────────────────────
def representative_images(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    k: int = 9,
) -> dict[int, list[dict[str, Any]]]:
    """
    For each cluster pick ``k`` representative images.

    Ranking: membership probability (HDBSCAN) first, then distance to the
    cluster medoid, then row index (deterministic tie-break).
    """
    reps: dict[int, list[dict[str, Any]]] = {}
    emb = np.asarray(embeddings)
    for c in sorted(np.unique(labels[labels >= 0])):
        idx = np.where(labels == c)[0]
        sub = emb[idx]
        medoid = sub.mean(axis=0)
        medoid = medoid / (np.linalg.norm(medoid) + 1e-9)
        # cosine distance to medoid (embeddings are L2-normalised)
        dists = 1.0 - (sub @ medoid)
        prob = probabilities[idx]
        order = np.lexsort((idx, dists, -prob))[:k]  # prob desc, dist asc, idx asc
        rows = []
        for pos in order:
            i = int(idx[pos])
            rows.append(
                {
                    "row_index": i,
                    "post_id": str(df["post_id"].iloc[i]),
                    "image_path": str(df["image_path"].iloc[i]),
                    "probability": float(prob[pos]),
                    "medoid_distance": float(dists[pos]),
                }
            )
        reps[int(c)] = rows
    return reps


def save_representatives(reps: dict[int, list[dict[str, Any]]], path: Path | None = None) -> Path:
    path = Path(path) if path else config.ARTIFACTS_DIR / "cluster_metadata" / "representatives.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reps, indent=1))
    return path


def contact_sheets(
    reps: dict[int, list[dict[str, Any]]],
    image_root: Path | None = None,
    out_dir: Path | None = None,
    grid: tuple[int, int] = (3, 3),
    size: tuple[int, int] = (256, 256),
) -> dict[int, Path]:
    """
    Build a PNG contact sheet per cluster and return {cluster_id: path}.
    Images that fail to load become a labelled placeholder tile.
    """
    from src import preprocessing as pp
    from PIL import Image, ImageDraw

    image_root = image_root or config.IMAGE_ROOT
    out_dir = Path(out_dir) if out_dir else config.FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[int, Path] = {}
    for c, rows in tqdm(reps.items(), desc="Contact sheets"):
        cols, rows_n = grid
        sheet = Image.new("RGB", (cols * size[0], rows_n * size[1]), (245, 245, 245))
        draw = ImageDraw.Draw(sheet)
        for pos, r in enumerate(rows[: cols * rows_n]):
            x = (pos % cols) * size[0]
            y = (pos // cols) * size[1]
            try:
                img = pp.load_image(r["image_path"], image_root, resize=size)
                sheet.paste(img, (x, y))
            except (OSError, ValueError):
                draw.rectangle([x, y, x + size[0] - 1, y + size[1] - 1], outline=(200, 0, 0), width=2)
                draw.text((x + 4, y + 4), "load failed", fill=(200, 0, 0))
        draw.text((4, sheet.height - 14), f"cluster {c} — synthetic demo", fill=(80, 80, 80))
        path = out_dir / f"cluster_{int(c):03d}.jpg"
        sheet.save(path, quality=90)
        results[int(c)] = path
    return results


# ──────────────────────────────────────────────────────────────────────────
# Driver helpers
# ──────────────────────────────────────────────────────────────────────────
def load_embeddings(path: Path | None = None) -> np.ndarray:
    path = Path(path) if path else config.EMBEDDINGS_DIR / "embeddings.npy"
    return np.load(path)


def load_aligned_metadata(path: Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else config.EMBEDDINGS_DIR / "metadata.parquet"
    return pd.read_parquet(path)


# ──────────────────────────────────────────────────────────────────────────
# CLI driver
# ──────────────────────────────────────────────────────────────────────────
def run_pipeline(
    min_cluster_size: int = 50,
    min_samples: int = 10,
    method: str = "eom",
    k_reps: int = 9,
    force: bool = False,
) -> dict[str, Any]:
    """
    Phase 3 driver over the current embedding artifact.

    Runs HDBSCAN on (A) raw embeddings and (B) UMAP-10d, reports both,
    saves all artifacts, and generates representative contact sheets for B.
    """
    import pickle

    emb = load_embeddings()
    meta = load_aligned_metadata()
    assert len(emb) == len(meta), "embeddings/metadata misaligned"

    results: dict[str, Any] = {"n_points": int(len(emb)), "dim": int(emb.shape[1])}

    # ---- A: HDBSCAN on raw embeddings ----
    labels_a, probs_a, cl_a = run_hdbscan(
        emb,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=method,
        save_model=config.CLUSTER_MODELS_DIR / "hdbscan_raw.pkl",
    )
    results["A_raw"] = cluster_report(emb, labels_a, probs_a, cl_a)
    np.save(config.CLUSTER_MODELS_DIR / "labels_raw.npy", labels_a)
    np.save(config.CLUSTER_MODELS_DIR / "probabilities_raw.npy", probs_a)

    # ---- B: UMAP-10d + HDBSCAN ----
    reduced = reduce_dimensions(
        emb, method="umap", n_components=10, force=force
    )
    labels_b, probs_b, cl_b = run_hdbscan(
        reduced,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=method,
        save_model=config.CLUSTER_MODELS_DIR / "hdbscan_umap10.pkl",
    )
    results["B_umap10"] = cluster_report(reduced, labels_b, probs_b, cl_b)
    np.save(config.CLUSTER_MODELS_DIR / "labels_umap10.npy", labels_b)
    np.save(config.CLUSTER_MODELS_DIR / "probabilities_umap10.npy", probs_b)

    # ---- choose config B for downstream (report both honestly) ----
    chosen = ("B_umap10", labels_b, probs_b)
    summary = cluster_summary(meta, chosen[1], chosen[2])
    results["chosen"] = "B_umap10"
    results["cluster_summary"] = summary.to_dict("records")

    reps = representative_images(meta, emb, labels_b, probs_b, k=k_reps)
    save_representatives(reps)
    sheets = contact_sheets(reps)
    results["contact_sheets"] = {str(k): str(v) for k, v in sheets.items()}

    # ---- experiment manifest (reproducibility) ----
    manifest = {
        "experiment": "phase3_clustering",
        "config": config.experiment_config(
            {
                "hdbscan_min_cluster_size": min_cluster_size,
                "hdbscan_min_samples": min_samples,
                "hdbscan_method": method,
                "umap_components": 10,
                "k_representatives": k_reps,
                "embeddings": str(config.EMBEDDINGS_DIR / "embeddings.npy"),
            }
        ),
        "results": results,
    }
    out_path = config.ARTIFACTS_DIR / "cluster_metadata" / "experiment_config.json"
    out_path.write_text(json.dumps(manifest, indent=1, default=str))
    print(f"\nExperiment manifest -> {out_path}")
    return results


if __name__ == "__main__":
    run_pipeline()
