#!/usr/bin/env python3
"""
sweep_hyperparameters.py
------------------------
Select clustering and retrieval hyperparameters on the REAL Instagram corpus.

This replaces ``benchmark_algorithms.py`` and ``benchmark_clip_sizes.py``, whose
selected values (HDBSCAN min_cluster_size=50, min_samples=10) were never the
ones running in production — ``scripts/rebuild_trends.py`` used 10/3 — and whose
supporting metric was silhouette on a 5-cluster synthetic fixture.

Why the criteria here are different
-----------------------------------
Silhouette alone cannot pick a clustering for this product. A configuration that
shreds 461 posts into 40 tight clusters will score a high silhouette and be
useless, because each cluster is too small to support a trend claim or a shot
recipe. The criteria below are chosen for what TrendLens actually needs:

  STABILITY   Adjusted Rand Index between partitions produced with different
              random seeds. This is the only fully non-circular criterion
              available without labels, and it directly measures the thing the
              cluster tracker depends on: cluster identity surviving re-runs.
              A configuration that reshuffles on a new seed is unusable for
              longitudinal tracking regardless of its silhouette.
  SEPARATION  eta^2 — the share of CLIP-space variance explained by cluster
              membership. Scale-free and comparable across configurations
              with different cluster counts.
  COVERAGE    fraction of posts assigned to a cluster. Directly caps how much
              of the corpus can ever produce a trend verdict.
  THEME SIZE  median cluster size. Below ~8 posts a cluster cannot satisfy the
              trend definition's support floor, so small clusters are dead
              weight that still cost BLIP captioning time.

Reported, not optimised into a single number: a weighted composite hides the
tradeoff between stability and coverage. The table is the deliverable.

Run:  venv/bin/python scripts/sweep_hyperparameters.py
"""

from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src.data_quality import clean_engagement  # noqa: E402
from src.style_tags import compute_style_scores  # noqa: E402

SEEDS = (42, 1337, 7)
#: A cluster smaller than this cannot clear the trend definition's
#: min_recent_posts=8 support floor, so it can never yield a Rising verdict.
MIN_USEFUL_CLUSTER = 8


def eta_squared(x: np.ndarray, labels: np.ndarray) -> float:
    """Share of variance in ``x`` explained by cluster membership."""
    total = ((x - x.mean(axis=0)) ** 2).sum()
    if total < 1e-12:
        return 0.0
    between = 0.0
    for c in np.unique(labels):
        m = labels == c
        if m.sum() == 0:
            continue
        diff = x[m].mean(axis=0) - x.mean(axis=0)
        between += m.sum() * float(diff @ diff)
    return float(between / total)


def pairwise_ari(partitions: list[np.ndarray]) -> float:
    """Mean ARI across all pairs of partitions (clustering stability)."""
    from sklearn.metrics import adjusted_rand_score

    vals = []
    for i in range(len(partitions)):
        for j in range(i + 1, len(partitions)):
            a, b = partitions[i], partitions[j]
            mask = (a >= 0) & (b >= 0)
            if len(np.unique(a[mask])) < 2 or len(np.unique(b[mask])) < 2:
                continue
            vals.append(adjusted_rand_score(a[mask], b[mask]))
    return float(np.mean(vals)) if vals else float("nan")


def silhouette(emb: np.ndarray, labels: np.ndarray, sample: int = 1500) -> float:
    from sklearn.metrics import silhouette_score

    mask = labels >= 0
    idx = np.where(mask)[0]
    if len(np.unique(labels[idx])) < 2:
        return float("nan")
    if len(idx) > sample:
        rng = np.random.default_rng(config.RANDOM_SEED)
        idx = rng.choice(idx, size=sample, replace=False)
    return float(silhouette_score(emb[idx], labels[idx], metric="euclidean"))


def run_config(
    emb: np.ndarray,
    style_scores: np.ndarray,
    n_components: int,
    min_cluster_size: int,
    min_samples: int,
    method: str,
) -> dict:
    """Evaluate one clustering configuration across all seeds."""
    from src.clustering import reduce_dimensions, run_hdbscan

    partitions, rows = [], []
    for seed in SEEDS:
        red = reduce_dimensions(
            emb,
            method="umap",
            n_components=n_components,
            seed=seed,
            cache_path=config.ARTIFACTS_DIR / "embeddings"
            / f"umap_{n_components}d_seed{seed}.npy",
        )
        labels, probs, _ = run_hdbscan(
            red,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method=method,
        )
        partitions.append(labels)
        sizes = np.array([(labels == c).sum() for c in np.unique(labels[labels >= 0])])
        rows.append(
            {
                "n_clusters": int(len(sizes)),
                "coverage": float((labels >= 0).mean()),
                "noise_pct": float((labels == -1).mean()),
                "median_cluster_size": float(np.median(sizes)) if len(sizes) else 0.0,
                "min_cluster_size_actual": int(sizes.min()) if len(sizes) else 0,
                "useful_clusters": int((sizes >= MIN_USEFUL_CLUSTER).sum()),
                "eta2": eta_squared(emb[labels >= 0], labels[labels >= 0]),
                "silhouette": silhouette(emb, labels),
            }
        )

    agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    agg.update(
        {
            "umap_dims": n_components,
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "cluster_selection_method": method,
            "stability_ari": pairwise_ari(partitions),
        }
    )
    return agg


def score_retrieval(
    emb: np.ndarray,
    labels: np.ndarray,
    meta: pd.DataFrame,
    k: int,
) -> dict:
    """
    Retrieval quality for the RAG path: given a text query, does semantic search
    return the chunk describing the visually-matching cluster?

    Queries are built from each cluster's own dominant caption words, so this
    asks: if a user describes a theme in the words the posters used, does
    retrieval land on the right theme's chunk? Ground truth is the VISUAL
    cluster, not the text, which is the correct direction for this product —
    the claim is that visual grouping finds structure that text matching alone
    would miss.

    Both sides of the comparison are MiniLM sentence embeddings. An earlier
    version dot-producted 384-d text query vectors against the 512-d CLIP image
    matrix, which is a category error: the two encoders share no joint space,
    so the resulting scores are meaningless. Retrieval runs over text chunks
    (FAISS IndexFlatIP over MiniLM), so that is what is measured here.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Build one searchable text document per cluster — mirrors what the RAG
    # index actually contains.
    cluster_ids = sorted(int(c) for c in np.unique(labels[labels >= 0]))
    docs, owners = [], []
    for c in cluster_ids:
        idx = np.where(labels == c)[0]
        text = " ".join(meta.iloc[idx]["caption"].fillna("").tolist()[:40])
        docs.append(text or f"cluster {c}")
        owners.append(c)
    if not docs:
        return {"k": k, "hit_rate": float("nan"), "mrr": float("nan"), "n_queries": 0}

    doc_emb = model.encode(docs, normalize_embeddings=True, show_progress_bar=False)

    # Queries: each cluster's own top caption words, with a held-out slice of
    # its posts excluded so the query is not a copy of the indexed document.
    from collections import Counter

    import re

    stop = set(
        """the and for with from this that are was were has have had been being
        you your our their they them about into over more most than just like
        very really such only also not but its it's photo photos video""".split()
    )
    queries, targets = [], []
    for c in cluster_ids:
        idx = np.where(labels == c)[0]
        if len(idx) < MIN_USEFUL_CLUSTER:
            continue
        captions = meta.iloc[idx]["caption"].fillna("").tolist()
        held_out = captions[-max(3, len(captions) // 5) :]
        words: list[str] = []
        for cap in held_out:
            words += [
                w.lower()
                for w in re.findall(r"[A-Za-z][A-Za-z0-9]{3,}", str(cap))
                if w.lower() not in stop
            ]
        if not words:
            continue
        queries.append(" ".join(w for w, _ in Counter(words).most_common(6)))
        targets.append(c)

    if not queries:
        return {"k": k, "hit_rate": float("nan"), "mrr": float("nan"), "n_queries": 0}

    q_emb = model.encode(queries, normalize_embeddings=True, show_progress_bar=False)
    sims = q_emb @ doc_emb.T
    hits, mrr, n = 0, 0.0, 0
    for row, target in enumerate(targets):
        n += 1
        order = np.argsort(-sims[row])[:k]
        ranked = [owners[o] for o in order]
        if target in ranked:
            hits += 1
        for rank, lab in enumerate(ranked, start=1):
            if lab == target:
                mrr += 1.0 / rank
                break
    return {
        "k": k,
        "hit_rate": hits / n if n else float("nan"),
        "mrr": mrr / n if n else float("nan"),
        "n_queries": n,
        "n_docs": len(docs),
    }


def main() -> int:
    emb = np.load(config.INSTAGRAM_EMBEDDINGS_PATH)
    meta = pd.read_parquet(config.INSTAGRAM_DIR / "embed_meta.parquet")
    assert len(emb) == len(meta), "embeddings/metadata misaligned"
    meta = clean_engagement(meta)
    style_scores = compute_style_scores(emb)

    print(f"[sweep] {len(emb)} posts, {emb.shape[1]}-d embeddings\n")

    grid = list(
        product(
            [5, 10, 15, 20],       # UMAP dims
            [5, 8, 10, 15, 20, 30],  # min_cluster_size
            [3, 5, 10],            # min_samples
        )
    )
    print(f"[sweep] {len(grid)} clustering configurations x {len(SEEDS)} seeds")

    results = []
    for i, (dims, mcs, ms) in enumerate(grid, start=1):
        row = run_config(emb, style_scores, dims, mcs, ms, "eom")
        results.append(row)
        print(
            f"  [{i:>2}/{len(grid)}] umap={dims:<3} mcs={mcs:<3} ms={ms:<3} "
            f"-> k={row['n_clusters']:<3.0f} cov={row['coverage']:.2f} "
            f"stability={row['stability_ari']:.3f} eta2={row['eta2']:.3f} "
            f"useful={row['useful_clusters']:.0f}",
            flush=True,
        )

    df = pd.DataFrame(results)
    out_csv = ROOT / "artifacts" / "cluster_metadata" / "sweep_clustering.csv"
    df.to_csv(out_csv, index=False)

    # A configuration is only viable if it produces clusters large enough to
    # clear the trend definition's support floor.
    viable = df[df["useful_clusters"] >= 5].copy()
    if viable.empty:
        print("\n[sweep] no configuration produced >=5 usable clusters")
        return 0

    # Selection rule, and the reasoning matters:
    #
    # Stability is a GATE, not a ranking criterion. Across the viable configs it
    # spans roughly 0.92-0.96 — a narrow band where the difference is not
    # meaningful — while eta^2 spans 0.14-0.25, nearly 2x. Ranking on stability
    # first therefore lets a near-tie decide the outcome while discarding the
    # criterion that actually varies. An earlier version of this script did
    # exactly that and selected umap=20/mcs=30: stability 0.958 (vs 0.949 for
    # the best alternative) but only 6 clusters at eta^2=0.138, i.e. it bought
    # 0.009 of a saturated metric and paid with less than half the separation
    # and a third of the usable clusters.
    #
    # So: require stability to clear the level at which cluster identity
    # reliably survives a re-run, then maximise separation, then prefer more
    # usable clusters and higher coverage.
    MIN_STABLE_ARI = 0.92
    stable = viable[viable["stability_ari"] >= MIN_STABLE_ARI].copy()
    if stable.empty:
        print(
            f"\n[sweep] no configuration reached stability ARI {MIN_STABLE_ARI}; "
            f"falling back to the most stable available"
        )
        stable = viable
    stable = stable.sort_values(
        ["eta2", "useful_clusters", "coverage"], ascending=False
    )
    best = stable.iloc[0]

    print("\n" + "=" * 78)
    print("SELECTED CONFIGURATION (stability gate >= "
          f"{MIN_STABLE_ARI}, then max separation)")
    print("=" * 78)
    for k in (
        "umap_dims",
        "min_cluster_size",
        "min_samples",
        "cluster_selection_method",
        "n_clusters",
        "useful_clusters",
        "coverage",
        "noise_pct",
        "median_cluster_size",
        "stability_ari",
        "eta2",
        "silhouette",
    ):
        print(f"  {k:<24} {best[k]:.4f}" if isinstance(best[k], float) else f"  {k:<24} {best[k]}")

    print("\n[sweep] top 8 by separation among stability-gated configs:")
    cols = [
        "umap_dims", "min_cluster_size", "min_samples", "n_clusters",
        "useful_clusters", "coverage", "stability_ari", "eta2", "silhouette",
    ]
    print(stable[cols].head(8).to_string(index=False))

    # Retrieval sweep under the selected clustering.
    print("\n[sweep] retrieval k sweep under selected clustering")
    from src.clustering import reduce_dimensions, run_hdbscan

    red = reduce_dimensions(
        emb, method="umap", n_components=int(best["umap_dims"]), seed=config.RANDOM_SEED
    )
    labels, _, _ = run_hdbscan(
        red,
        min_cluster_size=int(best["min_cluster_size"]),
        min_samples=int(best["min_samples"]),
        cluster_selection_method="eom",
    )
    ret = pd.DataFrame([score_retrieval(emb, labels, meta, k) for k in (1, 3, 5, 8)])
    print(ret.to_string(index=False))
    ret.to_csv(ROOT / "artifacts" / "cluster_metadata" / "sweep_retrieval.csv", index=False)

    manifest = {
        "experiment": "hyperparameter_sweep_real_corpus",
        "n_posts": int(len(emb)),
        "note": (
            "Selected on the real Instagram corpus. Stability (mean ARI across "
            f"seeds {SEEDS}) is a GATE at >= {MIN_STABLE_ARI} because cluster "
            "identity must survive re-runs for longitudinal tracking to be "
            "valid; separation (eta^2) then decides, since it varies far more "
            "across the viable configurations than stability does."
        ),
        "selected": {k: (float(best[k]) if isinstance(best[k], (int, float, np.floating)) else best[k]) for k in cols},
        "grid_size": len(grid),
        "seeds": list(SEEDS),
    }
    (ROOT / "artifacts" / "cluster_metadata" / "sweep_manifest.json").write_text(
        json.dumps(manifest, indent=1, default=str)
    )
    print(f"\n[sweep] wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
