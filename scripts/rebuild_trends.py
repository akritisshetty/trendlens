#!/usr/bin/env python3
"""
rebuild_trends.py
-----------------
Regenerate the production TrendLens artifacts from the FULL, regenerated
Instagram embeddings:

  * cluster labels (HDBSCAN on UMAP-10d over all embedded posts)
  * cluster summaries      -> trends.json themes
  * temporal trends        -> trends.json growth / emerging score
  * hashtag trends
  * RAG index (FAISS)      -> rag_chunks.json + rag_index.faiss

This replaces the stale artifacts that were built when only ~184 posts had
embeddings, so the live app reflects the whole dataset.

Run:  venv/bin/python scripts/rebuild_trends.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from src.clustering import reduce_dimensions, run_hdbscan


def main() -> int:
    emb = np.load(config.INSTAGRAM_EMBEDDINGS_PATH)
    meta = pd.read_parquet(config.INSTAGRAM_DIR / "embed_meta.parquet")
    assert len(emb) == len(meta), "embeddings/metadata misaligned"

    print(f"[rebuild-trends] {len(emb)} posts")

    # 1. Clustering (deterministic, cached)
    red = reduce_dimensions(emb, method="umap", n_components=10, seed=42, force=False)
    labels, _, _ = run_hdbscan(red, min_cluster_size=10, min_samples=3,
                               cluster_selection_method="eom")
    np.save(config.CLUSTER_MODELS_DIR / "labels_instagram.npy", labels)

    clusters: dict[int, list[int]] = {}
    for i, lb in enumerate(labels):
        if lb >= 0:
            clusters.setdefault(int(lb), []).append(i)

    # 2. Summaries + temporal + hashtag trends
    from src import data_collector as dc

    summaries = dc.summarize_clusters(meta, labels, clusters, emb)
    temporal = dc.compute_temporal_trends(meta, labels, clusters)
    hashtag_trends = dc.compute_hashtag_trends(meta)

    # 3. RAG index + trends.json
    dc.build_rag_index(summaries, temporal, meta)
    trends = dc.save_trends_json(summaries, temporal, meta, labels, hashtag_trends)

    print(f"\n[rebuild-trends] {len(summaries)} themes written -> "
          f"{config.INSTAGRAM_TRENDS_PATH}")
    for t in trends["themes"][:5]:
        print(f"  - {t['name']:<40} emerging={t['emerging_score']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
