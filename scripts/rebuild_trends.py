#!/usr/bin/env python3
"""
rebuild_trends.py
-----------------
Regenerate the production TrendLens artifacts from the FULL, regenerated
Instagram embeddings:

  * cluster labels (HDBSCAN on UMAP-10d over all embedded posts)
  * cluster summaries      -> trends.json themes
  * per-cluster photography-execution profile (style tags, zero-shot CLIP)
  * temporal verdicts      -> trends.json, via the formal trend definition
  * hashtag trends
  * RAG index (FAISS)      -> rag_chunks.json + rag_index.faiss
  * longitudinal history   -> data/trend_history.db (one snapshot per run)

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
    # Hyperparameters come from config, selected on the real corpus by
    # scripts/sweep_hyperparameters.py.
    red = reduce_dimensions(
        emb,
        method="umap",
        n_components=config.UMAP_COMPONENTS,
        seed=config.RANDOM_SEED,
        force=False,
    )
    labels, _, _ = run_hdbscan(
        red,
        min_cluster_size=config.HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=config.HDBSCAN_MIN_SAMPLES,
        cluster_selection_method=config.HDBSCAN_SELECTION_METHOD,
    )
    np.save(config.CLUSTER_MODELS_DIR / "labels_instagram.npy", labels)

    clusters: dict[int, list[int]] = {}
    for i, lb in enumerate(labels):
        if lb >= 0:
            clusters.setdefault(int(lb), []).append(i)

    # 2. Summaries + temporal + hashtag trends
    from src import data_collector as dc
    from src.data_quality import clean_engagement
    from src.style_tags import compute_style_scores

    meta = clean_engagement(meta)
    meta["timestamp"] = pd.to_datetime(meta["timestamp"], utc=True, errors="coerce")

    # Photography-execution profile ("how it is shot", as opposed to "what is
    # in it"). This MUST be passed explicitly: summarize_clusters takes
    # style_scores as an optional argument defaulting to None, so omitting it
    # silently yields empty style_tags on every cluster. That is what happened
    # before this line existed — all 12 themes shipped with zero style tags,
    # which left the LLM writing layer with no execution evidence to turn into
    # shooting advice, and with instructions forbidding it from inventing any.
    style_scores = compute_style_scores(emb)

    summaries = dc.summarize_clusters(
        meta, labels, clusters, emb, style_scores=style_scores
    )
    temporal = dc.compute_temporal_trends(meta, labels, clusters)
    hashtag_trends = dc.compute_hashtag_trends(meta)

    # 3. RAG index + trends.json
    dc.build_rag_index(summaries, temporal, meta)
    trends = dc.save_trends_json(summaries, temporal, meta, labels, hashtag_trends)

    # 4. Append this observation set to the longitudinal history store
    from src.trend_definition import TrendHistory

    obs_rows = []
    daily_rows = []
    for s in summaries:
        cid = s["cluster_id"]
        t = temporal.get(cid, {})
        obs_rows.append(
            {
                "cluster_id": cid,
                "theme_name": s.get("name"),
                "n_total": t.get("total_posts"),
                "n_recent": t.get("n_recent"),
                "n_prior": t.get("n_prior"),
                "recent_rate_per_day": t.get("recent_rate_per_day"),
                "prior_rate_per_day": t.get("prior_rate_per_day"),
                "rate_ratio": t.get("rate_ratio"),
                "relative_growth": t.get("relative_growth"),
                "p_value": t.get("p_value"),
                "classification": t.get("classification"),
                "reason": t.get("classification_reason"),
                "median_likes": t.get("median_likes"),
                "likes_coverage": t.get("likes_coverage"),
            }
        )
        for day, n in (t.get("daily_counts") or {}).items():
            daily_rows.append({"cluster_id": cid, "day": day, "n_posts": n})

    history = TrendHistory()
    snap = history.record_snapshot(
        pd.DataFrame(obs_rows),
        pd.DataFrame(daily_rows),
        source="instagram",
        n_posts=len(meta),
        post_ids=meta["post_id"].tolist(),
    )
    status = history.history_status()
    if snap is None:
        print(
            "[rebuild-trends] history unchanged — this run covered the same "
            "post set as the previous snapshot, so it is a recomputation, not "
            "a new observation (pass allow_duplicate_corpus=True to force)"
        )
    else:
        print(
            f"[rebuild-trends] history snapshot {snap} recorded "
            f"({status['n_distinct_corpora']} distinct corpora — longitudinal "
            f"claims need >= 2)"
        )
    history.close()

    n_tagged = sum(1 for s in summaries if s.get("style_tags"))
    print(f"[rebuild-trends] {n_tagged}/{len(summaries)} themes carry style tags")
    print(f"\n[rebuild-trends] {len(summaries)} themes written -> "
          f"{config.INSTAGRAM_TRENDS_PATH}")
    for t in trends["themes"][:5]:
        print(f"  - {t['name']:<40} {t.get('classification','?'):<18} "
              f"priority={t['emerging_score']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
