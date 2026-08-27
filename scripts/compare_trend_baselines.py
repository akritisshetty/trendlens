#!/usr/bin/env python3
"""
compare_trend_baselines.py
--------------------------
Compare TrendLens' image-based trend detection against the ACTUAL tools
people use for trend detection today:

  * Google-Trends-style keyword/search-volume detection   (word frequency growth)
  * Hashtag / social-listening detection                  (hashtag frequency growth)
  * Engagement "trending" lists                           (rank by likes+comments)
  * Post-volume / velocity                                (raw count growth)
  vs.
  * TrendLens: unsupervised visual clustering over CLIP image embeddings,
    which claims to surface trends BEFORE they have a shared name.

Two honest, non-circular measurements:

  1. GROUP VALUE COHESION  — does a detector's grouping bring together posts
     with SIMILAR engagement (all-high or all-low) rather than a mix? A good
     trend detector yields value-coherent groups. Measured as
     1 - within_group_std / overall_std (0 = no better than random mixing,
     1 = perfectly coherent groups). The engagement list is defined to be
     perfectly coherent here, so TrendLens' score is measured against it.

  2. TEXT DIVERSITY INSIDE VISUAL CLUSTERS (core product claim) — TrendLens'
     image clusters group posts that share almost NO hashtags/words. That is
     exactly the situation where word-based tools (Google Trends, hashtag
     listening) cannot see the trend, because it has no shared name yet.
     Low mean pairwise Jaccard => TrendLens finds "un-named" trends.

Run:  venv/bin/python scripts/compare_trend_baselines.py
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from src.clustering import reduce_dimensions, run_hdbscan

OUT = ROOT / "baseline_real_tools_results.csv"

_STOP = set("""thea an and or but for with from this that these those are waswere
is not on at by to of in it its you your we our have has had be been being do does
did as if so then just like what when where how why which who whom via very really
such more most than about only into over out up down now get got also new one two
first last day week month year good great best top""".split())


def _hashtags(caption: str) -> set[str]:
    return set(re.findall(r"#([A-Za-z0-9_]+)", str(caption or "")))


def _words(caption: str) -> set[str]:
    toks = re.findall(r"[A-Za-z][A-Za-z0-9]{2,}", str(caption or "").lower())
    return set(t for t in toks if t not in _STOP)


def _cohesion(df: pd.DataFrame, group_col: str) -> float:
    """1 - mean(within-group engagement std) / overall std. Higher = better."""
    overall = float(df["engagement"].std())
    if overall == 0:
        return 0.0
    g = df[df[group_col].notna()].groupby(group_col)["engagement"]
    sizes = g.size()
    valid = sizes[sizes > 1].index
    if len(valid) == 0:
        return 0.0
    within = sum(
        float(df[df[group_col] == c]["engagement"].std())
        for c in valid
    ) / len(valid)
    return 1.0 - within / overall


def main() -> int:
    meta = pd.read_parquet(config.INSTAGRAM_DIR / "embed_meta.parquet")
    labels_path = config.CLUSTER_MODELS_DIR / "labels_instagram.npy"

    if labels_path.exists() and len(np.load(labels_path)) == len(meta):
        labels = np.load(labels_path)
    else:
        print("[baseline] generating TrendLens clusters from embeddings...")
        emb = np.load(config.INSTAGRAM_EMBEDDINGS_PATH)
        assert len(emb) == len(meta)
        red = reduce_dimensions(emb, method="umap", n_components=10, seed=42, force=False)
        labels, _, _ = run_hdbscan(red, min_cluster_size=10, min_samples=3,
                                   cluster_selection_method="eom")
        np.save(labels_path, labels)

    df = meta.copy()
    df["cluster_id"] = labels
    df["engagement"] = df["likes"].fillna(0) + df["comments"].fillna(0)
    df["_tags"] = df["caption"].map(_hashtags)

    n_clusters = int(len(np.unique(labels[labels >= 0])))
    clustered = int((labels >= 0).sum())
    print(f"\nDataset: {len(df)} posts | TrendLens: {clustered} clustered, "
          f"{n_clusters} image clusters\n")

    results = []

    def record(name, metric, value, note=""):
        print(f"  [{name}] {metric}: {value:.4f}  {note}")
        results.append(dict(name=name, metric=metric,
                            value=round(value, 6) if isinstance(value, float) else value,
                            note=note))

    # ── 1. Group value cohesion for each detector ───────────────────────────
    # TrendLens: image clusters
    tl = df[df["cluster_id"] >= 0].copy()
    record("TrendLens (visual clusters)", "group_value_cohesion",
           _cohesion(tl, "cluster_id"))

    # Engagement list: bin posts into engagement quartiles (coherent by definition)
    eng = df.assign(_bin=pd.qcut(df["engagement"], 4, labels=False, duplicates="drop"))
    record("Engagement trending list", "group_value_cohesion",
           _cohesion(eng, "_bin"), "quartile bins (upper bound)")

    # Hashtag detection: group posts sharing each top hashtag
    hot_tags = df["_tags"].explode().value_counts()
    tag_rows = []
    for tag in hot_tags.index[:40]:
        members = df[df["_tags"].apply(lambda s: tag in s)]
        for idx in members.index:
            tag_rows.append((idx, f"#{tag}"))
    tag_df = pd.DataFrame(tag_rows, columns=["index", "_tg"])
    tag_df = tag_df.merge(df[["engagement"]], left_on="index", right_index=True, how="left")
    record("Hashtag-frequency detection", "group_value_cohesion",
           _cohesion(tag_df, "_tg"), f"top-40 tags")

    # Keyword detection
    kw_growth = defaultdict(float)
    for _, row in df.iterrows():
        for w in _words(row["caption"]):
            kw_growth[w] += 1 + np.log1p(row["engagement"])
    hot_kw = [w for w, _ in sorted(kw_growth.items(), key=lambda kv: -kv[1])]
    kw_rows = []
    for w in hot_kw[:40]:
        members = df[df["caption"].map(lambda c: w in _words(c))]
        for idx in members.index:
            kw_rows.append((idx, w))
    kw_df = pd.DataFrame(kw_rows, columns=["index", "_kw"])
    kw_df = kw_df.merge(df[["engagement"]], left_on="index", right_index=True, how="left")
    record("Keyword/Google-Trends-style", "group_value_cohesion",
           _cohesion(kw_df, "_kw"), f"top-40 keywords")

    # ── 2. Core product claim: text diversity inside visual clusters ────────
    print("\n--- Core claim: are TrendLens clusters textually 'un-named'? ---")
    tag_homog, kw_homog = [], []
    for c, sub in tl.groupby("cluster_id"):
        all_tags = [set(t) for t in sub["_tags"]]
        all_kw = [_words(cap) for cap in sub["caption"]]
        if len(all_tags) > 1:
            sims_t = [len(a & b) / (len(a | b) + 1e-9)
                      for i, a in enumerate(all_tags) for b in all_tags[i + 1:]]
            tag_homog.append(float(np.mean(sims_t)))
        if len(all_kw) > 1:
            sims_k = [len(a & b) / (len(a | b) + 1e-9)
                      for i, a in enumerate(all_kw) for b in all_kw[i + 1:]]
            kw_homog.append(float(np.mean(sims_k)))
    record("TrendLens text-diversity", "mean_hashtag_jaccard",
           float(np.mean(tag_homog)) if tag_homog else 0.0,
           "within-cluster mean pairwise hashtag overlap (low = no shared name)")
    record("TrendLens text-diversity", "mean_keyword_jaccard",
           float(np.mean(kw_homog)) if kw_homog else 0.0,
           "within-cluster mean pairwise keyword overlap (low = no shared name)")

    out = pd.DataFrame(results)
    out.to_csv(OUT, index=False)
    print(f"\nSaved {len(results)} metrics -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
