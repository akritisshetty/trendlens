# Evaluation Against Popular Baselines

This report evaluates TrendLens' image-based trend detection against the
**actual tools people use for trend detection today** — Google Trends,
hashtag / social-listening, and engagement trending lists — rather than
against alternate ML components.

All metrics are computed on the **regenerated** full dataset:
461 embedded Instagram posts (CLIP ViT-B/32), clustered into **12 image
clusters** (351 posts clustered, 110 noise) via HDBSCAN on UMAP-10d
(`min_cluster_size=10`, `min_samples=3`).

Source: `scripts/compare_trend_baselines.py`
Results: `baseline_real_tools_results.csv`

---

## 1. Group Value Cohesion

Does a detector's grouping bring together posts with *similar* engagement
(all-high or all-low) rather than a mix? A good trend detector yields
value-coherent groups.

Measured as `1 - within_group_std / overall_std`, averaged over non-singleton
groups (0 = no better than random mixing, 1 = perfectly coherent groups).

| Detector | Group value cohesion |
|---|---|
| Hashtag-frequency detection | **0.687** |
| TrendLens (visual clusters) | **0.556** |
| Engagement trending list | 0.510 |
| Keyword / Google-Trends-style | 0.474 |

TrendLens' image clusters are more value-coherent than simple engagement
lists or keyword/Google-Trends-style detection, and close to hashtag-based
detection — while additionally capturing trends that text-based tools cannot
see (Section 2).

---

## 2. Core Product Claim: Text Diversity Inside Visual Clusters

TrendLens' image clusters group posts that share **almost no hashtags or
words**. This is exactly the situation where word-based tools (Google Trends,
hashtag/social-listening) are blind — the trend has **no shared name yet**.

Low within-cluster mean pairwise Jaccard ⇒ TrendLens finds "un-named" trends.

| Metric | Value |
|---|---|
| Mean hashtag Jaccard inside visual clusters | **0.024** |
| Mean keyword Jaccard inside visual clusters | **0.030** |

At a Jaccard overlap of ~0.025, posts grouped by **visual similarity** share
essentially no hashtags or keywords. A Google-Trends/keyword detector or
hashtag listenenr literally has no query term to run — the trend exists only
at the image level, which is TrendLens' unique advantage.

---

## 3. Notes on Metric Design

An earlier version used *engagement MRR* (rank by engagement vs. an
engagement-based held-in set). This proved **circular and degenerate**: every
baseline ranked by engagement while the ground truth was also engagement, so
each scored 1.0. It was removed in favor of the non-circular measures above.

---

## Artifacts

- `scripts/compare_trend_baselines.py` — reproducible comparison pipeline
- `baseline_real_tools_results.csv` — machine-readable results
- `scripts/rebuild_trends.py` — regenerated `data/instagram/trends.json`
  (12 themes / 461 posts / 20 tracked hashtags) from the fixed pipeline
