"""
trends.py
---------
Temporal trend detection (Stage 9) + text baseline (Stage 11).

Design rules:
  * A trend is GROWTH, not size. A large-but-flat cluster must not
    automatically outrank a small-but-growing one.
  * No rigged timestamps — these metrics run on the neutral synthetic
    timestamps, so results are honest demo numbers (mostly noise/stable).
  * The emerging-trend score formula is documented and treated as
    experimental: we compute several alternatives and compare their
    rankings with Spearman correlation.
  * Text baseline v1: tag-frequency growth (no captions exist in SMPD).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

import config


# ──────────────────────────────────────────────────────────────────────────
# Aggregation
# ──────────────────────────────────────────────────────────────────────────
def aggregate_cluster_trends(
    df: pd.DataFrame,
    cluster_col: str = "cluster_id",
    period: str = "M",
    id_col: str = "post_id",
    user_col: str = "user_id",
    likes_col: str = "likes",
    comments_col: str = "comments",
) -> pd.DataFrame:
    """
    Aggregate posts per (cluster, period).

    Returns columns: cluster_id, period, post_count, unique_users,
    average_engagement, median_engagement.
    """
    out = df.copy()
    t = out["timestamp"]
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_localize(None)
    out["period"] = t.dt.to_period(period)
    eng = out[likes_col].fillna(0) + out[comments_col].fillna(0)
    out["_eng"] = eng

    # Robust user column: callers may pass Instagram metadata that has
    # ``author`` instead of ``user_id``. Fall back gracefully rather than crash.
    if user_col not in out.columns:
        for candidate in ("author", "full_name", "owner_id"):
            if candidate in out.columns:
                user_col = candidate
                break
        else:
            out[user_col] = "anonymous"

    g = out.groupby([cluster_col, "period"])
    agg = (
        g.size()
        .to_frame("post_count")
        .join(g[user_col].nunique().rename("unique_users"))
        .join(g["_eng"].mean().rename("average_engagement"))
        .join(g["_eng"].median().rename("median_engagement"))
    ).reset_index()
    return agg


def _period_index(series: pd.Series, period: str) -> pd.Index:
    start = series.min()
    end = series.max()
    return pd.period_range(start=start, end=end, freq=period)


# ──────────────────────────────────────────────────────────────────────────
# Growth metrics
# ──────────────────────────────────────────────────────────────────────────
def growth_metrics(
    agg: pd.DataFrame,
    period: str = "M",
    window: int = 3,
    eps: float = 1.0,
) -> pd.DataFrame:
    """
    Per-cluster growth metrics computed on the post-count series.

    Metrics (documented):
      n_posts           total posts in the cluster
      mean_period_posts mean posts/period over the observed range
      recent_growth     (sum last `window` periods - sum prior `window`)
                        / (sum prior `window` + eps)   [primary signal]
      percentage_growth (last period - first period) / (first period + eps)
      slope             OLS slope of counts vs period index (per-period rate)
      rolling_growth    mean of per-period relative change over last `window`
      acceleration      slope of recent_growth computed on rolling halves
                        > 0 => accelerating, < 0 => decelerating
    """
    rows = []
    for c, sub in agg.groupby("cluster_id"):
        sub = sub.set_index("period")
        full = sub["post_count"].reindex(_period_index(sub.index, period), fill_value=0)
        counts = full.to_numpy(dtype="float64")

        n = len(counts)
        recent = float(counts[-window:].sum())
        prior = float(counts[-2 * window : -window].sum())
        r_growth = (recent - prior) / (prior + eps)

        first = float(counts[0])
        last = float(counts[-1])
        p_growth = (last - first) / (first + eps)

        # OLS slope over period indices
        x = np.arange(n, dtype="float64")
        if n >= 2:
            slope, _, _, _, _ = sp_stats.linregress(x, counts)
        else:
            slope = 0.0

        # per-period relative change in the last window
        if n >= 2:
            diffs = np.diff(counts[-window - 1 :]) / (counts[-window - 1 : -1] + eps)
            rolling_growth = float(diffs.mean()) if len(diffs) else 0.0
        else:
            rolling_growth = 0.0

        # acceleration: slope of recent_growth across successive windows
        if n >= 2 * window:
            first_half = float(counts[: n // 2].sum())
            second_half = float(counts[n // 2 :].sum())
            acceleration = (second_half - first_half) / (first_half + eps)
        else:
            acceleration = 0.0

        rows.append(
            {
                "cluster_id": int(c),
                "n_posts": int(sub["post_count"].sum()),
                "mean_period_posts": float(counts.mean()),
                "recent_growth": r_growth,
                "percentage_growth": p_growth,
                "slope": float(slope),
                "rolling_growth": rolling_growth,
                "acceleration": acceleration,
                "median_engagement": float(sub["median_engagement"].median()),
                "average_engagement": float(sub["average_engagement"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────
# Emerging trend score
# ──────────────────────────────────────────────────────────────────────────
def _minmax(s: pd.Series) -> pd.Series:
    lo, hi = float(s.min()), float(s.max())
    if hi - lo < 1e-12:
        return pd.Series(0.5, index=s.index)
    return (s - lo) / (hi - lo)


def trend_scores(
    metrics: pd.DataFrame,
    stability: dict[int, float] | pd.Series | None = None,
) -> pd.DataFrame:
    """
    Emerging trend scores, three documented alternatives:

      S1 = norm(recent_growth)                         growth alone
      S2 = S1 * norm(n_posts)                          growth x size
      S3 = S2 * norm(stability)                        growth x size x stability

    stability defaults to ``mean_period_posts`` when not supplied.
    """
    m = metrics.copy()
    g = _minmax(m["recent_growth"].fillna(0))
    size = _minmax(np.log1p(m["n_posts"]))
    if stability is not None:
        if isinstance(stability, dict):
            stab = m["cluster_id"].map(stability).fillna(0)
        else:
            stab = m["cluster_id"].map(
                pd.Series(stability, index=m["cluster_id"]).to_dict()
            ).fillna(0)
    else:
        stab = _minmax(m["mean_period_posts"])

    m["trend_score_growth"] = g
    m["trend_score_growth_size"] = g * size
    m["trend_score_growth_size_stability"] = g * size * stab
    return m


def compare_score_alternatives(scored: pd.DataFrame) -> dict[str, float]:
    """Spearman correlations between the three score alternatives."""
    cols = [
        "trend_score_growth",
        "trend_score_growth_size",
        "trend_score_growth_size_stability",
    ]
    result = {}
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            rho, p = sp_stats.spearmanr(scored[a], scored[b])
            result[f"{a}_vs_{b}"] = {"spearman": float(rho), "p": float(p)}
    return result


def classify_lifecycle(
    metrics: pd.DataFrame,
    rising_threshold: float = 0.25,
    declining_threshold: float = -0.25,
) -> pd.DataFrame:
    """
    Lifecycle stage from ``recent_growth`` (documented, configurable):

      Rising   : recent_growth >= rising_threshold
      Declining: recent_growth <= declining_threshold
      Stable   : otherwise
    """
    out = metrics.copy()
    out["lifecycle"] = np.where(
        out["recent_growth"] >= rising_threshold,
        "Rising",
        np.where(out["recent_growth"] <= declining_threshold, "Declining", "Stable"),
    )
    return out


# ──────────────────────────────────────────────────────────────────────────
# Text baseline (Stage 11, v1 — tags only)
# ──────────────────────────────────────────────────────────────────────────
def text_trend_scores(
    df: pd.DataFrame,
    cluster_col: str = "cluster_id",
    tags_col: str = "tags",
    period: str = "M",
    top_k: int = 10,
    window: int = 3,
) -> pd.DataFrame:
    """
    Per-cluster text baseline.

    For each cluster: take its ``top_k`` most frequent tags, measure each
    tag's monthly post frequency across the WHOLE dataset, compute that
    tag's recent growth (same formula as visual), and return the cluster's
    text trend score = weighted mean (by tag frequency) of tag growth.
    """
    from src.data_loader import parse_tags

    out = df.copy()
    t = out["timestamp"]
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_localize(None)
    out["period"] = t.dt.to_period(period)
    out["_tags"] = out[tags_col].map(parse_tags)

    # global tag x period frequency
    tag_rows = []
    for tags, p in zip(out["_tags"], out["period"]):
        for t in tags:
            tag_rows.append((t, p))
    if not tag_rows:
        return pd.DataFrame()
    tag_freq = (
        pd.DataFrame(tag_rows, columns=["tag", "period"])
        .groupby(["tag", "period"])
        .size()
        .to_frame("count")
        .reset_index()
    )

    rows = []
    for c, sub in out.groupby(cluster_col):
        if c < 0:
            continue
        tag_counts = sub["_tags"].explode().value_counts()
        top = tag_counts.head(top_k).index.tolist()
        weights = tag_counts.head(top_k).to_numpy(dtype="float64")
        w = weights / (weights.sum() + 1e-9)

        gs = []
        for tag in top:
            tf = tag_freq[tag_freq["tag"] == tag].set_index("period")["count"]
            tf = tf.reindex(_period_index(tf.index, period), fill_value=0).to_numpy(
                dtype="float64"
            )
            n = len(tf)
            recent = float(tf[-window:].sum())
            prior = float(tf[-2 * window : -window].sum())
            gs.append((recent - prior) / (prior + 1.0))
        text_score = float(np.dot(w, np.asarray(gs))) if gs else 0.0
        rows.append({"cluster_id": int(c), "text_trend_score": text_score})
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────
def plot_cluster_trends(
    agg: pd.DataFrame,
    metrics: pd.DataFrame,
    top_n: int = 6,
    out_dir: Path | None = None,
) -> list[Path]:
    """Growth-over-time + engagement charts for the top-ranked clusters."""
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir) if out_dir else config.FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    top = (
        metrics.sort_values("trend_score_growth_size_stability", ascending=False)
        .head(top_n)["cluster_id"]
        .tolist()
    )
    paths = []
    for c in top:
        sub = agg[agg["cluster_id"] == c].set_index("period")
        fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
        sub["post_count"].plot(ax=ax[0], marker="o", title=f"cluster {c} — posts/period")
        sub["average_engagement"].plot(ax=ax[1], marker="o", title="avg engagement")
        fig.tight_layout()
        p = out_dir / f"trend_cluster_{int(c):03d}.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)
    return paths


# ──────────────────────────────────────────────────────────────────────────
# CLI driver
# ──────────────────────────────────────────────────────────────────────────
def run_pipeline(
    period: str = "M",
    window: int = 3,
    rising_threshold: float = 0.25,
    declining_threshold: float = -0.25,
) -> dict:
    """
    Phase 4 driver over the Phase 3 clustering artifacts.
    """
    import json

    from src import clustering

    meta = clustering.load_aligned_metadata()
    labels = np.load(config.CLUSTER_MODELS_DIR / "labels_umap10.npy")

    df = meta.copy()
    df["cluster_id"] = labels
    df = df[df["cluster_id"] >= 0].copy()  # drop noise

    agg = aggregate_cluster_trends(df, period=period)
    metrics = growth_metrics(agg, period=period, window=window)
    scored = trend_scores(metrics)
    ranked = classify_lifecycle(scored, rising_threshold, declining_threshold)
    scored = scored.sort_values("trend_score_growth_size_stability", ascending=False)
    compare = compare_score_alternatives(scored)

    text_scores = text_trend_scores(df, period=period)
    ranked = ranked.merge(text_scores, on="cluster_id", how="left")

    ranked.to_csv(config.ARTIFACTS_DIR / "cluster_metadata" / "trend_metrics.csv", index=False)
    agg.to_csv(config.ARTIFACTS_DIR / "cluster_metadata" / "cluster_trend_agg.csv", index=False)

    lifecycle_counts = ranked["lifecycle"].value_counts().to_dict()
    print("=" * 60)
    print("TREND DETECTION (neutral synthetic timestamps — demo only)")
    print("=" * 60)
    print("clusters analysed :", len(ranked))
    print("lifecycle         :", lifecycle_counts)
    print("\nScore-alternative agreement (Spearman):")
    for k, v in compare.items():
        print(f"  {k:<52} rho={v['spearman']:.3f}")
    print("\nTop 10 emerging trends (S3 = growth×size×stability):")
    cols = ["cluster_id", "n_posts", "recent_growth", "trend_score_growth_size_stability", "lifecycle", "text_trend_score"]
    print(ranked[cols].head(10).to_string(index=False))

    paths = plot_cluster_trends(agg, scored)
    print(f"\ncharts: {[p.name for p in paths]}")

    manifest = {
        "experiment": "phase4_trends",
        "config": config.experiment_config(
            {
                "period": period,
                "window": window,
                "rising_threshold": rising_threshold,
                "declining_threshold": declining_threshold,
                "score_primary": "trend_score_growth_size_stability",
                "timestamps": "neutral-synthetic (not rigged)",
            }
        ),
        "lifecycle_counts": lifecycle_counts,
        "score_alternative_agreement": compare,
    }
    out = config.ARTIFACTS_DIR / "cluster_metadata" / "trends_experiment.json"
    out.write_text(json.dumps(manifest, indent=1, default=str))
    print(f"\nmanifest -> {out}")
    return {"ranked": ranked, "agg": agg, "metrics": scored, "compare": compare}


if __name__ == "__main__":
    run_pipeline()
