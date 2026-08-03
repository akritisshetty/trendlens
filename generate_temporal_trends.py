"""
generate_temporal_trends.py
----------------------------------------------------------------------
Step 5 of the TrendLens pipeline: Temporal Trend Tracking.

Run AFTER generate_clusters.py.

METHODOLOGY:
    For each post, we have:
      - timestamp         : when the photo was posted (photo-ID ordered,
                            mapped to 2010-2019)
      - trend_active_until: when the trend expires (engagement-driven)

    Together these define an "active window" per post:
        [timestamp  →  trend_active_until]

    For each quarter we count how many posts are simultaneously active.
    The quarter with the highest overlap count = PEAK QUARTER.

    This combines two synthetic signals in a way that still produces
    meaningful relative comparisons:
      - Relative post ordering IS preserved (photo IDs are chronological)
      - trend_active_until IS engagement-driven (high engagement = longer)

Inputs  (trendlens_outputs/):
    - metadata_clustered.csv

Outputs (trendlens_outputs/):
    - trend_metrics.csv            per-cluster peak + lifecycle stats
    - trend_graphs/                activity curve PNG per cluster
    - trend_summary.png            top 20 Rising trends overview
----------------------------------------------------------------------
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
OUTPUT_DIR  = Path("trendlens_outputs")
GRAPHS_DIR  = OUTPUT_DIR / "trend_graphs"
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

METADATA_PATH = OUTPUT_DIR / "metadata_clustered.csv"
MIN_SIZE      = 50   # skip tiny clusters

# ----------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------
meta = pd.read_csv(METADATA_PATH)
meta = meta[meta["cluster"] != -1].copy()

# Parse both time columns — strip timezone for simplicity
meta["ts_start"] = pd.to_datetime(
    meta["timestamp"], utc=True
).dt.tz_localize(None)

meta["ts_end"] = pd.to_datetime(
    meta["trend_active_until"], utc=True
).dt.tz_localize(None)

print(f"Loaded {len(meta):,} clustered images across "
      f"{meta['cluster'].nunique()} clusters\n")

# ----------------------------------------------------------------------
# 2. Build quarterly activity curves
#    For each quarter Q, count posts where ts_start <= Q <= ts_end
# ----------------------------------------------------------------------
quarters = pd.date_range(
    start="2010-01-01", end="2021-01-01", freq="QS"
)

def activity_curve(cluster_df):
    """Count active posts per quarter for one cluster."""
    counts = []
    for q in quarters:
        q_end = q + pd.offsets.QuarterEnd(1)
        n = ((cluster_df["ts_start"] <= q_end) &
             (cluster_df["ts_end"]   >= q)).sum()
        counts.append(int(n))
    return pd.Series(counts, index=quarters)

# ----------------------------------------------------------------------
# 3. Compute per-cluster metrics
# ----------------------------------------------------------------------
records = []

for cluster_id, group in meta.groupby("cluster"):
    if len(group) < MIN_SIZE:
        continue

    curve        = activity_curve(group)
    peak_idx     = int(curve.values.argmax())
    peak_quarter = quarters[peak_idx]
    peak_count   = int(curve.iloc[peak_idx])

    # Lifecycle stage based on where peak falls in the timeline
    total_q      = len(quarters)
    peak_position = peak_idx / total_q   # 0.0 = very start, 1.0 = very end

    if peak_position >= 0.60:
        lifecycle = "Rising"       # peak in last 40% of timeline
    elif peak_position <= 0.35:
        lifecycle = "Declining"    # peak in first 35% of timeline
    else:
        lifecycle = "Stable"       # peak in the middle

    # Active window: first and last quarter with any activity
    active_quarters = quarters[curve > 0]
    window_start    = str(active_quarters[0].date())  if len(active_quarters) > 0 else "N/A"
    window_end      = str(active_quarters[-1].date()) if len(active_quarters) > 0 else "N/A"

    # Engagement stats
    dominant_cat  = group["category"].mode().iloc[0]
    mean_eng      = float(group["engagement_rate"].mean())
    viral_rate    = float(group["is_viral"].mean())
    mean_duration = float(group["trend_duration_days"].mean())
    total_posts   = int(len(group))

    records.append({
        "cluster":                  cluster_id,
        "dominant_category":        dominant_cat,
        "total_posts":              total_posts,
        "peak_quarter":             str(peak_quarter.date()),
        "peak_active_posts":        peak_count,
        "lifecycle_stage":          lifecycle,
        "trend_window_start":       window_start,
        "trend_window_end":         window_end,
        "mean_engagement_rate":     round(mean_eng, 4),
        "viral_rate":               round(viral_rate, 4),
        "mean_trend_duration_days": round(mean_duration, 2),
    })

trend_df = pd.DataFrame(records).sort_values(
    ["lifecycle_stage", "mean_engagement_rate"],
    ascending=[True, False]
)
trend_df.to_csv(OUTPUT_DIR / "trend_metrics.csv", index=False)

# Print summary
counts = trend_df["lifecycle_stage"].value_counts()
print("Trend lifecycle classification:")
for stage in ["Rising", "Stable", "Declining"]:
    print(f"  {stage:10s}: {counts.get(stage, 0)} clusters")

# ----------------------------------------------------------------------
# 4. Per-cluster activity curve graphs
# ----------------------------------------------------------------------
color_map = {
    "Rising":   "#2ecc71",
    "Stable":   "#3498db",
    "Declining":"#e74c3c",
}

for row in trend_df.itertuples():
    cid      = row.cluster
    stage    = row.lifecycle_stage
    color    = color_map[stage]
    group    = meta[meta["cluster"] == cid]
    curve    = activity_curve(group)

    fig, ax1 = plt.subplots(figsize=(11, 4))

    # Activity curve (main)
    ax1.fill_between(quarters, curve.values, alpha=0.25, color=color)
    ax1.plot(quarters, curve.values, color=color, linewidth=2,
             label="Active posts (overlap count)")
    ax1.set_ylabel("Simultaneously Active Posts", fontsize=10)
    ax1.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Mark peak
    peak_q = pd.Timestamp(row.peak_quarter)
    peak_v = int(curve[quarters == peak_q].iloc[0]) if any(quarters == peak_q) else 0
    ax1.axvline(peak_q, color="black", linewidth=1.2,
                linestyle="--", alpha=0.6, label=f"Peak: {row.peak_quarter}")
    ax1.annotate(
        f"PEAK\n{row.peak_quarter}",
        xy=(peak_q, peak_v),
        xytext=(15, 10), textcoords="offset points",
        fontsize=8, color="black",
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
    )

    # Engagement rate overlay
    eng_ts = (
        group.assign(
            qtr=group["ts_start"].dt.to_period("Q").dt.to_timestamp()
        )
        .groupby("qtr")["engagement_rate"].mean()
        .reindex(quarters, fill_value=np.nan)
    )
    ax2 = ax1.twinx()
    ax2.plot(quarters, eng_ts.values, color="gray", linewidth=1,
             linestyle="--", alpha=0.6, label="Avg engagement rate (%)")
    ax2.set_ylabel("Avg Engagement Rate (%)", fontsize=9, color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")

    ax1.set_title(
        f"Cluster {cid}  |  {row.dominant_category}  |  {stage}\n"
        f"Peak: {row.peak_quarter}  |  "
        f"Window: {row.trend_window_start} → {row.trend_window_end}  |  "
        f"Engagement: {row.mean_engagement_rate:.2f}%  |  "
        f"Viral: {row.viral_rate:.1%}  |  n={row.total_posts:,}",
        fontsize=9,
    )
    ax1.set_xlabel("Quarter")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(
        GRAPHS_DIR / f"cluster_{cid:03d}_{stage.lower()}.png",
        dpi=120,
    )
    plt.close()

print(f"Per-cluster graphs saved → {GRAPHS_DIR.resolve()}")

# ----------------------------------------------------------------------
# 5. Summary chart — top 20 Rising clusters
# ----------------------------------------------------------------------
rising = trend_df[trend_df["lifecycle_stage"] == "Rising"].nlargest(
    20, "mean_engagement_rate"
)

if len(rising) > 0:
    fig, ax = plt.subplots(figsize=(12, 7))
    labels  = [
        f"C{r.cluster} · {r.dominant_category} (peak {r.peak_quarter})"
        for r in rising.itertuples()
    ]
    bars = ax.barh(
        labels,
        rising["mean_engagement_rate"],
        color="#2ecc71", edgecolor="white", linewidth=0.5,
    )
    ax.bar_label(bars, fmt="%.2f%%", padding=4, fontsize=8)
    ax.set_xlabel("Mean Engagement Rate (%)")
    ax.set_title(
        "Top 20 Rising Visual Trends\n"
        "Ranked by Mean Engagement Rate — Peak Quarter Shown",
        fontsize=12,
    )
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "trend_summary.png", dpi=150)
    plt.close()
    print("Summary chart saved → trend_summary.png")

# ----------------------------------------------------------------------
# 6. Console summary
# ----------------------------------------------------------------------
print("\nTop 5 Rising clusters:")
print(
    trend_df[trend_df["lifecycle_stage"] == "Rising"]
    [["cluster", "dominant_category", "peak_quarter",
      "mean_engagement_rate", "viral_rate", "total_posts"]]
    .head()
    .to_string(index=False)
)

print("\nTop 5 Declining clusters:")
print(
    trend_df[trend_df["lifecycle_stage"] == "Declining"]
    [["cluster", "dominant_category", "peak_quarter",
      "mean_engagement_rate", "viral_rate", "total_posts"]]
    .head()
    .to_string(index=False)
)

print("\nDone. New files in trendlens_outputs/:")
print("  - trend_metrics.csv")
print("  - trend_summary.png")
print(f"  - trend_graphs/  ({len(trend_df)} cluster graphs)")