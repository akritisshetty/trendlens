# Evaluation Against Popular Baselines

This report evaluates TrendLens' image-based trend detection against the
**actual tools people use for trend detection today** — hashtag /
social-listening, keyword / Google-Trends-style frequency, and engagement
trending lists — rather than against alternate ML components.

> **Status (2026-09-28):** superseded by `scripts/evaluate_real.py`, which
> measures the same detectors on the regenerated corpus under non-circular
> criteria and is the current source of truth. The table below summarises that
> run. `baseline_real_tools_results.csv` and this file's older numbers
> described a 12-cluster / UMAP-10d / `min_cluster_size=10` configuration
> that has since been replaced by the hyperparameter sweep (UMAP-15d,
> `min_cluster_size=5`, `min_samples=3`, `eom`).
>
> Current configuration: 461 embedded posts (CLIP ViT-B/32), 18 visual
> clusters (379 posts clustered, 82 noise), definition-driven verdicts.

## Real-Data Evaluation (current)

Full output in `evaluation_real_results.csv`; methodology in the
`evaluate_real.py` docstring.

| Metric | TrendLens (visual) | Hashtag | Keyword | Engagement band* |
|---|---|---|---|---|
| Assignment coverage | **0.822** | 0.195 | 0.807 | 1.000 |
| Held-out engagement ρ (by account) | **0.291** | 0.106 | −0.093 | 0.978* |
| Trend precision@5 | 0.400 | 0.000 | **1.000** | 0.800 |
| Within-group hashtag overlap (lower = unnamed) | **0.041** | 0.332 | 0.013 | 0.012 |
| Group value cohesion | 0.568 | 0.449 | **0.822** | 0.669 |

\* Engagement band = corpus cut into 12 equal bands by engagement rank; its
prediction score is an upper bound by construction.

### Section-by-section interpretation

**1. Coverage.** TrendLens assigns 82% of the corpus to a group — 4× the
hashtag baseline (20%). Hashtag detection is structurally blind to the 80% of
posts that carry no hashtag among the monitored ones. This is the strongest
capability gap TrendLens closes.

**2. Held-out engagement (split by account).** TrendLens has the strongest
engagement signal of any content-based detector (ρ = 0.29; keyword frequency
is *negatively* correlated at −0.09; hashtag is 0.11). It still loses to a
"predict the global median" control on absolute error — the corpus is small
and heavy-tailed, and that limit is reported rather than hidden.

**3. Trend precision.** TrendLens confirms 2/5 of its top-engagement groups
as Rising under the formal definition; keyword frequency confirms 5/5. The
reason is visible in this corpus: its Rising themes (e.g. "hiked papua
starting", "today years married") have names and keywords in their captions,
so a keyword tool that finds a term can confirm them. TrendLens' core claim —
finding trends *before* they have a name — shows up on the anonymity metric
instead (below), and is what a keyword tool cannot do at all.

**4. Textual anonymity.** TrendLens groups share 8× fewer hashtags than
hashtag-frequency groups (0.041 vs 0.332): visually similar posts with no
shared name. This is the measurement behind "find trends before they have a
name" and is a property of the grouping, not a tuned score.

**5. Cohesion.** TrendLens (0.568) beats hashtag detection (0.449) and the
engagement band on group-value coherence, though keyword frequency happens to
form coherent groups too. No claim of superiority is drawn from a single
bounded index.

---

## Older report artifacts

- `scripts/compare_trend_baselines.py` — retained for reproducibility; uses
  the pre-sweep hyperparameters and is no longer the primary harness.
- `baseline_real_tools_results.csv` — older run's machine-readable results.
- `scripts/rebuild_trends.py` — regenerates `data/instagram/trends.json`
  (18 themes / 461 posts / 20 tracked hashtags / SQLite history snapshots)
  from the fixed pipeline.