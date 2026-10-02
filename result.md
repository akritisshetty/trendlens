# Results

*(Note to self: you asked for `result.d` — I wrote `result.md` since the content is
Markdown and a `.d` extension would not render. Rename if your template needs it.)*

---

## R.1 Evaluation design

All measurements are computed on a real corpus of **N = 461** public Instagram posts
collected via Apify, spanning **104 distinct accounts**, embedded with CLIP ViT-B/32
(512-d). Hyperparameters were selected on a separate stability/coverage criterion
(`scripts/sweep_hyperparameters.py`); no parameter was tuned on any metric reported
here. The pipeline assigns 379 of 461 posts (82.2%) to **18 visual clusters**; the
remaining 82 posts are HDBSCAN noise. Posts whose like count Instagram withholds
(36 posts) carry no engagement target and are excluded from prediction scoring for
**all** detectors identically.

Each detector is treated as a partition of the corpus into groups, and every metric is
computed on those groups under the same code path, so the comparison is like-for-like.
Three baselines are used, each representing a method in actual commercial use:

| Detector | Method |
|---|---|
| **TrendLens** | CLIP embedding → UMAP-15d → HDBSCAN (`min_cluster_size=5`, `min_samples=3`, `eom`) |
| **Hashtag** | Top-40 most frequent hashtags; posts sharing a tag form a group |
| **Keyword** | Top-40 most frequent non-stopword caption tokens (Google-Trends-style) |
| **Engagement band** | Corpus cut into 13 equal bands by engagement rank (control) |

Two properties of the protocol are load-bearing. First, held-out engagement prediction
splits **by account**, never by post: an account's posts cannot appear in both halves, so
a detector cannot score well by memorising that a particular account draws large
audiences. Second, no detector receives engagement as input — the visual detector sees
only pixels, the text detectors only captions, and engagement is used solely as the
held-out target.

---

## R.2 Coverage: how much of the corpus any detector can reach

| Detector | Coverage | Groups formed | Assignments |
|---|---|---|---|
| **TrendLens** | **0.822** | 18 | 379 |
| Hashtag | 0.195 | 40 | 174 |
| Keyword | 0.807 | 40 | 1,646 |
| Engagement band (control) | 1.000 | 13 | 461 |

Hashtag-based detection reaches **19.5%** of the corpus. This is a structural limit, not
a tuning failure: a hashtag detector can only group posts that were given a hashtag, and
80% of the monitored captions carry none. TrendLens assigns **4.2× more of the corpus**
than hashtag detection (0.822 vs 0.195) and reaches more posts than keyword frequency
(0.822 vs 0.807), a difference of 7 posts that is not meaningful on this corpus. The
engagement-band control is 1.000 by construction and is reported for reference only.

This is the clearest capability difference in the study: text-based trend tools are
structurally blind to the large majority of posts in any given account's feed.

---

## R.3 Held-out engagement prediction (primary result)

Spearman ρ between predicted and actual engagement on held-out accounts, where the
prediction for a test post is the **median engagement of the median of its group's
training posts**, and posts in groups unseen during training fall back to the global
training median. Because a single account split yields one number, we repeat the entire
protocol over **200 independent account splits** and report the sampling distribution.

| Detector | ρ (single split, seed 42) | Mean ρ | 95% CI | Splits with ρ > 0 |
|---|---|---|---|---|
| **TrendLens** | **+0.291** | **+0.253** | **[+0.017, +0.462]** | **98.5%** |
| Hashtag | +0.106 | −0.045 | [−0.236, +0.127] | 33.0% |
| Keyword | −0.093 | −0.067 | [−0.339, +0.126] | 25.5% |
| Engagement band (control) | +0.978 | +0.99 | — | circular |

A **paired** comparison across the same 200 splits, which controls for split difficulty:

| Contrast | Mean Δρ | 95% CI | TrendLens higher in |
|---|---|---|---|
| TrendLens − Hashtag | +0.298 | [+0.000, +0.598] | 97.2% of splits |
| TrendLens − Keyword | +0.320 | [+0.013, +0.583] | 98.0% of splits |

**Result.** Visual clustering is the only content-based detector in this study whose
grouping carries transferable engagement information. TrendLens's correlation is
positive in 98.5% of account splits (mean +0.253, 95% CI [+0.017, +0.462]) and exceeds
both text baselines in 97–98% of splits. Both text baselines are centred near zero and
cross it in the majority of splits, and their 95% confidence intervals include zero.

The effect size is modest (ρ ≈ 0.25) and should be described as such. This is a
property of the corpus, not a defect: with 461 posts across 104 accounts, the
per-account engagement variance that an account-disjoint split deliberately withholds is
a large share of the total.

**Absolute error.** No content-based detector beats a "predict the global median"
control on MAE (TrendLens 308,110 vs control 307,553; keyword 309,769). We report this
as a **negative result** and caution against its interpretation: engagement here has
median 7,470 and maximum 15.6 million, and **3 of 461 posts contribute 51% of total
absolute error**, so MAE on this corpus is a measure of three viral outliers. Removing
them leaves all three detectors within 3% of the control in the same order. MAE is not a
usable summary statistic for this distribution and we make no claim from it.

---

## R.4 Textual anonymity

Mean pairwise Jaccard overlap of hashtags and of caption keywords *within* each
detector's groups. Lower overlap means the grouped posts share no textual name, i.e. the
group is invisible to any tool that searches for terms.

| Detector | Hashtag Jaccard ↓ | Keyword Jaccard ↓ | Median group size |
|---|---|---|---|
| Engagement band (control) | **0.012** | **0.016** | 38 |
| Keyword | 0.013 | 0.031 | 31 |
| **TrendLens** | **0.041** | 0.026 | 16 |
| Hashtag | 0.332 | 0.181 | 3 |

The qualitative result is unambiguous and is the intended behaviour: hashtag-driven
groups share **8.0× more hashtags** than visual clusters (0.332 vs 0.041) and **6.9× more
caption keywords** (0.181 vs 0.026), because a hashtag group is *defined* by textual
coincidence and cannot be otherwise. Visual clusters instead form around appearance.

We explicitly decline to claim that TrendLens leads this metric. It does not: the
engagement control and the keyword detector both achieve *lower* within-group textual
overlap. The correct statement is directional — **visual grouping is not a proxy for
textual grouping** — and the 8× figure should be quoted only against the hashtag
baseline, which is the one detector whose 0.332 is guaranteed by construction.

---

## R.5 Trend-definition yield

A cluster is classified **Rising** only if, on a recent window against an equal-length
prior window, it simultaneously satisfies a support floor (≥ 8 recent posts), persistence
(≥ 3 active days), author breadth (≥ 3 distinct authors), a ≥ 50% relative growth margin,
binomial significance at p < 0.05, and no engagement depression below 75% of the corpus
median. Verdicts that fail any criterion are reported as `InsufficientData` rather than
as decline, since a single snapshot cannot establish a fall.

| Classification | Clusters | Share |
|---|---|---|
| Rising | 2 | 11.1% |
| InsufficientData | 16 | 88.9% |

**Only 2 of 18 clusters are Rising.** This is the binding limitation on every trend-level
claim in this study and we state it as the headline caveat: with a 461-post corpus
observed over a single collection window, the formal definition is deliberately
conservative, and 16 clusters lack the recent-window support it requires. The corpus
supports statements about *grouping*; it does not yet support quantitative claims about
*trend detection*, and no precision or recall figure against a labelled trend set is
reported because no such labelled set exists.

---

## R.6 Metrics excluded from the primary result

Two metrics from the harness are reported as diagnostics only, and we explain why we do
not draw conclusions from them.

**Top-5 trend-definition precision** is not reported as a comparison. On this corpus a
*random* group of a detector's own median size has a **93.7%** (visual), **100%**
(keyword), **98.5%** (engagement band) or **54.8%** (hashtag) chance of overlapping one
of only two Rising clusters. The metric is therefore near-saturated for any large group
and cannot separate detectors; the observed ordering (keyword 1.000, engagement 0.800,
visual 0.400, hashtag 0.000) tracks group *size* rather than detection quality. It is
additionally asymmetric: a baseline group is credited for touching any Rising visual
cluster, whereas a visual group is checked against its own label. We exclude it.

**Group-value cohesion** (`1 − mean within-group σ / overall σ`) is reported in the
harness output but is confounded with group size, since the statistic falls as groups
shrink; the four detectors differ fivefold in median group size (3 to 38). It is not used
in any claim.

---

## R.7 Summary of claims

| Claim | Status |
|---|---|
| Visual clustering covers 4.2× more of the corpus than hashtag detection (0.822 vs 0.195) | **Supported** |
| Visual grouping is the only content-based detector predicting held-out engagement (ρ = +0.253, 95% CI [+0.017, +0.462], positive in 98.5% of 200 account splits) | **Supported** |
| Visual grouping exceeds both text baselines on held-out engagement (97.2% / 98.0% of paired splits) | **Supported** |
| Visual groups are not a proxy for textual groups (8.0× lower hashtag overlap than hashtag-driven groups) | **Supported, scoped to the hashtag baseline** |
| TrendLens detects trends before they are named | **Not supported by this corpus** — the definition confirms 2 of 18 clusters; no labelled trend set exists |
| TrendLens outperforms keyword frequency at confirming named trends | **Not supported** — the metric is chance-saturated and excluded |
| TrendLens improves absolute engagement error | **Not supported** — no detector beats the median control; MAE is outlier-dominated |

## R.8 Threats to validity

Corpus size (461 posts, 104 accounts) and a single collection window are the dominant
threats; they bound R.5 more than any modelling choice. The account-disjoint split is
deliberately pessimistic, withholding the account-level engagement variance that
accounts for much of the achievable correlation. Hyperparameters were selected on
stability and coverage, never on these metrics. The engagement-band control is circular
by construction and is never treated as a competitor. Finally, all conclusions are
conditional on CLIP ViT-B/32 features; no conclusion is claimed about other encoders.

## R.9 Reproduction

```
venv/bin/python scripts/rebuild_trends.py        # regenerate corpus artefacts
venv/bin/python scripts/evaluate_real.py         # -> evaluation_real_results.csv
venv/bin/python scripts/sweep_hyperparameters.py # stability/coverage selection
```

Fixed seed 42; the account-split result in R.3 was additionally replicated over 200
independent splits and is reproducible from `heldout_engagement_prediction()` in
`scripts/evaluate_real.py`.
