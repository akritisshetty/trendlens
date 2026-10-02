# Results

All tables are generated from measured runs. Primary source: `evaluation_real_results.csv`
(regenerate with `venv/bin/python scripts/evaluate_real.py`). Every value in every table
below was reproduced from the corpus on 2026-09-29.

---

## 1. Corpus

| Property | Value |
|---|---|
| Posts | 461 |
| Distinct accounts | 104 |
| Encoder | CLIP ViT-B/32 |
| Embedding dim | 512 |
| Posts with hidden like count (excluded from scoring) | 36 |
| Posts scored | 425 |
| Engagement target | `likes + 3 × comments` |
| Median engagement | 7,470 |
| Max engagement | 15,649,371 |
| Account split | by author, 1/3 held out |
| Seed | 42 |

---

## 2. Detector comparison — headline table

| Metric | TrendLens (visual) | Hashtag | Keyword | Engagement band |
|---|---|---|---|---|
| Assignment coverage | **0.822** | 0.195 | 0.807 | 1.000 |
| Held-out engagement ρ | **0.291** | 0.106 | −0.093 | 0.978 |
| MAE vs median control | 308,110 | 307,453 | 309,769 | 295,734 |
| Beats median control (MAE) | No | Yes | No | Yes |
| Trend precision@5 | 0.400 | 0.000 | **1.000** | 0.800 |
| Claims confirmed @5 | 2/5 | 0/5 | 5/5 | 4/5 |
| Hashtag Jaccard ↓ | 0.041 | 0.332 | **0.013** | **0.012** |
| Keyword Jaccard ↓ | **0.026** | 0.181 | 0.031 | 0.016 |
| Group value cohesion | 0.568 | 0.449 | **0.822** | 0.669 |

Median-engagement control MAE = 307,553. Engagement band is circular by construction and is
an upper bound, not a competitor.

---

## 3. Coverage

| Detector | Coverage | Groups formed | Assignments | Median group size |
|---|---|---|---|---|
| TrendLens (visual) | 0.822 | 18 | 379 | 16 |
| Hashtag | 0.195 | 40 | 174 | 3 |
| Keyword | 0.807 | 40 | 1,646 | 31 |
| Engagement band | 1.000 | 13 | 461 | 38 |

| Comparison | Value |
|---|---|
| Coverage ratio, TrendLens vs hashtag | 4.21× |
| Coverage difference, TrendLens vs keyword | +0.015 (7 posts) |

---

## 4. Held-out engagement across 200 account splits

| Detector | Mean ρ | 95% CI | Splits with ρ > 0 |
|---|---|---|---|
| TrendLens (visual) | **+0.253** | **[+0.017, +0.462]** | 98.5% (197/200) |
| Hashtag | −0.045 | [−0.236, +0.127] | 33.0% |
| Keyword | −0.065 | [−0.341, +0.130] | 28.0% |
| Engagement band | +0.975 | [+0.895, +0.996] | 100.0% |

### Paired contrasts (same split for both detectors)

| Contrast | Mean Δρ | 95% CI | TrendLens higher in | Splits compared |
|---|---|---|---|---|
| TrendLens − Hashtag | +0.298 | [+0.000, +0.598] | 97.2% | 179 |
| TrendLens − Keyword | +0.318 | [+0.013, +0.584] | 98.0% | 200 |

Hashtag returns an undefined ρ on 21 of 200 splits (fewer than 3 train members per group);
those splits are excluded from the paired contrast.

---

## 5. MAE is outlier-dominated

| Rank of post by absolute error | Share of total absolute error |
|---|---|
| Top 1 | 22.5% |
| Top 3 | **51.6%** |
| Top 5 | 68.2% |

| Detector | MAE | vs control (307,553) |
|---|---|---|
| TrendLens (visual) | 308,110 | +0.2% worse |
| Keyword | 309,769 | +0.7% worse |
| Hashtag | 307,453 | 0.03% better |
| Engagement band | 295,734 | 3.8% better |

No content-based detector clears the median control. MAE is not a usable summary statistic
on this corpus.

---

## 6. Trend-definition yield

| Classification | Clusters | Share |
|---|---|---|
| Rising | 2 | 11.1% |
| InsufficientData | 16 | 88.9% |

| Property | Value |
|---|---|
| Rising cluster ids | 4, 14 |
| Posts inside Rising clusters | 39 of 461 |
| Chance a random group of median size touches a Rising cluster — visual (n=16) | 76.3% |
| Chance — keyword (n=31) | 94.1% |
| Chance — engagement band (n=38) | 97.0% |
| Chance — hashtag (n=3) | 23.3% |

A cluster is Rising only if it clears support floor (≥ 8 recent posts), persistence
(≥ 3 active days), author breadth (≥ 3 authors), ≥ 50% relative growth, binomial p < 0.05,
and no engagement depression below 75% of corpus median.

---

## 7. Chance baseline for trend precision@5

| Detector | Observed precision | Chance precision at its own median group size |
|---|---|---|
| TrendLens (visual) | 0.400 | 76.3% |
| Hashtag | 0.000 | 23.3% |
| Keyword | 1.000 | 94.1% |
| Engagement band | 0.800 | 97.0% |

Observed precision tracks group size, not detection quality. The metric is excluded from
claims.

---

## 8. Component benchmarks

Source: `baseline_comparison_results.csv`. **Note:** this file was produced under the
pre-sweep configuration (UMAP-10d, `min_cluster_size=10`, 3 clusters), not the 18-cluster
UMAP-15d configuration used in sections 2–7. Kept for component reference only.

### 8.1 Embedding models

| Model | Dim | Throughput (img/s) | Latency (ms) | Intra-cluster coherence |
|---|---|---|---|---|
| CLIP ViT-B/32 (ours) | 512 | **10.5** | **165.0** | 0.4957 |
| CLIP ViT-L/14 | 768 | 0.5 | 2115.4 | 0.4636 |
| DINOv2 ViT-B/14 | 768 | 1.8 | 491.0 | 0.0837 |
| ResNet50 | 2048 | 5.5 | 234.8 | **0.6361** |

Coherence measured on 134 labelled real posts.

### 8.2 Dimensionality reduction

| Method | Time (s) | Clusters | Noise % | Silhouette | ARI | NMI | V-measure |
|---|---|---|---|---|---|---|---|
| UMAP (ours) | 0.52 | 3 | **0.1053** | **0.5793** | **1.0000** | **1.0000** | **1.0000** |
| PCA | **0.01** | 2 | 0.5066 | 0.3174 | 0.2123 | 0.3042 | 0.3042 |
| t-SNE | 1.03 | 2 | 0.6316 | 0.6068 | 0.1806 | 0.3859 | 0.3859 |

ARI/NMI are measured against HDBSCAN output as the reference labelling.

### 8.3 Clustering algorithms

| Method | Time (s) | Clusters | Noise % | Silhouette | ARI | NMI | Balance ratio |
|---|---|---|---|---|---|---|---|
| HDBSCAN (ours) | 0.009 | 3 | **0.1053** | **0.5793** | **1.0000** | **1.0000** | **0.5536** |
| KMeans (k=4) | 0.075 | 4 | 0.0 | 0.4769 | 0.7014 | 0.7309 | 0.5000 |
| KMeans (k=5) | 0.025 | 5 | 0.0 | 0.4171 | 0.5941 | 0.6819 | 0.3333 |
| KMeans (k=8) | 0.04 | 8 | 0.0 | 0.3748 | 0.4585 | 0.6374 | 0.3000 |
| DBSCAN (eps=0.5) | **0.005** | 6 | 0.1382 | 0.3628 | 0.6983 | 0.7312 | 0.1020 |
| DBSCAN (eps=1.0) | 0.003 | 1 | 0.0 | N/A | — | — | 1.0000 |
| Agglomerative (k=4) | 0.004 | 4 | 0.0 | 0.4704 | 0.6966 | 0.7310 | 0.5882 |

### 8.4 Trend scoring methods

| Method | Spearman vs engagement | Agreement with TrendLens | Precision@2 | Precision@3 |
|---|---|---|---|---|
| TrendLens (ours) | **0.500** | **1.000** | **0.500** | 0.3333 |
| Simple growth | −0.866 | −0.866 | 0.0000 | **0.3333** |
| MA slope | −0.500 | 0.500 | 0.0000 | **0.3333** |
| Linear slope | −0.500 | 0.500 | 0.0000 | **0.3333** |
| Exp. weighted growth | −0.500 | −1.000 | **0.500** | **0.3333** |
| Text tags baseline | −0.866 | −0.866 | 0.0000 | 0.0000 |

### 8.5 Retrieval encoders

> **Superseded — do not quote this table.** It was measured on a 4-chunk corpus
> (`Corpus size 4` below), which cannot separate retrieval encoders; with 4 chunks
> Recall@1 moves in steps of 0.25. The number that counts is the 18-chunk / 72-query
> measurement in `text_model_selection_results.csv` and §Algorithm Benchmarking of the
> README, which promoted `BAAI/bge-base-en-v1.5` (R@1 0.43, MRR 0.59) over this
> table's winner. Kept only as a record of the earlier run.

| Model | Dim | P@1 | R@1 | Hit@1 | P@3 | R@3 | Hit@3 | MRR |
|---|---|---|---|---|---|---|---|---|
| MiniLM-L6-v2 (ours) | 384 | **0.750** | **0.2812** | **0.750** | **0.7083** | **0.7812** | **1.000** | **0.8750** |
| all-mpnet-base-v2 | 768 | 0.375 | 0.1146 | 0.375 | 0.6250 | 0.6771 | 1.000 | 0.6458 |
| MiniLM-L12-v2 | 384 | 0.375 | 0.1146 | 0.375 | 0.6667 | 0.7396 | 1.000 | 0.6667 |

Corpus size 4.

### 8.6 Pipeline latency

> **Stale.** Measured on a 152-sample corpus with UMAP 10-d. Current figures are in the
> README hardware notes; BLIP captioning is now the dominant stage (~30 min) because the
> captioner was upgraded to `blip-image-captioning-large`.

| Stage | Time |
|---|---|
| CLIP embedding | 210.0 ms/image |
| UMAP 10d (152 samples) | 0.43 s |
| HDBSCAN | 0.01 s |
| Trend scoring | 0.05 s |
| Total (UMAP + HDBSCAN + trends) | 0.48 s |

---

## 9. Reproduction

| Command | Output |
|---|---|
| `venv/bin/python scripts/rebuild_trends.py` | corpus artefacts |
| `venv/bin/python scripts/evaluate_real.py` | `evaluation_real_results.csv` |
| `venv/bin/python scripts/sweep_hyperparameters.py` | stability/coverage selection |
| `venv/bin/python baseline_comparison.py` | `baseline_comparison_results.csv` |
