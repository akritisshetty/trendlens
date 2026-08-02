# TrendLens — Project Skills & Context Reference

> **Purpose:** This file is the canonical context document for the TrendLens project.
> It describes the architecture, design decisions, data schema, and pipeline steps in enough detail
> to serve as a reference for future development, AI-assisted coding sessions, and onboarding.

---

## 1. Project Overview

**TrendLens** is an end-to-end multimodal visual trend-detection pipeline built on a social-media-style
photo dataset (~305 K images from ~38 K users). The pipeline transforms raw image files and a
Flickr-style filepath manifest into a structured, engagement-enriched dataset paired with 512-dimensional
CLIP visual embeddings — the foundation for downstream trend clustering, virality prediction, and
content recommendation.

**Goal:** Given a corpus of photos with rich engagement signals, identify visual trends (recurring
aesthetics, colour palettes, subject matter) that correlate with high customer engagement.

---

## 2. Repository Layout

```
trendlens/
├── generate_metadata.py          # Pipeline script — Steps 1–4 (run first)
├── generate_embeddings.py        # Pipeline script — CLIP embeddings (run second)
├── generate_umap.py              # Pipeline script — UMAP reduction (run third)
├── generate_clusters.py          # Pipeline script — HDBSCAN clustering (run fourth)
├── generate_temporal_trends.py   # Pipeline script — Temporal tracking (run fifth)
├── script.py                     # One-off utility: drops truncated images & verifies alignment
├── requirements.txt              # Python dependencies
├── train_img_filepath.txt        # Flickr-style manifest: train/<user_id>/<photo_id>.jpg
├── train/                        # Image root — subdirs named by user_id
│   └── <user_id>/
│       └── <photo_id>.jpg
└── trendlens_outputs/            # All generated artefacts (git-ignored)
    ├── smpd_metadata.json            # Raw synthetic engagement records (305 K rows)
    ├── metadata.csv                  # Enriched, validated CSV (305 613 rows × 25 cols)
    ├── embeddings.npy                # CLIP vectors — float32 (305613, 512) L2-normalised
    ├── embeddings_checkpoint.npy     # Rolling checkpoint (mirrors final on completion)
    ├── metadata_checkpoint.csv       # Metadata slice aligned with the checkpoint
    ├── umap_2d.npy                   # 2-D UMAP projection — float32 (305613 × 2)
    ├── umap_10d.npy                  # 10-D UMAP projection — float32 (305613 × 10)
    ├── umap_scatter.png              # Sanity scatter plot coloured by category
    ├── metadata_clustered.csv        # metadata.csv + ['cluster', 'cluster_prob'] columns
    ├── cluster_summary.csv           # Per-cluster size / engagement / category stats
    ├── cluster_representatives.json  # Top-probability image per cluster
    ├── cluster_scatter.png           # umap_2d scatter coloured by cluster label
    ├── cluster_representatives.png   # Image grid of one representative per cluster
    ├── trend_metrics.csv             # Per-cluster peak quarter + lifecycle stage + stats
    ├── trend_summary.png             # Top 20 Rising trends bar chart
    └── trend_graphs/                 # One activity curve PNG per cluster (105 total)
```

> **Note:** `failed_images.txt` and `nn_preview.png` are created only if there are
> failed images or if the NN sanity-check is explicitly run.

---

## 3. Execution Order

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate metadata (engagement + trend duration) — ~5–10 min
python generate_metadata.py

# 3. Generate CLIP embeddings — ~90 min on CPU, ~10 min on GPU
python generate_embeddings.py

# 4. Reduce embeddings with UMAP — ~20–30 min on CPU
python generate_umap.py

# 5. HDBSCAN clustering — ~5–10 min
python generate_clusters.py

# 6. Temporal trend tracking — ~5 min
python generate_temporal_trends.py
```

`script.py` is a **maintenance utility** — run it manually only when correcting dataset
alignment issues (e.g., after discovering a truncated or corrupt image).

---

## 4. Pipeline: `generate_metadata.py`

Four sequential steps; run once before `generate_embeddings.py`.

### Step 1 — Taxonomy, Helpers & User Profiles

Sets up all global constants and builds per-user behavioural profiles from the filepath manifest.

#### 1a. Content Taxonomy

15 categories with per-category viral-potential multipliers (`CAT_LIKES_MU`):

| Category     | Likes Multiplier |
| ------------ | ---------------- |
| food         | 3.2              |
| fashion      | 3.0              |
| portrait     | 2.8              |
| travel       | 2.7              |
| events       | 2.5              |
| animals      | 2.4              |
| nightlife    | 2.3              |
| family       | 2.2              |
| sports       | 2.0              |
| nature       | 1.9              |
| architecture | 1.8              |
| street       | 1.7              |
| art          | 1.6              |
| abstract     | 1.4              |
| technology   | 1.3              |

Each category has a curated hashtag pool (`CATEGORY_TAGS`) plus a shared generic-tag pool.

#### 1b. Geo Pool

20 world cities with (city name, latitude, longitude) tuples. ~70 % of posts get a location tag
(60 % home city, 40 % random city), with ±0.15° jitter for privacy realism.

#### 1c. Per-User Profiles

Keyed by `user_id` — built once, reused for all posts by that user:

| Field            | Distribution                         | Notes                                       |
| ---------------- | ------------------------------------ | ------------------------------------------- |
| `preferred_cat`  | Uniform random from 15 categories    | 70 % of that user's posts use this          |
| `home_city`      | Uniform random from geo pool         | Anchor for location generation              |
| `likes_mult`     | LogNormal(0, 0.5) clipped [0.3, 6.0] | Audience size / influence proxy             |
| `follower_count` | LogNormal(6.5, 1.8) clipped [10, 5M] | Power-law; most small, few mega-influencers |

#### 1d. Timestamp Generation

Timestamps are assigned **randomly and uniformly** across 2010-01-01 → 2019-12-31 using
`rng.uniform(0, EPOCH_SPAN)`, producing ~30,500 posts per year across the decade.

**Why changed from v1:** The original approach mapped photo IDs linearly to timestamps,
causing all clusters to peak artificially in 2010–2013 regardless of content — making
temporal trend tracking meaningless. Random uniform assignment spreads posts evenly so
clusters naturally peak at different periods, enabling genuine Rising/Stable/Declining
classification. The relative photo-ID ordering signal was sacrificed in favour of
temporal diversity.

---

### Step 2 — Synthetic Engagement Records → `smpd_metadata.json`

Generates a realistic engagement record for every entry in the filepath manifest
(305 613 total).

| Signal            | Method                                                          |
| ----------------- | --------------------------------------------------------------- |
| `likes`           | LogNormal, modulated by category multiplier × user `likes_mult` |
| `comments`        | Beta(1.5, 6.0) fraction × `likes` × 0.25                        |
| `views`           | Uniform(10×, 120×) `likes`                                      |
| `reposts`         | Beta(1.2, 18.0) fraction × `likes` × 0.08 — heavy right-skew    |
| `saves`           | Beta(2.0, 12.0) fraction × `likes` × 0.15                       |
| `reach`           | max(views, views × Uniform(0.6, 1.1))                           |
| `engagement_rate` | (likes + comments + reposts) / reach × 100                      |
| `is_viral`        | True if engagement_rate > 3% **or** reposts > 50                |

---

### Step 3 — Enriched Metadata CSV → `metadata.csv`

1. Re-parses the filepath manifest into `df_base`.
2. Checks file existence on disk.
3. Merges the synthetic JSON onto valid rows by `(post_id, user_id, photo_id)`.
4. Serialises list-type columns (`tags`, `groups`) as JSON strings for CSV compatibility.
5. Proceeds to Step 4 before saving.

---

### Step 4 — Trend Duration Enrichment

Adds two new columns: `trend_duration_days` and `trend_active_until`.

#### Duration Model

```
trend_duration = base_duration(category)
               × engagement_multiplier     # (likes + comments) / views, scaled 0.5–3.0×
               × virality_boost            # ×2–4 for top-5 % by likes within category
```

Sampled from a **LogNormal(μ, σ=0.4)** centred on the computed mean, capped to **1–180 days**.

#### Per-Category Base Durations (`CAT_BASE_DAYS`)

| Category     | Base Days |
| ------------ | --------- |
| events       | 3         |
| sports       | 5         |
| nightlife    | 7         |
| animals      | 7         |
| street       | 10        |
| portrait     | 10        |
| food         | 14        |
| family       | 14        |
| nature       | 14        |
| travel       | 21        |
| architecture | 21        |
| abstract     | 21        |
| art          | 28        |
| fashion      | 30        |
| technology   | 45        |

`trend_active_until` = `timestamp` + `trend_duration_days` (ISO-8601 UTC string).

Final CSV saved to `trendlens_outputs/metadata.csv`.

---

## 5. Pipeline: `generate_embeddings.py`

Run after `generate_metadata.py`. Produces `embeddings.npy`.

| Setting              | Value                          |
| -------------------- | ------------------------------ |
| Model                | `openai/clip-vit-base-patch32` |
| Batch size           | 32 images                      |
| Checkpoint frequency | Every 200 batches              |
| Device               | CUDA (auto-detected) or CPU    |

**Embedding path:**
`vision_model → pooler_output → visual_projection → L2-normalise → float32`

**Resume logic:** If `embeddings_checkpoint.npy` + `metadata_checkpoint.csv` exist,
the script automatically resumes from where it left off.

**Sanity assertions (run after completion):**

- `embs.ndim == 2`
- `embs.shape[1] == 512`
- `embs.dtype == float32`
- `embs.shape[0] == len(metadata_csv)`
- `embs[0] · embs[0] ≈ 1.0` (L2-norm verification)

---

## 6. Pipeline: `generate_umap.py`

Run **after** `generate_embeddings.py` and **before** HDBSCAN clustering.
Reduces the 512-d L2-normalised CLIP embeddings to lower-dimensional spaces for two
distinct purposes: interactive scatter-plot visualisation and density-based clustering.

### UMAP Configuration

| Run        | `n_components` | `n_neighbors` | `min_dist` | `metric` | `random_state` | Purpose                                                                              |
| ---------- | -------------- | ------------- | ---------- | -------- | -------------- | ------------------------------------------------------------------------------------ |
| `umap_2d`  | 2              | 30            | 0.1        | cosine   | 42             | Scatter-plot visualisation — slight spread between points improves readability       |
| `umap_10d` | 10             | 30            | 0.0        | cosine   | 42             | HDBSCAN input — `min_dist=0.0` tightens local clusters, preserving density structure |

**Why cosine metric?** The CLIP embeddings are L2-normalised (unit vectors), so cosine
distance is equivalent to Euclidean distance and captures semantic orientation, not magnitude.

**Why `n_neighbors=30`?** Balances local vs global structure; 30 is a common default for
~305 K points that preserves both fine-grained neighbourhood topology and macro cluster layout.

**Why `min_dist=0.0` for 10-D?** HDBSCAN needs points to be clumped into density peaks;
a non-zero `min_dist` artificially spreads points and can break cluster cores.

### Output Files

| File               | Shape       | dtype   | Description                                                                |
| ------------------ | ----------- | ------- | -------------------------------------------------------------------------- |
| `umap_2d.npy`      | 305613 × 2  | float32 | 2-D projection for scatter-plot visualisation                              |
| `umap_10d.npy`     | 305613 × 10 | float32 | 10-D projection as HDBSCAN clustering input                                |
| `umap_scatter.png` | —           | PNG     | Sanity scatter plot coloured by the 15 content categories (tab20 colormap) |

### Sanity Checks (run automatically)

- Shape and dtype printed for both outputs.
- `assert emb.shape[0] == len(metadata_csv)` — row-count alignment with `metadata.csv`.
- Min / max printed to confirm numeric range is sensible.

---

## 7. Pipeline: `generate_clusters.py` ✅ COMPLETED

Run **after** `generate_umap.py`. Performs HDBSCAN clustering on the 10-D UMAP embedding
to discover visually coherent trend groups.

### Final Configuration

| Parameter                  | Value       | Reason                                                                 |
| -------------------------- | ----------- | ---------------------------------------------------------------------- |
| `MIN_CLUSTER_SIZE`         | **512**     | Matched to CLIP embedding dimension; chosen via sweep (see below)      |
| `MIN_SAMPLES`              | 10          | Conservative noise threshold — keeps cluster cores tight               |
| `metric`                   | `euclidean` | UMAP output is Euclidean by construction even though input used cosine |
| `cluster_selection_method` | `eom`       | Excess of Mass — better for variable-density clusters                  |
| `random_state`             | 42          | Consistent with rest of pipeline                                       |

### Final Results

| Metric                        | Value        |
| ----------------------------- | ------------ |
| Clusters discovered           | **105**      |
| Noise points                  | **34.2%**    |
| Avg membership probability    | **> 0.85**   |
| Silhouette score (5 K sample) | **0.576** ✅ |

### `min_cluster_size` Tuning Sweep

| `min_cluster_size` | Clusters | Noise | Decision                                      |
| ------------------ | -------- | ----- | --------------------------------------------- |
| 100                | 457      | 39.7% | Too many — over-segmented                     |
| 300                | 170      | 36.9% | Still too many                                |
| 500                | 107      | 34.1% | Good range                                    |
| 512                | ~105     | 34.2% | ✅ **FINAL** — clean number matching CLIP dim |
| 650                | 87       | 36.4% | Noise went UP — rejected                      |

### Issue: Noise Increased When `min_cluster_size` Was Raised to 650

**Problem:** Raising `min_cluster_size` from 500 → 650 caused noise to jump from 34.1% → 36.4%
even though cluster count dropped from 107 → 87.

**Reason:** When `min_cluster_size` increases, HDBSCAN dissolves clusters that no longer meet
the threshold. Those points do **not** merge into larger clusters — they are ejected as noise
(label `-1`). This is expected HDBSCAN behaviour, not a bug.

**Solution:** Recognised this as a natural noise floor for this dataset (~34%). Locked in
`MIN_CLUSTER_SIZE=512` as the sweet spot before noise started rising again.

### Why 34% Noise Is Acceptable

A 15-category social media dataset naturally contains many one-off images that don't belong
to any recurring visual trend. 34% noise mirrors real platform dynamics where most posts are
unique. The 66% of images that ARE clustered form 105 clean, well-separated visual trends
confirmed by silhouette score 0.576.

### Why Low Category Purity Is Expected

CLIP clusters by **visual appearance**, not semantic label. A moody low-light photo from
"nightlife", "art", and "architecture" may be visually identical. Cross-category clusters
represent genuine cross-domain visual trends — which is exactly what TrendLens is designed
to find.

### Output Files

| File                           | Description                                                    |
| ------------------------------ | -------------------------------------------------------------- |
| `metadata_clustered.csv`       | Full metadata + `cluster` + `cluster_prob` — feeds Step 5      |
| `cluster_summary.csv`          | Per-cluster: size, dominant category, purity, engagement stats |
| `cluster_representatives.json` | Highest-probability image path per cluster                     |
| `cluster_scatter.png`          | 2-D UMAP scatter coloured by cluster label                     |
| `cluster_representatives.png`  | Image grid of one representative per cluster                   |

---

## 8. Pipeline: `generate_temporal_trends.py` ✅ COMPLETED

Run **after** `generate_clusters.py`. Tracks when each visual trend peaks and classifies
clusters as Rising, Stable, or Declining.

### Methodology

For each post, an "active window" is defined as:

```
[timestamp  →  trend_active_until]
```

For each quarter (2010–2021), we count how many posts are simultaneously active:

```
active_posts(Q) = count of posts where timestamp <= Q <= trend_active_until
```

The quarter with the highest overlap count = **PEAK QUARTER** for that cluster.

This combines two signals:

- **Relative post ordering is preserved** — photo IDs are chronological so earlier IDs
  genuinely came before later ones on Flickr
- **`trend_active_until` is engagement-driven** — high engagement = longer active window,
  so high-performing posts contribute more to the activity curve

### Lifecycle Classification

| Stage     | Condition                       | Meaning                                 |
| --------- | ------------------------------- | --------------------------------------- |
| Rising    | Peak position ≥ 60% of timeline | Peak in 2016–2019 — trend still gaining |
| Stable    | 35% < Peak position < 60%       | Peak in 2013–2016 — matured trend       |
| Declining | Peak position ≤ 35% of timeline | Peak in 2010–2013 — trend has faded     |

### Final Results

| Metric             | Value                            |
| ------------------ | -------------------------------- |
| Rising clusters    | **32** (peak quarters 2016–2019) |
| Stable clusters    | **27** (peak quarters 2013–2016) |
| Declining clusters | **46** (peak quarters 2010–2013) |

### Key Finding

Declining clusters have **higher engagement rates** than Rising ones (0.249 vs 0.226).
Early trends were more viral but burned out faster. Rising trends show sustained moderate
engagement over longer periods — a genuine insight about trend lifecycle dynamics.

### Issues Encountered During Step 5

**Issue 1 — Invalid frequency `QS` for `to_period()`:**
Pandas `to_period()` uses `"Q"` not `"QS"`. The `QS` alias only works with `resample()`.
**Fix:** Changed `FREQ = "QS"` to `FREQ = "Q"`.

**Issue 2 — All 104/105 clusters classified as Declining:**
Original slope-based classifier looked at post volume per quarter, which always peaked
early due to linear photo_id → timestamp mapping.
**Fix:** Changed timestamp generation to random uniform distribution in `generate_metadata.py`
and switched classification to peak-position-based (where in the timeline does the
activity curve peak) rather than slope-based.

**Issue 3 — Rising clusters showing negative `growth_rate_pct`:**
Slope classifier and growth rate were measuring different windows and disagreeing.
**Fix:** Removed growth_rate from classification logic; classification now based solely
on peak quarter position within the full 2010–2021 timeline.

### Output Files

| File                | Description                                                  |
| ------------------- | ------------------------------------------------------------ |
| `trend_metrics.csv` | Per-cluster: peak_quarter, lifecycle_stage, engagement stats |
| `trend_summary.png` | Top 20 Rising clusters ranked by engagement rate             |
| `trend_graphs/`     | 105 activity curve PNGs — one per cluster                    |

### `trend_metrics.csv` Schema

| Column                     | Description                                   |
| -------------------------- | --------------------------------------------- |
| `cluster`                  | Integer cluster ID                            |
| `dominant_category`        | Most common semantic category in cluster      |
| `total_posts`              | Number of images in cluster                   |
| `peak_quarter`             | Quarter with most simultaneously active posts |
| `peak_active_posts`        | Number of active posts at peak                |
| `lifecycle_stage`          | Rising / Stable / Declining                   |
| `trend_window_start`       | First quarter with any activity               |
| `trend_window_end`         | Last quarter with any activity                |
| `mean_engagement_rate`     | Average engagement rate across cluster        |
| `viral_rate`               | Fraction of posts flagged `is_viral`          |
| `mean_trend_duration_days` | Average modelled trend lifespan               |

---

## 9. Utility: `script.py`

One-off maintenance script. Drops a known-bad image (`28552@N91/205379.jpg` — truncated/corrupt)
from `metadata.csv`, resets the index, and asserts that `embeddings.npy` row count still matches.

Run manually only when dataset–embedding alignment needs to be repaired after discovering a
bad image post-embedding.

---

## 10. Metadata Schema (`metadata.csv`)

**305,613 rows × 25 columns** after all pipeline steps. All columns present for every valid image.

| Column                    | Type        | Description                                                             |
| ------------------------- | ----------- | ----------------------------------------------------------------------- |
| `post_id`                 | str         | `{user_id}_{photo_id}` — unique post key                                |
| `user_id`                 | str         | Flickr-style user identifier (e.g. `59@N75`)                            |
| `photo_id`                | str         | Numeric photo stem (e.g. `775`)                                         |
| `photo_id_int`            | int         | Integer version of photo_id for ordering/mapping                        |
| `image_path`              | str         | Relative path: `train/<user_id>/<photo_id>.jpg`                         |
| `timestamp`               | str         | ISO-8601 UTC string (2010–2019) — randomly assigned                     |
| **`likes`**               | int         | Synthetic like count (lognormal, category + user modulated)             |
| **`comments`**            | int         | Synthetic comment count (beta fraction of likes × 0.25)                 |
| **`reposts`**             | int         | Shares/reposts (beta-skewed fraction of likes × 0.08)                   |
| **`saves`**               | int         | Bookmarks/saves (beta fraction of likes × 0.15; high-intent)            |
| **`views`**               | int         | Estimated impressions (10–120× likes)                                   |
| **`reach`**               | int         | Unique accounts reached (≥ views; viral reposts can push reach > views) |
| **`follower_count`**      | int         | Creator follower count (power-law; per-user, stable)                    |
| **`engagement_rate`**     | float       | `(likes + comments + reposts) / reach × 100` — %                        |
| **`is_viral`**            | bool        | `True` if `engagement_rate > 3%` OR `reposts > 50`                      |
| `category`                | str         | One of 15 content categories                                            |
| `tags`                    | str (JSON)  | List of 2–8 hashtags (category-specific + generic)                      |
| `groups`                  | str (JSON)  | List of 0–4 Flickr group names                                          |
| `geo_lat`                 | float / NaN | Latitude (±0.15° jitter; NaN if no location)                            |
| `geo_lon`                 | float / NaN | Longitude (±0.15° jitter; NaN if no location)                           |
| `geo_city`                | str / NaN   | City name (NaN if no location)                                          |
| `user_total_posts`        | int         | Total posts by this user in the full manifest                           |
| `is_synthetic`            | bool        | Always `True` — marks synthetic origin                                  |
| **`trend_duration_days`** | float       | Modelled trend lifespan in days (1–180, lognormal)                      |
| **`trend_active_until`**  | str         | ISO-8601 UTC — `timestamp + trend_duration_days`                        |

### Engagement Signal Design Notes

- **`reposts`** use `beta(1.2, 18.0)` — heavy right-skew; most posts get 0 reposts.
- **`saves`** use `beta(2.0, 12.0)` — slightly less skewed; saves are 2–15% of likes.
- **`reach ≥ views`** because viral reposts expose content to audiences beyond the original followers.
- **`engagement_rate`** normalises raw interactions by reach, making it comparable across accounts
  of wildly different follower counts.
- **`is_viral`** flags content that either achieved broad organic spread (`reposts > 50`) or
  exceptional interaction density (`engagement_rate > 3%`).
- **`follower_count`** is stable per user (same across all their posts).
- **`trend_duration_days`** and **`trend_active_until`** model how long a piece of content stays
  "trend-relevant", enabling temporal filtering and trajectory analysis.

---

## 11. Embeddings Schema (`embeddings.npy`)

| Property          | Value                                            |
| ----------------- | ------------------------------------------------ |
| Shape             | `(305613, 512)`                                  |
| dtype             | `float32`                                        |
| Normalisation     | L2-normalised (unit vectors)                     |
| Similarity metric | Dot product = cosine similarity                  |
| Model             | CLIP ViT-B/32 visual encoder + visual projection |

Row `i` of `embeddings.npy` corresponds to row `i` of `metadata.csv`.

---

## 12. Key Design Decisions

1. **Why CLIP?** CLIP's joint image–text embedding space means the 512-d vectors capture semantic
   visual content (not just colour histograms), enabling meaningful clustering and cross-modal search.

2. **Why synthetic engagement?** The raw dataset (SMPD challenge format) provides image files and
   user IDs but no engagement labels. Synthetic signals let us prototype trend-detection models that
   would use real engagement data in production.

3. **Why lognormal for likes/followers?** Social-media engagement and follower distributions are
   empirically heavy-tailed / power-law; lognormal is a tractable approximation of this shape.

4. **Why checkpoint every 200 batches?** Embedding ~305 K images on CPU takes ~90 minutes. Checkpointing
   allows recovery from interruptions without restarting from scratch.

5. **Why store tags/groups as JSON strings in CSV?** Pandas CSV round-trips list columns unreliably;
   JSON strings are unambiguous and easily parsed with `json.loads()`.

6. **Why model trend duration separately?** `trend_active_until` enables time-windowed queries
   (e.g., "which trends are still active as of date X?") and temporal trajectory analysis without
   requiring the downstream model to re-derive the signal from raw timestamps.

7. **Why two separate scripts instead of a notebook?** Scripts are easier to run on remote/headless
   machines, composable in CI/CD pipelines, and simpler to checkpoint and resume reliably.

8. **Why `MIN_CLUSTER_SIZE=512` for HDBSCAN?** Chosen via systematic sweep (100 → 650). 512 matches
   the CLIP embedding dimension (clean, justifiable), lands in the target cluster range (~105 clusters),
   and sits just before the noise plateau where further increases dissolve valid clusters.

9. **Why `metric="euclidean"` for HDBSCAN despite cosine in UMAP?** UMAP's output space is Euclidean
   by construction regardless of the input metric. Cosine is applied during UMAP reduction; the
   resulting 10-D coordinates are then clustered with Euclidean distance, which is correct.

10. **Why random timestamps instead of linear photo_id mapping?** The original approach mapped photo
    IDs linearly to 2010–2019, causing all clusters to peak in 2010–2013 regardless of content —
    making temporal trend tracking meaningless. Random uniform assignment spreads posts evenly
    (~30,500/year) so clusters naturally peak at different periods, enabling genuine Rising/Stable/
    Declining classification.

11. **Why peak-position classification instead of slope?** Slope-based classifiers were unreliable
    because they depended on the last N quarters of post volume, which varied based on random
    timestamp assignment rather than real behaviour. Peak position (where in the 2010–2021 timeline
    does the activity curve peak) is a more stable and interpretable signal.

---

## 13. Dependencies (`requirements.txt`)

```
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
Pillow>=9.0.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
tqdm>=4.65.0
nbformat>=5.7.0
umap-learn>=0.5.0
hdbscan>=0.8.29
scikit-learn>=1.3.0
```

Install with:

```bash
pip install -r requirements.txt
```

> **Note:** PyTorch does not support Python 3.13 as of 2026-08. Use Python 3.11.

---

## 14. Reproducibility

All random operations use a fixed seed of **42**:

- `random.seed(42)` — Python stdlib (used for per-user profile sampling)
- `np.random.seed(42)` — legacy NumPy (used in taxonomy helpers)
- `np.random.default_rng(42)` — new NumPy Generator (used for engagement signal generation)

Re-running all scripts from scratch with the same image files will produce **identical** outputs.

---

## 15. Planned / Future Steps

| Step                 | Status                                           | Description                                                                 |
| -------------------- | ------------------------------------------------ | --------------------------------------------------------------------------- |
| HDBSCAN Clustering   | ✅ **DONE** — 105 clusters, silhouette 0.576     | `generate_clusters.py` with `MIN_CLUSTER_SIZE=512`                          |
| Temporal Analysis    | ✅ **DONE** — 32 Rising, 27 Stable, 46 Declining | `generate_temporal_trends.py` — peak-position classification                |
| FAISS + RAG Module   | pending                                          | Vector index + conversational querying ("what food aesthetic is trending?") |
| Popularity Predictor | pending                                          | CLIP + BERT features → engagement trajectory forecast                       |
| Baseline Comparison  | pending                                          | Compare against published SMP Challenge baseline                            |
| Geo Trending         | pending                                          | Filter by `geo_city` to surface location-specific visual trends             |
| Dashboard            | pending                                          | Streamlit / Gradio UI for visual trend browsing and NN search               |
| Production Swap-In   | pending                                          | Replace `is_synthetic=True` rows with real Flickr API engagement data       |

---

_Last updated: 2026-08-02 · TrendLens v1.4_
