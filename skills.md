# TrendLens — Project Skills & Context Reference

> **Purpose:** This file is the canonical context document for the TrendLens project.
> It describes the architecture, design decisions, data schema, and pipeline steps in enough detail
> to serve as a reference for future development, AI-assisted coding sessions, and onboarding.

---

## 1. Project Overview

**TrendLens** is an end-to-end multimodal visual trend-detection pipeline built on a social-media-style
photo dataset (~69 K images from ~5 574 users). The pipeline transforms raw image files and a
Flickr-style filepath manifest into a structured, engagement-enriched dataset paired with 512-dimensional
CLIP visual embeddings — the foundation for downstream trend clustering, virality prediction, and
content recommendation.

**Goal:** Given a corpus of photos with rich engagement signals, identify visual trends (recurring
aesthetics, colour palettes, subject matter) that correlate with high customer engagement.

---

## 2. Repository Layout

```
trendlens/
├── generate_metadata.py       # Pipeline script — Steps 1–4 (run first)
├── generate_embeddings.py     # Pipeline script — CLIP embeddings (run second)
├── script.py                  # One-off utility: drops truncated images & verifies alignment
├── requirements.txt           # Python dependencies
├── train_img_filepath.txt     # Flickr-style manifest: train/<user_id>/<photo_id>.jpg
├── train/                     # Image root — subdirs named by user_id
│   └── <user_id>/
│       └── <photo_id>.jpg
└── trendlens_outputs/         # All generated artefacts (git-ignored)
    ├── smpd_metadata.json         # Raw synthetic engagement records (305 K rows)
    ├── metadata.csv               # Enriched, validated CSV (69 226 rows × 25 cols)
    ├── embeddings.npy             # CLIP vectors — float32 (69226, 512) L2-normalised
    ├── embeddings_checkpoint.npy  # Rolling checkpoint (mirrors final on completion)
    └── metadata_checkpoint.csv   # Metadata slice aligned with the checkpoint
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

| Category | Likes Multiplier |
|---|---|
| food | 3.2 |
| fashion | 3.0 |
| portrait | 2.8 |
| travel | 2.7 |
| events | 2.5 |
| animals | 2.4 |
| nightlife | 2.3 |
| family | 2.2 |
| sports | 2.0 |
| nature | 1.9 |
| architecture | 1.8 |
| street | 1.7 |
| art | 1.6 |
| abstract | 1.4 |
| technology | 1.3 |

Each category has a curated hashtag pool (`CATEGORY_TAGS`) plus a shared generic-tag pool.

#### 1b. Geo Pool

20 world cities with (city name, latitude, longitude) tuples. ~70 % of posts get a location tag
(60 % home city, 40 % random city), with ±0.15° jitter for privacy realism.

#### 1c. Per-User Profiles

Keyed by `user_id` — built once, reused for all posts by that user:

| Field | Distribution | Notes |
|---|---|---|
| `preferred_cat` | Uniform random from 15 categories | 70 % of that user's posts use this |
| `home_city` | Uniform random from geo pool | Anchor for location generation |
| `likes_mult` | LogNormal(0, 0.5) clipped [0.3, 6.0] | Audience size / influence proxy |
| `follower_count` | LogNormal(6.5, 1.8) clipped [10, 5M] | Power-law; most small, few mega-influencers |

#### 1d. Timestamp Generation

`photo_id` integers are mapped **linearly** onto the range **2010-01-01 → 2019-12-31** with a
±7-day random jitter, preserving the chronological ordering implied by Flickr photo IDs.

---

### Step 2 — Synthetic Engagement Records → `smpd_metadata.json`

Generates a realistic engagement record for every entry in the filepath manifest
(305 613 total, including paths for missing files on disk).

| Signal | Method |
|---|---|
| `likes` | LogNormal, modulated by category multiplier × user `likes_mult` |
| `comments` | Beta(1.5, 6.0) fraction × `likes` × 0.25 |
| `views` | Uniform(10×, 120×) `likes` |
| `reposts` | Beta(1.2, 18.0) fraction × `likes` × 0.08 — heavy right-skew |
| `saves` | Beta(2.0, 12.0) fraction × `likes` × 0.15 |
| `reach` | max(views, views × Uniform(0.6, 1.1)) |
| `engagement_rate` | (likes + comments + reposts) / reach × 100 |
| `is_viral` | True if engagement_rate > 3% **or** reposts > 50 |

---

### Step 3 — Enriched Metadata CSV → `metadata.csv`

1. Re-parses the filepath manifest into `df_base`.
2. Checks file existence on disk; drops the 236 387 entries whose files are absent.
3. Merges the synthetic JSON onto the valid 69 226 rows by `(post_id, user_id, photo_id)`.
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

| Category | Base Days |
|---|---|
| events | 3 |
| sports | 5 |
| nightlife | 7 |
| animals | 7 |
| street | 10 |
| portrait | 10 |
| food | 14 |
| family | 14 |
| nature | 14 |
| travel | 21 |
| architecture | 21 |
| abstract | 21 |
| art | 28 |
| fashion | 30 |
| technology | 45 |

`trend_active_until` = `timestamp` + `trend_duration_days` (ISO-8601 UTC string).

Final CSV saved to `trendlens_outputs/metadata.csv`.

---

## 5. Pipeline: `generate_embeddings.py`

Run after `generate_metadata.py`. Produces `embeddings.npy`.

| Setting | Value |
|---|---|
| Model | `openai/clip-vit-base-patch32` |
| Batch size | 32 images |
| Checkpoint frequency | Every 200 batches |
| Device | CUDA (auto-detected) or CPU |

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

## 6. Utility: `script.py`

One-off maintenance script. Drops a known-bad image (`28552@N91/205379.jpg` — truncated/corrupt)
from `metadata.csv`, resets the index, and asserts that `embeddings.npy` row count still matches.

Run manually only when dataset–embedding alignment needs to be repaired after discovering a
bad image post-embedding.

---

## 7. Metadata Schema (`metadata.csv`)

**69 226 rows × 25 columns** after all pipeline steps. All columns present for every valid image.

| Column | Type | Description |
|---|---|---|
| `post_id` | str | `{user_id}_{photo_id}` — unique post key |
| `user_id` | str | Flickr-style user identifier (e.g. `59@N75`) |
| `photo_id` | str | Numeric photo stem (e.g. `775`) |
| `photo_id_int` | int | Integer version of photo_id for ordering/mapping |
| `image_path` | str | Relative path: `train/<user_id>/<photo_id>.jpg` |
| `timestamp` | str | ISO-8601 UTC string (2010–2019) |
| **`likes`** | int | Synthetic like count (lognormal, category + user modulated) |
| **`comments`** | int | Synthetic comment count (beta fraction of likes × 0.25) |
| **`reposts`** | int | Shares/reposts (beta-skewed fraction of likes × 0.08) |
| **`saves`** | int | Bookmarks/saves (beta fraction of likes × 0.15; high-intent) |
| **`views`** | int | Estimated impressions (10–120× likes) |
| **`reach`** | int | Unique accounts reached (≥ views; viral reposts can push reach > views) |
| **`follower_count`** | int | Creator follower count (power-law; per-user, stable) |
| **`engagement_rate`** | float | `(likes + comments + reposts) / reach × 100` — % |
| **`is_viral`** | bool | `True` if `engagement_rate > 3%` OR `reposts > 50` |
| `category` | str | One of 15 content categories |
| `tags` | str (JSON) | List of 2–8 hashtags (category-specific + generic) |
| `groups` | str (JSON) | List of 0–4 Flickr group names |
| `geo_lat` | float / NaN | Latitude (±0.15° jitter; NaN if no location) |
| `geo_lon` | float / NaN | Longitude (±0.15° jitter; NaN if no location) |
| `geo_city` | str / NaN | City name (NaN if no location) |
| `user_total_posts` | int | Total posts by this user in the full manifest |
| `is_synthetic` | bool | Always `True` — marks synthetic origin |
| **`trend_duration_days`** | float | Modelled trend lifespan in days (1–180, lognormal) |
| **`trend_active_until`** | str | ISO-8601 UTC — `timestamp + trend_duration_days` |

### Engagement Signal Design Notes

The engagement signals are deliberately **correlated** to reflect real platform dynamics:

- **`reposts`** use `beta(1.2, 18.0)` — heavy right-skew; most posts get 0 reposts.
- **`saves`** use `beta(2.0, 12.0)` — slightly less skewed; saves are 2–15% of likes.
- **`reach ≥ views`** because viral reposts expose content to audiences beyond the original followers.
- **`engagement_rate`** normalises raw interactions by reach, making it comparable across accounts
  of wildly different follower counts.
- **`is_viral`** flags content that either achieved broad organic spread (`reposts > 50`) or
  exceptional interaction density (`engagement_rate > 3%`).
- **`follower_count`** is stable per user (same across all their posts) — enabling future
  normalisation of engagement by audience size.
- **`trend_duration_days`** and **`trend_active_until`** model how long a piece of content stays
  "trend-relevant", enabling temporal filtering and trajectory analysis.

---

## 8. Embeddings Schema (`embeddings.npy`)

| Property | Value |
|---|---|
| Shape | `(69226, 512)` |
| dtype | `float32` |
| Normalisation | L2-normalised (unit vectors) |
| Similarity metric | Dot product = cosine similarity |
| Model | CLIP ViT-B/32 visual encoder + visual projection |

Row `i` of `embeddings.npy` corresponds to row `i` of `metadata.csv`.

---

## 9. Key Design Decisions

1. **Why CLIP?** CLIP's joint image–text embedding space means the 512-d vectors capture semantic
   visual content (not just colour histograms), enabling meaningful clustering and cross-modal search.

2. **Why synthetic engagement?** The raw dataset (SMPD challenge format) provides image files and
   user IDs but no engagement labels. Synthetic signals let us prototype trend-detection models that
   would use real engagement data in production.

3. **Why lognormal for likes/followers?** Social-media engagement and follower distributions are
   empirically heavy-tailed / power-law; lognormal is a tractable approximation of this shape.

4. **Why checkpoint every 200 batches?** Embedding ~69 K images on CPU takes ~90 minutes. Checkpointing
   allows recovery from interruptions without restarting from scratch.

5. **Why store tags/groups as JSON strings in CSV?** Pandas CSV round-trips list columns unreliably;
   JSON strings are unambiguous and easily parsed with `json.loads()`.

6. **Why model trend duration separately?** `trend_active_until` enables time-windowed queries
   (e.g., "which trends are still active as of date X?") and temporal trajectory analysis without
   requiring the downstream model to re-derive the signal from raw timestamps.

7. **Why two separate scripts instead of a notebook?** Scripts are easier to run on remote/headless
   machines, composable in CI/CD pipelines, and simpler to checkpoint and resume reliably.

---

## 10. Dependencies (`requirements.txt`)

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
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 11. Reproducibility

All random operations use a fixed seed of **42**:
- `random.seed(42)` — Python stdlib (used for per-user profile sampling)
- `np.random.seed(42)` — legacy NumPy (used in taxonomy helpers)
- `np.random.default_rng(42)` — new NumPy Generator (used for engagement signal generation)

Re-running both scripts from scratch with the same image files will produce **identical** outputs.

---

## 12. Planned / Future Steps

| Step | Description |
|---|---|
| Clustering | K-Means / HDBSCAN on CLIP embeddings to identify visual trend clusters |
| Trend Scoring | Aggregate `engagement_rate` and `is_viral` within clusters over time to rank rising trends |
| Temporal Analysis | Use `timestamp` + `trend_active_until` to track cluster popularity trajectory (2010–2019) |
| Geo Trending | Filter by `geo_city` to surface location-specific visual trends |
| Follower-Normalised Ranking | Use `follower_count` to surface high-engagement micro-influencer content |
| Dashboard | Streamlit / Gradio UI for visual trend browsing and NN search |
| Production Swap-In | Replace `is_synthetic=True` rows with real Flickr API engagement data |

---

*Last updated: 2026-04-29 · TrendLens v1.1*
