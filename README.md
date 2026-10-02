# TrendLens

> **Find trends before they have a name.**
>
> Every existing trend tool — Google Trends, Exploding Topics, Brandwatch — can only detect trends that *already have words attached to them*. If it doesn't have a hashtag, it's invisible.
>
> TrendLens detects trends **visually**, from raw Instagram image clusters, before language catches up.

---

## The Core Insight

Visual aesthetics spread before language catches up. "Cottagecore" existed as a cluster of images for ~2 years before the word was coined. "Dark academia" spread visually before it got a hashtag. TrendLens finds these clusters *as visual patterns* using CLIP embeddings — which means it can surface a trend while it's still in the **emerging** lifecycle stage, **unnamed, with no hashtag, before it goes mainstream**.

| Existing Tool | How it detects trends | Limitation |
|---|---|---|
| Google Trends | Keyword search frequency | Needs the word first |
| Exploding Topics | Rising search queries | Still text-dependent |
| Brandwatch | Hashtag monitoring | Requires existing adoption |
| Pinterest Trends | On-platform search volume | Platform-locked |
| **TrendLens** | **CLIP visual embedding clusters from Instagram** | **No language needed** |

---

## How It Works

TrendLens pulls real Instagram posts from public accounts across **food, fashion, photography and beauty** via the [Apify API](https://apify.com/), downloads images, and runs them through a visual analysis pipeline:

```
Instagram accounts (80+ accounts across niches)
    │
    ▼  Fetch via Apify API (last 10 days)
Real posts: images, captions, timestamps, likes, comments, views
    │
    ▼  CLIP ViT-B/32 image embeddings (512-d, L2-normalised)
    │
    ▼  ┌─────────────────────────────────────────────────┐
       │  Cluster Tracker (centroid locking + KNN)        │
       │                                                  │
       │  If baseline: UMAP → HDBSCAN → lock centroids   │
       │  If incremental: FAISS KNN → assign to clusters  │
       │  Emerging candidates → HDBSCAN micro-clusters     │
       └─────────────────────────────────────────────────┘
    │
     ▼  BLIP captioning of representative images → cluster labels
     │
     ▼  CLIP zero-shot style tagging (framing / lighting / grading /
        process / composition) → per-cluster "how it is shot" profile
     │
     ▼  Trend definition: recent-vs-prior growth + binomial significance test →
        Rising / InsufficientData verdict, bounded priority, SQLite history store
    │
    ▼  FAISS RAG index (sentence-transformer embeddings)
    │
    ▼  LLM writing layer (Gemini/OpenAI/Ollama) → natural language answer
    │
    ▼  Query → semantic retrieval → evidence-grounded answer
```

---

## Example Query

**Input:** `"What cafe aesthetic is rising this week?"`

**TrendLens Output:**

```
## Trending Visual Aesthetics (Instagram, last 10 days)

Based on 187 posts from 10 Instagram food accounts:

### 1. minimalist latte art (rising)
Keywords: latte, art, minimal, white, ceramic
> a cup of coffee with latte art on a white saucer
Posts: 24 total (18 recent, 6 prior) — +200% vs prior period
Avg engagement: 1450 likes, 42 comments

Example posts:
> "Morning ritual always starts with this minimalist pour"

### 2. rustic brunch spread (emerging)
Keywords: brunch, rustic, wooden, spread, natural
> a wooden table with a full brunch spread
Posts: 15 total (10 recent, 5 prior) — +100% vs prior period
...
```

Every answer includes real engagement data, growth metrics, and example captions from actual Instagram posts.

> _Illustrative sample only — real output reflects the actual corpus (posts, engagement medians, and style tags measured on the collected images)._

---

## Setup

```bash
git clone <repo-url> && cd trendlens

# Python backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Environment
cp .env.example .env
# Edit .env and set APIFY_API_TOKEN (required for Instagram scraping)
```

> **APIFY_API_TOKEN** is required. Get one free at [apify.com/account#/integrations](https://apify.com/account#/integrations).

---

## Usage

### 1. Collect Instagram data (run daily/weekly)

```bash
source venv/bin/activate
python -m src.data_collector              # incremental (default) — smart KNN assignment
python -m src.data_collector --days 7     # override: last 7 days only
python -m src.data_collector --baseline   # force full re-cluster from scratch
```

**How it works:**
- **First run** (baseline): Full HDBSCAN clustering → locks cluster centroids → saves registry with stable IDs
- **Subsequent runs** (incremental): Fetches new posts → CLIP embeds → FAISS KNN assigns to existing clusters → detects emerging micro-clusters from unassigned images

Cluster IDs are stable UUIDs (`cls_*`) that persist across runs, enabling genuine time-series tracking of visual trends.

### 2. Ask questions

```bash
python -m src.rag "What cafe aesthetic is rising this week?"
python -m src.rag "What food photography styles are trending on Instagram?"
python -m src.rag "What kind of latte art gets the most engagement?"
python -m src.rag "What street style aesthetics are trending on Instagram?"
python -m src.rag "What editing and colour grading styles are trending in photography?"
python -m src.rag "What makeup looks are trending on Instagram?"
```

### 3. API server

```bash
python -m src.api    # serves on port 8000
```

**Endpoints:**
- `POST /api/rag-query` — `{"query": "..."}` → answer with evidence
- `GET /api/health` — service status
- `GET /api/trends` — top trends from the real Instagram `trends.json` (definition verdicts, priority, style tags)
- `GET /api/clusters` — the same themes as cluster-shaped records
- `GET /api/instagram-trends` — raw Instagram emerging themes
- `GET /api/instagram-images?name=` — serve downloaded Instagram images

---

## Configuration

All config is in `.env` (auto-loaded by `config.py`):

```env
# Required: Apify API token for Instagram scraping
APIFY_API_TOKEN=apify_api_xxxxx

# Optional: Instagram scan window (default: 10 days)
TRENDLENS_INSTAGRAM_DAYS=10

# Optional: LLM writing layer (plug and play — swap providers without code changes)
TRENDLENS_LLM_PROVIDER=gemini
TRENDLENS_LLM_API_KEY=...

# API server
TRENDLENS_API_HOST=0.0.0.0
TRENDLENS_API_PORT=8000
```

---

## Plug and Play API for Sentence Formation

The LLM API used for sentence formation and conversation is **plug and play**. You can swap between different LLM providers without any code changes:

```bash
# In .env
TRENDLENS_LLM_PROVIDER=gemini    # or openai, ollama
TRENDLENS_LLM_API_KEY=your_key   # not needed for ollama
```

Supported providers:
- **Gemini** (default) — `gemini-3.1-flash-lite`
- **OpenAI** — `gpt-4o-mini`
- **Ollama** — `llama3.2` (local, no API key needed)

The system automatically falls back to a deterministic formatter if the LLM fails.

---

## Instagram Accounts

The `account.txt` file lists the Instagram accounts to monitor — one URL or username per line. Currently configured for four niches (~81 accounts):

| Niche | Example accounts |
|-------|------------------|
| Food / cafe / dessert | `food52`, `tasty`, `jamieoliver`, `tartinebaker`, `frenchpress.latteart` |
| Fashion / street style | `voguemagazine`, `highsnobiety`, `styledumonde`, `matildadjerf`, `tokyofashion` |
| Photography | `natgeo`, `magnumphotos`, `jordi.koalitic`, `alan_schaller`, `moodygrams` |
| Beauty / skincare | `hudabeauty`, `rarebeauty`, `glossier`, `ctilburymakeup`, `theordinary` |

CLIP clusters images by visual semantics, so posts from different niches naturally form separate clusters — you can query any niche specifically (`"What street style aesthetics are trending?"`) and retrieval surfaces only the relevant clusters.

To add or remove accounts (or a whole new niche), edit `account.txt` with one Instagram URL or username per line. Note: the file is parsed literally, so no comment lines.

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Data source | **Instagram** via **Apify API** (`apify/instagram-scraper` actor) |
| Image embeddings | **CLIP** `openai/clip-vit-base-patch32` (512-d, L2-normalised) |
| Dimensionality reduction | **UMAP** (15-D for clustering) |
| Clustering | **HDBSCAN** (visual trend group discovery) |
| Cluster tracking | **FAISS centroid KNN** + stable UUID-based registry |
| Visual captioning | **BLIP** `blip-image-captioning-large` |
| Photography-style tagging | **CLIP zero-shot** over a curated style prompt bank (framing / lighting / grading / process / composition) |
| Trend definition | Formal criteria (min posts, authors, active days, ≥0.5 growth margin, binomial significance, engagement floor, coverage) → per-cluster **Rising / InsufficientData** verdict with bounded priority; **SQLite snapshot store** (`data/trend_history.db`) refuses to count a re-run over the same post set as new history |
| Vector index | **FAISS** `IndexFlatIP` over sentence-transformer embeddings |
| RAG retrieval | **sentence-transformers** `BAAI/bge-base-en-v1.5` |
| LLM writing layer | **Optional, plug and play** (Gemini/OpenAI/Ollama) — rewrites evidence into prose; swap providers via config without code changes |

---

## Algorithm Benchmarking

Every pretrained model in the pipeline is chosen by measurement on the **real corpus**, not by
public leaderboard rank and not by library defaults. `config.py` is the single place any of them
is named, so a swap cannot leave a second hard-coded copy behind.

**Image encoder + clustering** — `scripts/select_models.py`, 5 encoders × 18 clustering
configurations on 461 embedded posts, gated on coverage, cluster count and cross-seed stability
(`model_selection_results.csv`, `model_selection_summary.json`). Clustering parameters were then
re-measured on 80 held-out splits, paired — see the note below, which is the part that matters.

| Parameter | Sweep range | In production | Evidence |
|-----------|-------------|---------------|----------|
| Image encoder | ViT-B/32, ViT-B/16, ViT-L/14, laion ViT-B/32, SigLIP2-base | **openai/clip-vit-base-patch32** | tied with 3 of 4 on ρ; cheapest by far |
| UMAP components | 10, 15, 20 | **15** | ρ 0.248 vs 0.170 (10-d) and 0.177 (20-d) |
| HDBSCAN min_cluster_size | 5, 8, 12 | **5** | ρ 0.248 vs 0.084 at 8 |
| HDBSCAN min_samples | 3, 5 | **3** | |
| RAG retrieval k | 1–8 | **5** | MRR plateaus at k=5 |

Three results worth stating plainly, because all three cut against the obvious choice:

- **The image encoder is basically a tie, and the honest claim is "cheapest of the tied set".**
  Scored paired on the same 80 splits with identical clustering, ViT-B/32 (ρ 0.176) is
  statistically indistinguishable from ViT-B/16 (0.159, p = 0.22), SigLIP2-base (0.153, p = 0.088)
  and ViT-L/14 (0.187, p = 0.46). Only the laion ViT-B/32 checkpoint is clearly worse (0.071,
  p < 0.001). ViT-B/32 is kept because it is the smallest and fastest of the tied set — not
  because it was measurably the best, because it was not.
- **SigLIP2 did not win.** The newer image-text objective is usually the better bet; measured here
  it is not, and it costs ~2× the encode time. "Newer" is not a selection criterion.
- **The nominal winner was rejected, and the sweep script was fixed.** `select_models.py` averaged
  five account splits and nominated UMAP=20. That average has a standard error of ~0.05 — larger
  than every difference the sweep was trying to resolve. Re-measured on **80 held-out splits scored
  paired** (same splits for every configuration, so split noise cancels), 15-d wins:

  | UMAP dims | mean ρ | per-split sd | vs 15-d |
  |---|---|---|---|
  | 10 | 0.170 | 0.132 | 15-d better on 64/80, p < 0.001 |
  | **15** | **0.248** | 0.113 | — |
  | 20 | 0.177 | 0.118 | 15-d better on 65/80, paired t = 6.96, p = 9e-10 |

  Production therefore stays at 15-d, and `SEEDS_SPLITS` is now 25 (SE ≈ 0.025) so the script
  cannot quietly repeat the mistake. A 5× slower sweep is a good trade against rebuilding the
  whole pipeline on a noise artifact.

  Note that `model_selection_summary.json` records that sweep, so its `winner` field is the
  sweep's *nominee* (20-d), not what production runs. The `rho` column in
  `model_selection_results.csv` is likewise from the old 5-seed protocol; the 80-split numbers
  above supersede it for every decision that mattered.

  A second bug lived in the tie-break: the docstring promised ties within 1e-3 on ρ are settled on
  `trend_precision@5`, but the code did a plain lexicographic sort that ignored the tolerance. The
  two leading rows sat 4.9e-4 apart — inside the band — so the ranking was being decided by the
  last digits of a noisy coefficient, and the configuration whose trends never clear the trend
  definition (precision 0.0) was ranked above one that does (0.2).

> **How precise is ρ here? Less than any single number suggests.** Beyond split noise there is a
> second source: re-embedding the same 461 images produces float differences of ~5e-06, which is
> enough for UMAP to return a different projection and for HDBSCAN to return 17 clusters instead
> of 18 — moving ρ from 0.176 to 0.248. So a single-run ρ is reproducible only to about ±0.07, and
> every ρ in this document is a mean over ≥ 20 held-out splits with its spread stated. Quoting one
> to three decimals without that would be false precision.

**Text retriever** — `scripts/select_text_model.py`, 6 candidates on the project's own retrieval
task (`text_model_selection_results.csv`). Queries are BLIP captions of cluster-member images with
the representative image excluded, so no query is a copy of the text it must retrieve.

| Model | R@1 | R@3 | MRR |
|---|---|---|---|
| `all-MiniLM-L6-v2` (incumbent, was a library default) | 0.347 | 0.542 | 0.502 |
| **`BAAI/bge-base-en-v1.5`** | **0.431** | **0.694** | **0.590** |

The incumbent was only ever the sentence-transformers default. It is not the best retriever for
this index.

**Captioner** — `scripts/select_caption_model.py`, 60 real corpus images
(`caption_model_selection_results.csv`).

| Model | CLIPScore | distinct-2 | degenerate | img/s |
|---|---|---|---|---|
| `blip-image-captioning-base` | 0.2740 | 0.704 | 5% | 0.53 |
| **`blip-image-captioning-large`** | **0.2775** | **0.923** | **0%** | 0.04 |

The CLIPScore gap is within noise at n=60. Large is preferred for the two properties that are not:
markedly higher lexical diversity, so cluster names stop repeating each other in the UI, and no
degenerate repetitions. The cost is accepted knowingly and is real — interpreting ~19 clusters
takes ~30 min instead of ~2.5 min.

> Two earlier benchmark tables are retired rather than quietly overwritten. One (UMAP=10,
> MCS=50/MS=10) was measured on a synthetic 512-d fixture of 5 separated Gaussian blobs, never on
> real images, and MCS=50 was never usable at 461 posts. The second ranked separation (η²) behind a
> ≥0.92 stability gate and published ARI 0.949 — a figure not reproducible under the seed set used
> in production, whose honest measured range on this corpus is ~0.28–0.73.

## Evaluation Metrics

Real-data, non-circular evaluation via `scripts/evaluate_real.py` (see `evaluation_real_results.csv`). No synthetic fixture and no self-referential labels. Baselines are ranked the same way TrendLens is; the engagement band baseline can score ~1.0 on prediction by construction, which is exactly why the held-out split below is the load-bearing metric.

| Metric | TrendLens (visual) | Hashtag | Keyword | Engagement band* |
|--------|--------------------|---------|---------|------------------|
| Assignment coverage | **0.750** | 0.165 | 0.818 | 0.999 |
| Held-out engagement ρ (by account) | **0.281** | -0.061 | 0.140 | 0.995* |
| Trend precision@5 (definition-confirmed) | 0.000 | 0.400 | **1.000** | 1.000 |
| Within-group hashtag overlap (lower = unnamed) | **0.025** | 0.483 | 0.005 | 0.004 |
| Group value cohesion | 0.738 | 0.575 | **0.827** | 0.799 |

\* Engagement band = corpus cut into 12 equal bands by engagement rank; its ~1.0 prediction score is an upper bound by construction, not evidence of detection ability.

**Temporal note:** precision@5 = 0.000 is expected on the current 10-day corpus (47% in the last 7 days), as most visual clusters lack sufficient prior history to satisfy the formal Rising gates. This is a property of the fetch window, not the models. The 80-split paired measurement (**ρ = 0.248 ± 0.113**) remains the authoritative estimate for configuration choices.
`evaluate_real.py` reports, for reproducibility with the CSV. Re-scored over **80 held-out
account splits**, TrendLens averages **ρ = 0.248 ± 0.113** (sd across splits). So 0.291 is a
favourable draw from a wide distribution, not a stable point estimate, and the gap to hashtag
detection (0.106) is *not* established at this sample size: both detectors sit inside each other's
spread. The defensible claim is that TrendLens carries a positive, moderately strong engagement
signal across splits (keyword frequency's does not: −0.093) — not that it beats every baseline by
a specific margin. Reporting the single-split 0.291 without this caveat would overstate the result.

**Honest reading:** TrendLens covers 4× more of the corpus than hashtag detection (0.82 vs 0.20 — text tools are blind to posts without hashtags), carries the strongest engagement signal of any content-based detector (ρ = 0.25 ± 0.11 across splits, vs keyword's −0.09), and its groups share 8× fewer hashtags than hashtag-driven groups (0.041 vs 0.332) — the "trend before it has a name" property. It does **not** beat keyword frequency on confirming already-named trends (0.40 vs 1.00 precision@5), and no content-based detector beats a median baseline on absolute engagement error — this corpus is small and heavily skewed, and those limits are reported rather than hidden.

## Trend Definition (formal)

A cluster is **Rising** (see `src/trend_definition.py`) only if, on a recent window vs a prior window of equal length, it simultaneously meets: support (≥ 8 recent posts), persistence (≥ 3 active days), author breadth (≥ 3 distinct authors), growth margin (≥ 50% relative increase), significance (binomial test p < 0.05), and is not engagement-depressed below 75% of the corpus median (median-first, coverage-aware; hidden like-counts are treated as unknown, never as zero). Otherwise the verdict is **InsufficientData** — explicitly not "falling", because a single snapshot cannot establish a decline. A bounded priority in [0, 1] ranks only confirmed trends (0 otherwise).

**Snapshot store:** every pipeline run records the corpus fingerprint, cluster observations, and daily counts to an SQLite `data/trend_history.db`. A re-run that covers the same post set — even after a hyperparameter change — is rejected as a recomputation rather than stored as new history, so longitudinal deltas can only describe the passage of time (or a genuinely new scrape). Longitudinal claims surface only once ≥ 2 distinct corpora exist.

---

## Cluster Tracker (Vector Drift Prevention)

Traditional re-clustering from scratch breaks time-series history — cluster IDs shuffle every run. The Cluster Tracker solves this:

1. **Baseline run**: HDBSCAN clusters images → centroids are locked and saved to `artifacts/cluster_registry.json`
2. **Incremental runs**: New images are embedded with CLIP → FAISS KNN search against locked centroids → assigned to existing clusters (similarity ≥ 0.25)
3. **Emerging detection**: Images too far from all centroids accumulate as candidates → HDBSCAN micro-clusters when pool ≥ 3

**Artifacts:**
- `artifacts/cluster_registry.json` — stable cluster IDs, centroids, metadata
- `artifacts/centroid_index.faiss` — FAISS index over locked centroids

**CLI:**
```bash
python -m src.data_collector              # incremental (default)
python -m src.data_collector --baseline   # force full re-cluster
```

---

## Data Integrity

- **Instagram data is real.** Timestamps, likes, comments, views, and images come from public Instagram accounts via Apify. Not synthetic.
- **Hidden engagement is unknown, not zero.** Instagram returns `-1` for like counts it hides (47 posts in the current corpus). These are treated as NaN everywhere — never summed as 0 — and engagement stats are median-first with an explicit coverage flag.
- **Cluster names are VLM interpretations.** BLIP captions describe visual content, not ground truth meaning.
- **Photography advice is authored, not hallucinated.** Every step in a theme's "how to shoot it" recipe maps 1:1 to a style tag CLIP actually measured on that theme's images (see the permutation test in `style_tags.py`: η² = 0.351, p < 0.001 — the tags do discriminate). The LLM layer is never allowed to invent steps.
- **Evaluation is non-circular.** `scripts/evaluate_real.py` splits held-out accounts apart, scores every detector on the same rows, and treats the engagement-band baseline as an upper bound. The results are what they are — reported without retuning to look better.

---

## Hardware Notes

| Task | Time (estimated) |
|------|-----------------|
| Apify fetch | ~30s (depends on API) |
| Image download | ~2min (per ~10 accounts, 50 posts each) |
| CLIP embeddings (500 images) | ~1min |
| UMAP + HDBSCAN (baseline) | ~30s |
| BLIP captioning | ~30 min (`blip-image-captioning-large`) |
| FAISS index build | ~10s |
| **Incremental run (KNN assign)** | **~15s** (skip HDBSCAN + UMAP) |

> All pipeline stages are CPU-compatible and cacheable/resumable.

---

_TrendLens · Instagram visual trend detection · Last updated: 2026-10-02_
