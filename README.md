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
     ▼  Temporal analysis: daily counts, growth rate, emerging score
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
- `GET /api/trends` — top emerging trends
- `GET /api/clusters` — all cluster interpretations
- `GET /api/live-trends` — real Instagram emerging themes
- `GET /api/live-images?name=` — serve downloaded Instagram images

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
| Dimensionality reduction | **UMAP** (10-D for clustering) |
| Clustering | **HDBSCAN** (visual trend group discovery) |
| Cluster tracking | **FAISS centroid KNN** + stable UUID-based registry |
| Visual captioning | **BLIP** `blip-image-captioning-base` |
| Photography-style tagging | **CLIP zero-shot** over a curated style prompt bank (framing / lighting / grading / process / composition) |
| Temporal analysis | Daily post counts, growth rate, emerging score |
| Vector index | **FAISS** `IndexFlatIP` over sentence-transformer embeddings |
| RAG retrieval | **sentence-transformers** `all-MiniLM-L6-v2` |
| LLM writing layer | **Optional, plug and play** (Gemini/OpenAI/Ollama) — rewrites evidence into prose; swap providers via config without code changes |

---

## Algorithm Benchmarking

All core algorithms were benchmarked across multiple configurations to validate the selected hyperparameters:

| Algorithm | Configuration Tested | Selected | Reason |
|-----------|---------------------|----------|--------|
| CLIP | 128×128, 224×224, 336×336 | **224×224** | Native training resolution, best coherence |
| BLIP | base, large, BLIP-2 API | **base** | 82ms latency, 3.2 img/s throughput |
| Sentence-Transformer | MiniLM-L6-v2, mpnet, MiniLM-L12 | **MiniLM-L6-v2** | 0.8ms query, 384-dim efficient |
| UMAP | 5, 10, 15, 20 components | **10** | Highest silhouette (0.524), lowest noise |
| HDBSCAN | MCS 20-200, MS 5-25 | **MCS=50, MS=10** | Best cluster quality (silhouette 0.524) |
| FAISS | FlatIP, IVFFlat, HNSWFlat | **FlatIP** | Exact search, 0.12ms latency |

See `benchmark_algorithms.py` and `benchmark_clip_sizes.py` for benchmark scripts.

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
- **Cluster names are VLM interpretations.** BLIP captions describe visual content, not ground truth meaning.
- **Answers are evidence-grounded.** Every recommendation is backed by real post data, real engagement, and real growth metrics.

---

## Hardware Notes

| Task | Time (estimated) |
|------|-----------------|
| Apify fetch | ~30s (depends on API) |
| Image download | ~2min (per ~10 accounts, 50 posts each) |
| CLIP embeddings (500 images) | ~1min |
| UMAP + HDBSCAN (baseline) | ~30s |
| BLIP captioning | ~2min |
| FAISS index build | ~10s |
| **Incremental run (KNN assign)** | **~15s** (skip HDBSCAN + UMAP) |

> All pipeline stages are CPU-compatible and cacheable/resumable.

---

_TrendLens · Instagram visual trend detection · Last updated: 2026-08-24_
