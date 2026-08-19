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

TrendLens pulls real Instagram posts from public food accounts via the [Apify API](https://apify.com/), downloads images, and runs them through a visual analysis pipeline:

```
Instagram accounts (10 food accounts)
    │
    ▼  Fetch via Apify API (last 10 days)
Real posts: images, captions, timestamps, likes, comments, views
    │
    ▼  CLIP ViT-B/32 image embeddings (512-d, L2-normalised)
    │
    ▼  UMAP dimensionality reduction (10-D)
    │
    ▼  HDBSCAN clustering → visual trend groups
    │
    ▼  BLIP captioning of representative images → cluster labels
    │
    ▼  Temporal analysis: daily counts, growth rate, emerging score
    │
    ▼  FAISS RAG index (sentence-transformer embeddings)
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
python -m src.data_collector              # fetch last 10 days + full pipeline
python -m src.data_collector --days 7     # override: last 7 days only
```

This fetches posts from all accounts in `account.txt`, downloads images, runs CLIP embeddings, HDBSCAN clustering, BLIP captioning, temporal analysis, and builds the RAG index.

### 2. Ask questions

```bash
python -m src.rag "What cafe aesthetic is rising this week?"
python -m src.rag "What food photography styles are trending on Instagram?"
python -m src.rag "What kind of latte art gets the most engagement?"
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

# Optional: LLM writing layer
TRENDLENS_LLM_PROVIDER=gemini
TRENDLENS_LLM_API_KEY=...

# API server
TRENDLENS_API_HOST=0.0.0.0
TRENDLENS_API_PORT=8000
```

---

## Instagram Accounts

The `account.txt` file lists the Instagram accounts to monitor. Currently configured for **food / cafe aesthetics**:

```
artplatterforyou
healthyyfoodiary
sakandpepper
zamanaindia
kraving.bykay
le.kene
fionasfunkyfood
whipitupwithwitz
3idiotsrichmond
holistic.jo
```

To add fashion/beauty accounts, edit `account.txt` with one Instagram URL or username per line.

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Data source | **Instagram** via **Apify API** (`apify/instagram-scraper` actor) |
| Image embeddings | **CLIP** `openai/clip-vit-base-patch32` (512-d, L2-normalised) |
| Dimensionality reduction | **UMAP** (10-D for clustering) |
| Clustering | **HDBSCAN** (visual trend group discovery) |
| Visual captioning | **BLIP** `blip-image-captioning-base` |
| Temporal analysis | Daily post counts, growth rate, emerging score |
| Vector index | **FAISS** `IndexFlatIP` over sentence-transformer embeddings |
| RAG retrieval | **sentence-transformers** `all-MiniLM-L6-v2` |
| LLM writing layer | **Optional** (Gemini/OpenAI/Ollama) — rewrites evidence into prose |

---

## Data Integrity

- **Instagram data is real.** Timestamps, likes, comments, views, and images come from public Instagram accounts via Apify. Not synthetic.
- **Cluster names are VLM interpretations.** BLIP captions describe visual content, not ground truth meaning.
- **No fabricated metrics.** Missing data is 0 or empty, never invented.
- **Answers are evidence-grounded.** Every recommendation is backed by real post data, real engagement, and real growth metrics.

---

## Hardware Notes

| Task | Time (estimated) |
|------|-----------------|
| Apify fetch | ~30s (depends on API) |
| Image download | ~2min (10 accounts, 50 posts each) |
| CLIP embeddings (500 images) | ~1min |
| UMAP + HDBSCAN | ~30s |
| BLIP captioning | ~2min |
| FAISS index build | ~10s |

> All pipeline stages are CPU-compatible and cacheable/resumable.

---

_TrendLens · Instagram visual trend detection · Last updated: 2026-08-18_
