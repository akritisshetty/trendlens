# TrendLens — Project Skills & Context Reference

> **Purpose:** Canonical context document for TrendLens. Describes the
> Instagram-first architecture, pipeline stages, data flow, and design
> decisions — for future development, AI-assisted sessions and onboarding.

---

## 0. Data Integrity Policy (non-negotiable)

1. **No fabricated results.** Nothing is reported that was not measured.
   Fields that are not measured are `null`/omitted — never invented.
2. **Instagram data is real.** Timestamps, likes, comments, views, and
   images come from public Instagram accounts via the Apify API. Not
   synthetic. Labelled `INSTAGRAM_DATA_WARNING`.
3. **VLM output is interpretation, not ground truth.** Cluster names and
   descriptions come from BLIP captions of representative images.
4. **LLM is a writing layer only (if enabled).** RAG answers are always
   formed from pipeline artifacts first. The LLM may rewrite that evidence
   into prose — it is never a knowledge source.
5. **Answers hide pipeline internals.** Cluster IDs, engagement scores,
   lifecycle labels, and trend scores are NOT shown to users. Answers
   focus on actionable visual advice (what to shoot, how to style it).

---

## 1. Project Overview

**TrendLens** is a multimodal visual trend-detection system that pulls
real Instagram posts via the Apify API, clusters them by CLIP visual
semantics, tracks each cluster's activity over time, interprets each
cluster with a vision-language model, and answers natural-language queries
through semantic retrieval + honest formatting.

**Research question:** Can visual clusters derived from Instagram image
embeddings reveal emerging patterns before textual labels appear?

**Current niches:** Food/cafe, fashion/street style, photography and
beauty/skincare (~81 Instagram accounts in `account.txt`).

**Extensible to:** Any visual niche — travel, interiors, fitness — by
editing `account.txt`. CLIP clusters whatever images the pipeline is fed.

---

## 2. Repository Layout

```
trendlens/
├── config.py                   # Central config: paths, Instagram/Apify settings
├── requirements.txt            # Python dependencies
├── README.md                   # User-facing overview
├── commands.md                 # All commands to run the project
├── skills.md                   # This document — canonical context
├── account.txt                 # Instagram accounts to monitor (food, fashion, photography, beauty)
├── .env                        # APIFY_API_TOKEN + optional LLM config
├── .env.example                # Template for .env
├── src/                        # Pipeline modules (python -m src.<module>)
│   ├── apify_client.py         # Apify REST API client for Instagram scraping
│   ├── data_collector.py       # Full pipeline orchestrator (fetch → RAG index)
│   ├── cluster_tracker.py      # Stable cluster tracking (centroid locking + KNN)
│   ├── embeddings.py           # CLIP image embeddings
│   ├── clustering.py           # UMAP + HDBSCAN clustering
│   ├── interpretation.py       # BLIP captioning + cluster interpretation
│   ├── trends.py               # Temporal trend detection
│   ├── retrieval.py            # FAISS index + retrieval
│   ├── rag.py                  # RAG query layer (Instagram-first, legacy fallback)
│   ├── llm.py                  # Optional LLM writing layer
│   ├── live.py                 # Optional Reddit/Wikimedia live trends
│   ├── api.py                  # HTTP API server
│   ├── synthetic_data.py       # Legacy: synthetic metadata
│   ├── data_loader.py          # Legacy: dataset loading
│   └── preprocessing.py        # Legacy: image preprocessing
├── tests/                      # Test suite
├── notebooks/                  # Execution notebooks
├── scripts/                    # Shell scripts
├── data/
│   ├── instagram/              # PRIMARY: Instagram data
│   │   ├── images/             # Downloaded Instagram images
│   │   ├── posts.parquet       # Post metadata
│   │   ├── embeddings.npy      # CLIP embeddings
│   │   ├── trends.json         # Emerging themes
│   │   ├── rag_index.faiss     # FAISS RAG index
│   │   └── rag_chunks.json     # RAG text chunks
│   ├── live/                   # Legacy: Reddit/Wikimedia data
│   ├── embeddings/             # Legacy: SMPD embeddings
│   └── metadata/               # Legacy: SMPD metadata
└── artifacts/                  # Legacy pipeline outputs + cluster registry
    ├── cluster_registry.json   # Stable cluster IDs, centroids, metadata
    ├── centroid_index.faiss    # FAISS index over locked centroids
    └── cluster_metadata/       # Cluster summaries, trend metrics
```

---

## 3. Execution Order

```bash
source venv/bin/activate
cp .env.example .env   # set APIFY_API_TOKEN

# Main pipeline (incremental — default):
python -m src.data_collector           # fetch + KNN assign + micro-cluster + RAG

# Force full re-cluster (baseline):
python -m src.data_collector --baseline

# Override scan window:
python -m src.data_collector --days 7

# Query:
python -m src.rag "What cafe aesthetic is rising this week?"

# API server:
python -m src.api

# Tests:
python -m pytest tests/ -q
```

---

## 4. Pipeline Phases

| Phase | Module | What it does |
|-------|--------|--------------|
| 1 | `apify_client.py` | Fetch Instagram posts via Apify REST API (last N days) |
| 2 | `data_collector.py` | Download images to `data/instagram/images/` |
| 3 | `data_collector.py` | CLIP ViT-B/32 embeddings (512-d, L2-normalised) |
| 4a | `data_collector.py` | **Baseline**: UMAP-10 + HDBSCAN clustering |
| 4b | `cluster_tracker.py` | **Baseline**: Lock centroids, assign stable UUIDs, build FAISS index |
| 4c | `data_collector.py` | **Incremental**: FAISS KNN assignment to existing clusters |
| 4d | `cluster_tracker.py` | **Incremental**: Detect emerging micro-clusters from unassigned images |
| 5 | `data_collector.py` | BLIP captioning of representative images per cluster |
| 6 | `data_collector.py` | Temporal analysis: daily counts, growth rate, emerging score |
| 7 | `data_collector.py` | FAISS RAG index (sentence-transformer embeddings) |
| 8 | `rag.py` | Query → semantic retrieval → LLM-polished answer |

### Temporal Analysis

For each cluster:
- Daily post counts over the scan window
- Recent vs prior window comparison
- Growth rate = (recent / prior) - 1
- Emerging score = f(growth, recency, novelty penalty)
- Average engagement (likes, comments, views)

### RAG Index

- Cluster summaries are text-chunked with keywords, BLIP captions,
  growth metrics, and example post captions
- Embedded with `all-MiniLM-L6-v2` sentence-transformer
- Indexed with FAISS `IndexFlatIP`
- Query retrieves top-k relevant clusters

---

## 4b. Cluster Tracker (Vector Drift Prevention)

**Problem:** Re-clustering from scratch at each time step shuffles cluster
IDs, breaking time-series history.

**Solution:** Lock baseline cluster centroids and use FAISS KNN for
incremental assignment.

### Architecture

```
Baseline Run:
  HDBSCAN → cluster labels → compute centroids → lock to registry

Incremental Run:
  New images → CLIP embed → FAISS KNN against centroids
    ├── similarity ≥ 0.25 → assign to existing cluster
    └── similarity < 0.25 → add to pending candidates
        └── if candidates ≥ 3 → HDBSCAN micro-clusters → new stable IDs
```

### Artifacts

| File | Purpose |
|------|---------|
| `artifacts/cluster_registry.json` | Stable cluster IDs (`cls_{uuid}`), centroids, metadata, pending candidates |
| `artifacts/centroid_index.faiss` | FAISS `IndexFlatIP` over locked centroids |
| `artifacts/centroid_index.json` | Stable ID mapping (FAISS position → `cls_*`) |

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `assignment_threshold` | 0.25 | Min cosine similarity to assign to existing cluster |
| `micro_cluster_min_size` | 3 | Min unassigned images to trigger HDBSCAN |

### Cluster Registry Schema

```json
{
  "version": 1,
  "baseline_date": "2026-08-19T...",
  "assignment_threshold": 0.25,
  "micro_cluster_min_size": 3,
  "clusters": {
    "cls_a1b2c3d4": {
      "stable_id": "cls_a1b2c3d4",
      "original_hdbscan_id": 0,
      "centroid": [0.123, -0.456, ...],
      "n_members": 15,
      "first_seen": "2026-08-19T...",
      "last_updated": "2026-08-19T...",
      "name": "tzatziki doughnuts",
      "keywords": ["tzatziki", "doughnuts", "oregano"],
      "lifecycle": "Rising",
      "total_posts_all_time": 42
    }
  },
  "pending_candidates": []
}
```

---

## 5. Data Schema

### Instagram posts (`data/instagram/posts.parquet`)

| Column | Type | Description |
|--------|------|-------------|
| `post_id` | str | Instagram post ID |
| `author` | str | Instagram username |
| `image_url` | str | CDN image URL |
| `caption` | str | Post caption text |
| `timestamp` | datetime | Post timestamp (UTC, real) |
| `likes` | int | Like count (real) |
| `comments` | int | Comment count (real) |
| `views` | int | View count (0 for non-video posts) |
| `hashtags` | list[str] | Hashtags extracted from caption |
| `source` | str | Always "instagram" |

### Trends JSON (`data/instagram/trends.json`)

```json
{
  "disclaimer": "REAL INSTAGRAM DATA: ...",
  "generated_at": "2026-08-18T...",
  "source": "instagram",
  "scan_days": 10,
  "n_posts": 187,
  "n_themes": 12,
  "themes": [
    {
      "name": "minimalist latte art",
      "keywords": ["latte", "art", "minimal", "white"],
      "blip_caption": "a cup of coffee with latte art",
      "n_posts": 24,
      "recent_posts": 18,
      "prior_posts": 6,
      "growth_rate": 2.0,
      "emerging_score": 0.45,
      "avg_likes": 1450,
      "avg_comments": 42,
      "daily_counts": {"2026-08-15": 5, "2026-08-16": 8, ...},
      "example_captions": ["Morning ritual...", ...]
    }
  ]
}
```

---

## 6. RAG Query System

### Query Path

```
User query
    → check Instagram data available (data/instagram/trends.json)
    → if Instagram data: retrieve via FAISS RAG index
    → if live-trend intent ("what's rising"): sort by emerging_score
    → format answer with real engagement data + example captions
    → if no Instagram data: fall back to legacy pipeline
```

### Answer Format

Answers include:
- Cluster name (from BLIP keywords)
- Visual characteristics
- Growth rate and emerging score
- Real engagement metrics (likes, comments)
- Example post captions
- Daily post count trend

Answers never include:
- Pipeline internals (cluster IDs, embedding dimensions)
- Fabricated metrics
- Synthetic data labels (when using real Instagram data)

---

## 7. Key Design Decisions

1. **Instagram via Apify** — real public post data, no scraping needed
2. **CLIP for embeddings** — joint image–text space, one encoder for both
3. **UMAP-10 → HDBSCAN** — proven clustering pipeline for visual groups
4. **BLIP for captioning** — CPU-capable VLM for cluster interpretation
5. **Sentence-transformer for RAG** — `all-MiniLM-L6-v2` for text retrieval
6. **Emerging score** — growth velocity with novelty penalty
7. **10-day rolling window** — recent enough for actionable trends
8. **Idempotent pipeline** — re-running only adds new posts
9. **No new dependencies** — uses `requests` for Apify API, no SDK needed
10. **Answers focus on actionable advice** — what to shoot, not data science
11. **Stable cluster IDs** — UUID-based registry prevents vector drift across runs
12. **Centroid locking** — HDBSCAN centroids frozen after baseline, KNN for incremental assignment

---

## 8. Dependencies

```
numpy, pandas, pyarrow, scipy, scikit-learn, Pillow,
torch, torchvision, transformers, safetensors, huggingface_hub,
faiss-cpu, hdbscan, umap-learn, pynndescent,
matplotlib, seaborn, tqdm, PyYAML, requests, pytest,
sentence-transformers
```

No new dependencies added — Cluster Tracker uses existing `faiss-cpu` and `numpy`.
No Apify SDK needed — raw `requests` calls to the Apify REST API.

---

## 9. Multi-Niche Monitoring

`account.txt` currently spans four niches (~81 accounts):

| Niche | Example accounts |
|-------|------------------|
| Food / cafe / dessert | `food52`, `tasty`, `halfbakedharvest`, `frenchpress.latteart` |
| Fashion / street style | `voguemagazine`, `highsnobiety`, `styledumonde`, `matildadjerf`, `tokyofashion` |
| Photography | `natgeo`, `magnumphotos`, `jordi.koalitic`, `alan_schaller`, `moodygrams` |
| Beauty / skincare | `hudabeauty`, `rarebeauty`, `glossier`, `ctilburymakeup` |

**How cross-niche works:** the pipeline is niche-agnostic — CLIP clusters
images by visual semantics, so posts from different niches naturally land
in separate clusters. RAG retrieval is semantic, so a query like "What
street style aesthetics are trending?" surfaces fashion clusters only.

### Adding another niche

1. Append Instagram URLs/usernames to `account.txt`
   (one per line, no comment lines — the parser is literal)
2. Run `python -m src.data_collector` (incremental) or `--baseline`
   for a clean single-niche registry
3. Ask niche-specific queries

All new accounts were verified as real, active public Instagram accounts
(2026-08). Note: more accounts = larger Apify fetch per run; use `--days N`
to bound the scan window.

---

_Last updated: 2026-08-24_
