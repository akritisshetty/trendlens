# TrendLens — Commands Reference

> All commands assume you are in the **project root**: `/path/to/trendlens/`

---

## Quick Start (one command)

```bash
# Run everything: pipeline → backend → frontend → open browser
./run.sh          # default: 10-day window
./run.sh 7        # override to 7-day window
```

This single script:
1. Runs the full data pipeline (fetch → embed → cluster → caption → trends → RAG)
2. Starts the Python backend on `:8000`
3. Starts the React frontend on `:3000`
4. Opens your browser to `http://localhost:3000`

Press `Ctrl+C` to stop both servers.

---

## Prerequisites (one-time setup)

```bash
# Python venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Frontend deps
cd frontend && npm install && cd ..

# Environment
cp .env.example .env
# Edit .env and set APIFY_API_TOKEN (required for Instagram scraping)
# Get one free at: https://apify.com/account#/integrations
```

---

## Individual Commands

### Data Pipeline

```bash
source venv/bin/activate

# Incremental run (default) — KNN assign new images to existing clusters
python -m src.data_collector

# Override scan window
python -m src.data_collector --days 7

# Force full re-cluster from scratch (baseline)
python -m src.data_collector --baseline
```

Pipeline stages (incremental):
1. Fetch posts from `account.txt` via Apify
2. Download only new images
3. CLIP image embeddings (new images only)
4. FAISS KNN assignment to existing clusters
5. Detect emerging micro-clusters from unassigned images
6. Photography-style tagging over all embeddings
7. Temporal trend analysis (stable cluster IDs)
8. FAISS RAG index rebuild

Pipeline stages (baseline — `--baseline`):
1. Fetch posts from `account.txt` via Apify
2. Download images
3. CLIP image embeddings
4. UMAP dim reduction + HDBSCAN clustering
5. Lock centroids → build cluster registry with stable UUIDs
6. BLIP captioning of representative images
7. CLIP zero-shot photography-style tagging (per-cluster execution profile)
8. Temporal trend analysis
9. FAISS RAG index build

### Backend API (port 8000)

```bash
python -m src.api
```

**Endpoints:**
- `GET  /api/health` — service status
- `POST /api/rag-query` — `{"query": "..."}` → answer with evidence
- `GET  /api/trends` — top emerging trends
- `GET  /api/clusters` — all cluster interpretations
- `GET  /api/live-trends` — real Instagram emerging themes
- `GET  /api/live-images?name=` — serve downloaded Instagram images

**Smoke test:**
```bash
curl -s http://127.0.0.1:8000/api/health
curl -s -X POST http://127.0.0.1:8000/api/rag-query \
  -H 'Content-Type: application/json' \
  -d '{"query": "What cafe aesthetic is rising this week?"}'
```

### Frontend (port 3000)

```bash
cd frontend && npx tsx server.ts
```

Requires the backend to be running on `:8000`. All `/api/*` requests are proxied.

### RAG Query (CLI)

```bash
python -m src.rag "What cafe aesthetic is rising this week?"
python -m src.rag "What food photography styles are trending on Instagram?"
python -m src.rag "What street style aesthetics are trending on Instagram?"
python -m src.rag "What editing and colour grading styles are trending in photography?"
python -m src.rag "What makeup looks are trending on Instagram?"
```

### Tests

```bash
python -m pytest tests/ -q
```

---

## Adding / Changing Accounts

Edit `account.txt` — one Instagram URL or username per line:

```
https://www.instagram.com/username
@username
username
```

All three formats work. Then re-run `python -m src.data_collector`.

---

## Niches Currently Monitored

`account.txt` covers four niches (~81 accounts):

| Niche | Accounts (examples) |
|-------|---------------------|
| Food / cafe / dessert | `food52`, `tasty`, `halfbakedharvest`, `frenchpress.latteart`, `crumblcookies` |
| Fashion / street style | `voguemagazine`, `highsnobiety`, `styledumonde`, `matildadjerf`, `tokyofashion` |
| Photography | `natgeo`, `magnumphotos`, `jordi.koalitic`, `alan_schaller`, `moodygrams` |
| Beauty / skincare | `hudabeauty`, `rarebeauty`, `glossier`, `ctilburymakeup`, `theordinary` |

CLIP clusters by visual semantics, so niches form separate clusters — niche-specific queries retrieve only the relevant ones.

---

## Adding a New Niche

1. Append Instagram URLs/usernames to `account.txt` (one per line, no comments)
2. Run `python -m src.data_collector`
3. Query: e.g. `python -m src.rag "What travel photography aesthetics are trending?"`

To start fresh for a single niche: swap the file contents and run `python -m src.data_collector --baseline`.

---

## Output Artifacts

All generated in `data/instagram/` and `artifacts/` (regenerable via `data_collector`):

| File | Description |
|------|-------------|
| `data/instagram/posts.parquet` | Instagram posts metadata (deduped, real timestamps) |
| `data/instagram/all_posts.parquet` | Cumulative posts across all runs (incremental mode) |
| `data/instagram/images/<post_id>.jpg` | Downloaded Instagram images |
| `data/instagram/embeddings.npy` | CLIP embeddings (N, 512) |
| `data/instagram/embed_meta.parquet` | Aligned post metadata for embeddings |
| `data/instagram/trends.json` | Emerging themes with growth + engagement |
| `data/instagram/rag_index.faiss` | FAISS RAG index |
| `data/instagram/rag_chunks.json` | RAG text chunks for retrieval |
| `artifacts/cluster_registry.json` | Stable cluster IDs, centroids, metadata |
| `artifacts/centroid_index.faiss` | FAISS index over locked centroids |
| `artifacts/centroid_index.json` | Stable ID mapping (FAISS position → `cls_*`) |

---

_Last updated: 2026-08-24_
