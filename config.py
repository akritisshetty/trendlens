"""
TrendLens — central configuration.

Every pipeline stage reads paths, dataset schema mappings and experiment
parameters from here so nothing is hard-coded deep inside the code.

DATA INTEGRITY NOTICE
---------------------
The SMPD download available locally contains ONLY real image files
(train/) plus a real file-path index (train_img_filepath.txt). All
engagement metadata (likes, comments, timestamps, tags, geo, categories)
is SYNTHETIC — generated for demo purposes and marked is_synthetic=True.

To keep the project scientifically honest:
  * Results derived from synthetic metadata are labelled "SYNTHETIC DEMO"
    and must never be presented as research findings.
  * The temporal rigging used in the legacy pipeline (category-biased
    Gaussian timestamps designed to force Rising/Stable/Declining
    lifecycles) is DISABLED in this codebase. Timestamps are treated as
    synthetic demo data only.
"""

from __future__ import annotations

import os
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """
    Minimal stdlib .env loader (no python-dotenv dependency).

    Loads KEY=VALUE lines from <project_root>/.env into the environment
    WITHOUT overriding variables that are already set. Values may be quoted.
    """
    if not path.exists():
        return
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


ROOT: Path = Path(__file__).resolve().parent
_load_dotenv(ROOT / ".env")

# ──────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
METADATA_DIR = DATA_DIR / "metadata"

ARTIFACTS_DIR = ROOT / "artifacts"
CLUSTER_MODELS_DIR = ARTIFACTS_DIR / "cluster_models"
CLUSTER_METADATA_DIR = ARTIFACTS_DIR / "cluster_metadata"
FAISS_DIR = ARTIFACTS_DIR / "faiss"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
CLUSTER_REGISTRY_PATH = ARTIFACTS_DIR / "cluster_registry.json"
CENTROID_FAISS_PATH = ARTIFACTS_DIR / "centroid_index.faiss"

LEGACY_OUTPUTS_DIR = ROOT / "trendlens_outputs"

# ──────────────────────────────────────────────────────────────────────────
# Live (real-time) trend data
# ──────────────────────────────────────────────────────────────────────────
# Unlike the SMPD sample (synthetic engagement/timestamps), live data comes
# from real official feeds (Reddit etc.) with REAL post timestamps and REAL
# engagement. It is stored separately and labelled as such — never mixed into
# the synthetic demo corpus.
LIVE_DIR = DATA_DIR / "live"
LIVE_IMAGES_DIR = LIVE_DIR / "images"
LIVE_EMBEDDINGS_PATH = LIVE_DIR / "live_embeddings.npy"
LIVE_POSTS_PATH = LIVE_DIR / "live_posts.parquet"
LIVE_TRENDS_PATH = ARTIFACTS_DIR / "cluster_metadata" / "live_trends.json"

# Default subreddits watched for live trends. Override with TRENDLENS_SUBREDDITS
# (comma-separated) — e.g. "foodporn,coffee,streetwear,sneakers".
LIVE_SUBREDDITS = [s.strip() for s in os.environ.get(
    "TRENDLENS_SUBREDDITS", "foodporn,coffee"
).split(",") if s.strip()]
# How far back to scan, and the recent/prior windows (days) used for growth.
LIVE_SCAN_DAYS = int(os.environ.get("TRENDLENS_LIVE_SCAN_DAYS", "14"))
LIVE_RECENT_WINDOW_DAYS = int(os.environ.get("TRENDLENS_LIVE_RECENT_DAYS", "7"))
LIVE_PER_SUBREDDIT_LIMIT = int(os.environ.get("TRENDLENS_LIVE_LIMIT", "50"))

# Live source selector (TRENDLENS_LIVE_SOURCE):
#   "auto"      → try Reddit first, fall back to Wikimedia Commons (key-free)
#   "reddit"    → Reddit only (needs unblocked network or OAuth creds)
#   "wikimedia" → Wikimedia Commons only (key-free, real timestamps, no engagement)
LIVE_SOURCE = (os.environ.get("TRENDLENS_LIVE_SOURCE") or "auto").strip().lower()
LIVE_WIKIMEDIA_QUERIES = [q.strip() for q in os.environ.get(
    "TRENDLENS_WIKIMEDIA_QUERIES",
    "latte art,coffee,street food,breakfast",
).split(",") if q.strip()]
LIVE_WIKIMEDIA_LIMIT = int(os.environ.get("TRENDLENS_WIKIMEDIA_LIMIT", "10"))
# Upper bound on how many live posts are downloaded/embedded per run (recent
# first). Wikimedia throttles anonymous image downloads, so runs stay bounded.
LIVE_MAX_EMBED_POSTS = int(os.environ.get("TRENDLENS_LIVE_MAX_EMBED", "40"))
# Commons uploads are sparse per topic, so its scan + growth windows are wider
# than Reddit's. Growth is still a real recent-vs-prior comparison on real
# upload timestamps — just measured over these wider windows.
LIVE_WIKIMEDIA_SCAN_DAYS = int(os.environ.get("TRENDLENS_WIKIMEDIA_DAYS", "90"))
LIVE_WIKIMEDIA_RECENT_DAYS = int(os.environ.get("TRENDLENS_WIKIMEDIA_RECENT_DAYS", "30"))

LIVE_DATA_WARNING = (
    "REAL LIVE DATA: posts, upload timestamps and images come from a real "
    "public image feed (official Reddit feed and/or Wikimedia Commons) — "
    "not the synthetic demo corpus. Images belong to their posters. "
    "Upvote/comment engagement is Reddit-only and is absent from other "
    "sources. Views are demo-only."
)

# ──────────────────────────────────────────────────────────────────────────
# Instagram (Apify) — primary data source
# ──────────────────────────────────────────────────────────────────────────
INSTAGRAM_DIR = DATA_DIR / "instagram"
INSTAGRAM_IMAGES_DIR = INSTAGRAM_DIR / "images"
INSTAGRAM_POSTS_PATH = INSTAGRAM_DIR / "posts.parquet"
INSTAGRAM_EMBEDDINGS_PATH = INSTAGRAM_DIR / "embeddings.npy"
INSTAGRAM_TRENDS_PATH = INSTAGRAM_DIR / "trends.json"
INSTAGRAM_RAG_INDEX_PATH = INSTAGRAM_DIR / "rag_index.faiss"
INSTAGRAM_RAG_CHUNKS_PATH = INSTAGRAM_DIR / "rag_chunks.json"
INSTAGRAM_ACCOUNTS_FILE = ROOT / "account.txt"
INSTAGRAM_SCAN_DAYS = int(os.environ.get("TRENDLENS_INSTAGRAM_DAYS", "10"))
APIFY_API_TOKEN = os.environ.get("APIFY_API_TOKEN", "")

INSTAGRAM_DATA_WARNING = (
    "REAL INSTAGRAM DATA: posts, timestamps, likes, comments, views, "
    "hashtags, content types, and images come from public Instagram "
    "accounts via the Apify API. Not synthetic."
)

for _d in (
    RAW_DIR,
    PROCESSED_DIR,
    EMBEDDINGS_DIR,
    METADATA_DIR,
    CLUSTER_MODELS_DIR,
    CLUSTER_METADATA_DIR,
    FAISS_DIR,
    FIGURES_DIR,
    LIVE_DIR,
    LIVE_IMAGES_DIR,
    INSTAGRAM_DIR,
    INSTAGRAM_IMAGES_DIR,
):
    _d.mkdir(parents=True, exist_ok=True)

# Legacy location of the actual image files and path index.
# These are the ONLY genuinely real dataset signals available locally.
IMAGE_ROOT: Path = ROOT / "train"
IMAGE_PATH_LIST: Path = ROOT / "train_img_filepath.txt"

# ──────────────────────────────────────────────────────────────────────────
# Dataset schema mapping
#
# Map the dataset's actual column names onto TrendLens' canonical names.
# Set a field to None when the column is not present in the source.
# ──────────────────────────────────────────────────────────────────────────
DATASET_CONFIG = {
    "image_column": "image_path",
    "caption_column": None,        # no captions in local SMPD files
    "timestamp_column": "timestamp",
    "likes_column": "likes",
    "comments_column": "comments",
    "post_id_column": "post_id",
    "user_id_column": "user_id",
}

# Where metadata for the current run lives. The legacy pipeline writes
# to trendlens_outputs/metadata.csv; the new pipeline writes a parquet
# version under data/metadata/ with the row index aligned to embeddings.
METADATA_CSV_PATH: Path = LEGACY_OUTPUTS_DIR / "metadata.csv"
METADATA_PARQUET_PATH: Path = METADATA_DIR / "metadata.parquet"

# ──────────────────────────────────────────────────────────────────────────
# Dataset / sampling
# ──────────────────────────────────────────────────────────────────────────
N_IMAGES: int = 5000                # default pipeline subset size
MAX_IMAGES: int = 69226             # actual number of image files present
RANDOM_SEED: int = 42               # deterministic sampling everywhere
VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ──────────────────────────────────────────────────────────────────────────
# Image preprocessing
# ──────────────────────────────────────────────────────────────────────────
IMAGE_RESIZE: tuple[int, int] = (224, 224)   # CLIP ViT-B/32 native size
CACHE_VALIDATED_PATHS = True

# ──────────────────────────────────────────────────────────────────────────
# Data-integrity labelling
# ──────────────────────────────────────────────────────────────────────────
# True => every report/artifact asserts that engagement metadata is
# synthetic demo data and must not be quoted as research findings.
SYNTHETIC_DATA_WARNING = (
    "SYNTHETIC DEMO DATA: likes/comments/timestamps/tags/geo are generated, "
    "not real. Results derived from them are demonstration only and are NOT "
    "research findings."
)


# ──────────────────────────────────────────────────────────────────────────
# Experiment bookkeeping
# ──────────────────────────────────────────────────────────────────────────
def experiment_config(extra: dict | None = None) -> dict:
    """Return a reproducible experiment-config manifest."""
    cfg = {
        "random_seed": RANDOM_SEED,
        "n_images": N_IMAGES,
        "dataset_schema": DATASET_CONFIG,
        "synthetic_metadata": True,
        "synthetic_data_warning": SYNTHETIC_DATA_WARNING,
    }
    if extra:
        cfg.update(extra)
    return cfg
