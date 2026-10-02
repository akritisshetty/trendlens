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

# User accounts (login) — SQLite, created on first use
AUTH_DB_PATH = DATA_DIR / "users.db"

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
# Model selection
#
# Every pretrained model the pipeline uses is named HERE and nowhere else, so
# swapping one cannot leave a second hard-coded copy behind. Before this
# existed the same CLIP checkpoint string was duplicated in
# src/embeddings.py, src/style_tags.py, benchmark_algorithms.py and
# baseline_comparison.py, which is how the style-tag cache came to be written
# with a model name it never checked against.
#
# Each value is chosen by measurement on the real corpus, not by default:
#   * CLIP   scripts/select_models.py      (see model_selection_summary.json)
#   * text   scripts/select_text_model.py  (see text_model_selection_results.csv)
#   * BLIP   scripts/select_caption_model.py
#
# Overridable via environment so a sweep can vary them without editing code:
#   TRENDLENS_CLIP_MODEL / TRENDLENS_RAG_MODEL / TRENDLENS_BLIP_MODEL
# ──────────────────────────────────────────────────────────────────────────

#: Image+text encoder. Supplies the image embeddings that clustering runs on
#: AND the text tower used for zero-shot style tagging and cluster naming, so
#: image and text vectors always share one space.
#:
#: Measured over 5 encoders x 18 clustering configurations
#: (scripts/select_models.py, model_selection_summary.json). ViT-B/32 wins on
#: the primary metric and on cluster stability; SigLIP2-base was measured too
#: and did NOT displace it (rho 0.141 vs 0.166, stability ARI 0.56 vs 0.73,
#: ~2x the encode time), so the newer model family is not automatically better
#: here. ViT-L/14 is statistically indistinguishable on rho but 3x the encode
#: cost and slower per-image on CPU, so it was not adopted.
CLIP_MODEL: str = os.environ.get(
    "TRENDLENS_CLIP_MODEL", "openai/clip-vit-base-patch32"
)

#: RAG chunk/query encoder. Separate from CLIP on purpose: it only ever
#: compares text to text, and the strongest general text-retrieval checkpoint
#: is not the strongest image-text one.
#:
#: Chosen by scripts/select_text_model.py on the project's own retrieval task
#: (BLIP captions of cluster-member images as queries, the representative
#: image excluded so no query is a copy of indexed text). Over 6 candidates
#: bge-base-en-v1.5 wins on every metric — see
#: text_model_selection_results.csv:
#:     all-MiniLM-L6-v2 (incumbent)  R@1 0.347  R@3 0.542  MRR 0.502
#:     BAAI/bge-base-en-v1.5          R@1 0.431  R@3 0.694  MRR 0.590
#: bge models need no query/passsage prefix (that is an e5-family convention,
#: handled explicitly in the selection script).
RAG_EMBED_MODEL: str = os.environ.get(
    "TRENDLENS_RAG_MODEL", "BAAI/bge-base-en-v1.5"
)

#: VLM captioner. Supplies the raw captions that cluster names, keywords and
#: RAG chunk text are derived from — so caption quality is user-visible output,
#: not an internal detail.
#:
#: Chosen by scripts/select_caption_model.py on 60 real corpus images
#: (caption_model_selection_results.csv):
#:     blip-image-captioning-base   CLIPScore 0.2740  distinct-2 0.704  5% degenerate
#:     blip-image-captioning-large  CLIPScore 0.2775  distinct-2 0.923  0% degenerate
#: The CLIPScore gap is within noise at n=60; the reasons to prefer large are
#: the markedly higher lexical diversity (cluster names stop repeating each
#: other in the UI) and the absence of BLIP-base's degenerate repetitions.
#: The cost is real and is accepted knowingly: large captions at ~0.04 img/s
#: vs ~0.53 img/s for base, so interpreting ~19 clusters costs roughly half an
#: hour instead of ~2.5 min. Expect that in the hardware notes.
BLIP_MODEL: str = os.environ.get(
    "TRENDLENS_BLIP_MODEL", "Salesforce/blip-image-captioning-large"
)

#: Sidecar recording which encoder produced embeddings.npy, so a stale artifact
#: from a different checkpoint is detected instead of silently reused.
INSTAGRAM_EMBEDDING_MANIFEST_PATH: Path = (
    INSTAGRAM_DIR / "embeddings_manifest.json"
)

# ──────────────────────────────────────────────────────────────────────────
# Image preprocessing
# ──────────────────────────────────────────────────────────────────────────
#: 224×224 is the native input of every CLIP checkpoint evaluated by
#: scripts/select_models.py (ViT-B/32, ViT-B/16, ViT-L/14, SigLIP2-224), so no
#: candidate is penalised by resampling.
IMAGE_RESIZE: tuple[int, int] = (224, 224)
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
# Clustering hyperparameters
#
# Production values are 15 / 5 / 3, and getting there took two sweeps that
# disagreed. Both are kept honest below rather than the losing one deleted.
#
# scripts/sweep_hyperparameters.py ranked on separation (eta^2) behind a
# stability gate and landed on 15 / 5 / 3.
# scripts/select_models.py ranks on mean Spearman rho and nominated
# 20 / 5 / 3 — but it averaged only five account splits, and its own
# docstring says five splits is too few to choose on.
#
# So the choice was re-measured properly: 80 held-out account splits, the
# SAME splits scored for every configuration (paired), which removes the
# split-to-split noise that the 5-seed mean was riding:
#
#     UMAP   10   rho 0.170 +- 0.132
#     UMAP   15   rho 0.248 +- 0.113   <- production
#     UMAP   20   rho 0.177 +- 0.118
#
#   15 vs 20: +0.071, better on 65/80 splits, paired t = 6.96, p = 9e-10
#   15 vs 10: +0.079, better on 64/80 splits, paired p < 0.001
#
# The nominally "selected" 20-d configuration is actually WORSE, and at
# min_cluster_size=8 — the row select_models.py ranked first — it is worse
# again (rho 0.084 over 40 splits). Five seeds could not see that.
# The RAG index, trends and README numbers are all built on 15 / 5 / 3.
# ──────────────────────────────────────────────────────────────────────────
UMAP_COMPONENTS = int(os.environ.get("TRENDLENS_UMAP_COMPONENTS", "15"))
HDBSCAN_MIN_CLUSTER_SIZE = int(
    os.environ.get("TRENDLENS_HDBSCAN_MIN_CLUSTER_SIZE", "5")
)
HDBSCAN_MIN_SAMPLES = int(os.environ.get("TRENDLENS_HDBSCAN_MIN_SAMPLES", "3"))
HDBSCAN_SELECTION_METHOD = os.environ.get("TRENDLENS_HDBSCAN_METHOD", "eom")

#: RAG retrieval depth. k=5 sits at the knee of the measured curve on this
#: corpus: hit-rate keeps climbing to k=8 (0.71 -> 0.79) but MRR is flat from
#: k=5 (0.335 vs 0.345), so k=8 adds context without adding rank quality.
RAG_RETRIEVAL_K = int(os.environ.get("TRENDLENS_RAG_K", "5"))

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
