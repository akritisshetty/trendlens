"""
data_collector.py
-----------------
Full Instagram data pipeline orchestrator.

Run with: ``python -m src.data_collector``

Pipeline stages:
  0. Clean existing data (fresh start every run)
  1. Fetch posts from Instagram via Apify (last N days)
  2. Download images locally
  3. CLIP image embeddings
  4. HDBSCAN clustering
  5. BLIP captioning of representative images
  6. Temporal trend analysis (daily counts, growth, emerging score)
  7. Build RAG index (FAISS + sentence-transformer)
  8. Save all artifacts

Every run deletes the previous data and re-downloads so the system
always works with the most recent posts.

INTEGRITY
---------
* All data comes from Instagram via Apify — real timestamps, real
  engagement (likes, comments, views), real images.
* Cluster names/descriptions are VLM interpretations (BLIP), not ground
  truth.
* No metrics are fabricated — missing data is 0 or empty.
"""

from __future__ import annotations

import json
import re
import shutil
import string
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pandas as pd

import config
from src.trend_definition import DEFAULT_DEFINITION

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.trend_definition import TrendDefinition

STOPWORDS = set(
    "the a an and or but of in on for with at by to from this that these those "
    "it is are was were be been being i we you they he she has have had my our "
    "your their its just like get got one two new day night time post pics pic "
    "photo im dont made make use using used first last best top every more most "
    "over under again about into also what your link bio story".split()
)


def _json_default(obj: Any) -> Any:
    """JSON serializer fallback for numpy / pandas scalar types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

# ──────────────────────────────────────────────────────────────────────────
# Cleanup: remove all existing data before a fresh run
# ──────────────────────────────────────────────────────────────────────────
def clean_instagram_data() -> None:
    """Delete all previously collected Instagram data so each run starts fresh."""
    targets = [
        config.INSTAGRAM_POSTS_PATH,
        config.INSTAGRAM_EMBEDDINGS_PATH,
        config.INSTAGRAM_TRENDS_PATH,
        config.INSTAGRAM_RAG_INDEX_PATH,
        config.INSTAGRAM_RAG_CHUNKS_PATH,
        config.INSTAGRAM_DIR / "embed_meta.parquet",
    ]
    for p in targets:
        if p.exists():
            p.unlink()
    if config.INSTAGRAM_IMAGES_DIR.exists():
        shutil.rmtree(config.INSTAGRAM_IMAGES_DIR)
    print("[collector] cleaned existing Instagram data")


# ──────────────────────────────────────────────────────────────────────────
# Stage 1: Fetch + save posts
# ──────────────────────────────────────────────────────────────────────────
def fetch_and_save(days: int = 10) -> pd.DataFrame:
    """Fetch Instagram posts via Apify and save to parquet."""
    from src.apify_client import fetch_instagram_posts

    posts = fetch_instagram_posts(days=days)
    if not posts:
        print("[collector] no posts fetched from Instagram")
        return pd.DataFrame()

    df = pd.DataFrame(posts)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["post_id", "image_url"])
    df = df.drop_duplicates(subset=["post_id"], keep="last")
    df = df.sort_values("timestamp").reset_index(drop=True)

    config.INSTAGRAM_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.INSTAGRAM_POSTS_PATH, index=False)
    print(f"[collector] saved {len(df)} posts to {config.INSTAGRAM_POSTS_PATH}")
    return df


def load_posts() -> Optional[pd.DataFrame]:
    """
    Load previously saved Instagram posts, with engagement sanitised.

    ``clean_engagement`` is applied here (and not only at scrape time) so that
    parquet files written by earlier versions — which persisted the Instagram
    ``likesCount = -1`` sentinel — are cleaned on read too. That keeps the fix
    effective without requiring a full re-scrape.
    """
    if not config.INSTAGRAM_POSTS_PATH.exists():
        return None
    df = pd.read_parquet(config.INSTAGRAM_POSTS_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    from src.data_quality import clean_engagement

    df = clean_engagement(df)
    n_hidden = int((df["likes"].isna()).sum()) if "likes" in df.columns else 0
    if n_hidden:
        print(
            f"[collector] {n_hidden} post(s) have hidden like counts "
            f"(excluded from engagement stats, not treated as 0)"
        )
    return df


# ──────────────────────────────────────────────────────────────────────────
# Stage 2: Download images
# ──────────────────────────────────────────────────────────────────────────
def _local_image_path(post_id: str, image_url: str) -> Path:
    ext = Path(image_url.split("?")[0]).suffix.lower()
    if ext not in (".jpg", ".jpeg", ".png", ".webp"):
        ext = ".jpg"
    return config.INSTAGRAM_IMAGES_DIR / f"{post_id}{ext}"


def download_images(df: pd.DataFrame) -> list[str]:
    """Download Instagram images that are not yet local. Returns saved filenames."""
    import requests as _requests

    config.INSTAGRAM_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    for i, (_, row) in enumerate(df.iterrows()):
        if i > 0:
            time.sleep(0.5)
        dest = _local_image_path(row["post_id"], row["image_url"])
        if dest.exists():
            continue
        for attempt in range(3):
            try:
                r = _requests.get(row["image_url"], timeout=30)
                if r.status_code == 200 and len(r.content) > 500:
                    dest.write_bytes(r.content)
                    saved.append(dest.name)
                    break
                if r.status_code == 429:
                    time.sleep(3 * (attempt + 1))
            except Exception:
                time.sleep(2 * (attempt + 1))
    return saved


# ──────────────────────────────────────────────────────────────────────────
# Stage 3: CLIP embeddings
# ──────────────────────────────────────────────────────────────────────────
def embed_images(df: pd.DataFrame) -> tuple[Optional[np.ndarray], pd.DataFrame]:
    """CLIP-embed all downloaded Instagram images."""
    import torch
    from PIL import Image

    from src.embeddings import l2_normalize, load_clip

    model, processor, device = load_clip()
    embs: list[np.ndarray] = []
    keep: list[str] = []

    with torch.no_grad():
        for _, row in df.iterrows():
            path = _local_image_path(row["post_id"], row["image_url"])
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                continue
            inputs = processor(images=img, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out = model.get_image_features(**inputs)
            if hasattr(out, "pooler_output"):
                out = out.pooler_output
            feats = out.detach().cpu().numpy()
            embs.append(feats[0])
            keep.append(row["post_id"])

    if not embs:
        return None, df.iloc[0:0]

    emb = l2_normalize(np.vstack(embs).astype("float32"))
    aligned = df[df["post_id"].isin(keep)].reset_index(drop=True)
    np.save(config.INSTAGRAM_EMBEDDINGS_PATH, emb)
    aligned.to_parquet(config.INSTAGRAM_DIR / "embed_meta.parquet", index=False)
    # Record which encoder produced this matrix. Without it, a later run that
    # changes config.CLIP_MODEL has no way to tell a fresh artifact from one
    # built by the previous checkpoint, and every downstream cache keys off the
    # row count — which does not change.
    config.INSTAGRAM_EMBEDDING_MANIFEST_PATH.write_text(
        json.dumps(
            {
                "model": config.CLIP_MODEL,
                "dim": int(emb.shape[1]),
                "n_rows": int(emb.shape[0]),
                "n_posts_total": int(len(df)),
            },
            indent=1,
        )
    )
    print(f"[collector] embedded {emb.shape[0]} images → {emb.shape}")
    return emb, aligned


_CLIP_TEXT_CACHE: dict[str, np.ndarray] = {}


def _clip_text_vecs(prompts: list[str]) -> dict[str, np.ndarray]:
    """L2-normalised CLIP text embeddings for many prompts, in one forward pass.

    Same shared embedding space as the stored image vectors. Only the prompts
    not already in ``_CLIP_TEXT_CACHE`` are encoded, so the concept vocabulary
    is embedded once per process and reused by every cluster afterwards.
    Prompts that cannot be encoded are simply absent from the result.
    """
    todo = [p for p in dict.fromkeys(prompts) if p not in _CLIP_TEXT_CACHE]
    if todo:
        try:
            import torch

            from src.embeddings import load_clip

            model, processor, device = load_clip()
            with torch.no_grad():
                inp = processor(text=todo, return_tensors="pt", padding=True)
                inp = {k: v.to(device) for k, v in inp.items()}
                out = model.get_text_features(**inp)
                if hasattr(out, "pooler_output"):
                    out = out.pooler_output
                mat = out.detach().cpu().numpy().astype("float32")
            mat = mat / np.maximum(
                np.linalg.norm(mat, axis=-1, keepdims=True), 1e-12
            )
            for prompt, vec in zip(todo, mat):
                _CLIP_TEXT_CACHE[prompt] = vec
        except Exception:  # noqa: BLE001 — caller falls back to centroid/likes
            pass
    return {p: _CLIP_TEXT_CACHE[p] for p in prompts if p in _CLIP_TEXT_CACHE}


def _clip_text_vec(prompt: str) -> Optional[np.ndarray]:
    """L2-normalised CLIP text embedding for a prompt (cached). Same shared
    embedding space as the stored image vectors."""
    return _clip_text_vecs([prompt]).get(prompt)


# ──────────────────────────────────────────────────────────────────────────
# Stage 4: HDBSCAN clustering
# ──────────────────────────────────────────────────────────────────────────
def cluster_posts(
    emb: np.ndarray, df: pd.DataFrame
) -> tuple[np.ndarray, dict[int, list[int]]]:
    """Cluster posts by CLIP embeddings. Returns labels and cluster→indices map."""
    import hdbscan

    n = len(df)
    from sklearn.preprocessing import normalize
    emb_norm = normalize(emb, norm="l2")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=max(2, min(3, n // 5)),
        min_samples=1,
        metric="euclidean",
        cluster_selection_epsilon=0.6,
    )
    labels = clusterer.fit_predict(np.asarray(emb_norm, dtype="float32"))

    clusters: dict[int, list[int]] = {}
    for i, label in enumerate(labels):
        if label < 0:
            continue
        clusters.setdefault(int(label), []).append(i)

    n_clusters = len(clusters)
    noise_pct = float((labels == -1).sum()) / max(len(labels), 1) * 100
    print(f"[collector] {n_clusters} clusters, {noise_pct:.1f}% noise")
    return labels, clusters


# ──────────────────────────────────────────────────────────────────────────
# Stage 5: Captioning + cluster summarization
# ──────────────────────────────────────────────────────────────────────────
def _title_keywords(text: str) -> list[str]:
    # Strip emojis and non-ASCII characters
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    tokens = re.sub(f"[{re.escape(string.punctuation)}]", " ", text.lower()).split()
    seen: list[str] = []
    for t in tokens:
        if t not in STOPWORDS and len(t) > 2 and t not in seen:
            seen.append(t)
    return seen[:10]


_blip_cache: tuple | None = None


def _style_scores_for(emb: np.ndarray):
    """Zero-shot photography-style scores for embeddings, or None on failure."""
    try:
        from src.style_tags import compute_style_scores
        return compute_style_scores(np.asarray(emb, dtype="float32"))
    except Exception as e:  # noqa: BLE001 — style tagging must never break collection
        print(f"[collector] style tagging skipped ({e})")
        return None


def _blip_caption(path: Path) -> tuple[str, float]:
    global _blip_cache
    try:
        from PIL import Image
        from src.interpretation import caption_image, load_blip
        if _blip_cache is None:
            _blip_cache = load_blip()
        model, processor, device = _blip_cache
        img = Image.open(path).convert("RGB")
        caption = caption_image(model, processor, img, device=device)
        return caption, 1.0
    except Exception:
        return "", 0.0


def _visual_concept_match(
    member_emb: Optional[np.ndarray],
) -> tuple[Optional[str], Optional[np.ndarray]]:
    """Zero-shot CLIP name for a cluster, scored over its member images.

    Returns ``(concept, per_member_sims)``. ``(None, None)`` when CLIP is
    unavailable, so callers fall back to caption keywords.

    This was factored out of ``_build_summaries_from_registry`` because the
    concept vocabulary was only ever consulted on the incremental pipeline
    path. ``summarize_clusters`` — the function
    ``scripts/rebuild_trends.py`` actually calls to produce the shipped
    artifacts — named every cluster from caption keywords alone, which is why
    production theme names read "heard her thegirljt" and "good fried chicken":
    not a bad model, just a naming path that never ran.
    """
    if member_emb is None or len(member_emb) == 0:
        return None, None
    mat = member_emb.astype("float32")
    prompts = ["a photo of " + concept for concept in _VISUAL_CONCEPTS]
    vecs = _clip_text_vecs(prompts)
    scored = [
        (concept, mat @ vecs["a photo of " + concept])
        for concept in _VISUAL_CONCEPTS
        if "a photo of " + concept in vecs
    ]
    if not scored:
        return None, None
    means = np.array([float(s.mean()) for _, s in scored])
    best = int(np.argmax(means))
    return scored[best][0], scored[best][1]


def _concept_display_name(concept: str) -> str:
    """Human-facing form of a concept prompt.

    Concepts are phrased as CLIP prompts ("a close-up of a made-up face"),
    which reads badly as a theme headline, so the leading article is dropped.
    """
    text = re.sub(r"^(a|an|the)\s+", "", concept.strip(), flags=re.IGNORECASE)
    return text[:1].upper() + text[1:]


def summarize_clusters(
    df: pd.DataFrame,
    labels: np.ndarray,
    clusters: dict[int, list[int]],
    emb: np.ndarray,
    top_k_caption: int = 3,
    style_scores=None,
) -> list[dict[str, Any]]:
    """Generate cluster summaries with BLIP captions and keywords.

    BLIP captioning strategy: find the mathematical centroid of the
    HDBSCAN group and caption only the top *top_k_caption* closest
    vectors. This keeps descriptions concentrated on the core visual
    style while captioning only 3 images instead of the full cluster.

    ``style_scores`` (optional, from src.style_tags.compute_style_scores)
    adds a per-cluster *photography execution* profile — how the images
    are shot (framing, lighting, grading) as opposed to what is in them.
    """
    summaries: list[dict[str, Any]] = []
    seen_names: set[str] = set()

    for cid, indices in sorted(clusters.items()):
        members = df.iloc[indices]
        member_emb = emb[indices]

        # ── Centroid-based representative selection ──
        # Compute cluster centroid (mean embedding)
        centroid = member_emb.mean(axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 0:
            centroid = centroid / centroid_norm

        # Cosine distance from each member to centroid
        dists = 1.0 - (member_emb @ centroid)
        # Top-k closest to centroid
        k = min(top_k_caption, len(indices))
        closest_pos = np.argsort(dists)[:k]
        closest_df_indices = [indices[p] for p in closest_pos]
        closest_members = df.iloc[closest_df_indices]

        # BLIP caption of top-k closest images
        captions: list[str] = []
        for _, row in closest_members.iterrows():
            local = _local_image_path(row["post_id"], row["image_url"])
            if local.exists():
                cap, _ = _blip_caption(local)
                if cap:
                    captions.append(cap)

        # Combine captions: take the most descriptive one (longest)
        combined_caption = max(captions, key=len) if captions else ""

        # Representative: highest *known* engagement among top-k. The old
        # `likes.sum() > 0 / likes.idxmax()` path could select a post whose
        # count was merely hidden, because the -1 sentinel beat a real 0.
        from src.data_quality import best_post_by_engagement

        best = best_post_by_engagement(closest_members)
        if best is None:
            best = closest_members.loc[closest_members["timestamp"].idxmax()]

        # Keywords from captions of all members
        all_captions_text = " ".join(members["caption"].fillna("").tolist())
        keywords = _title_keywords(all_captions_text)

        # Caption keywords for naming
        cap_keywords = _title_keywords(
            " ".join(members["caption"].fillna("").head(20).tolist())
        )

        # Preferred name: what the images actually show, scored zero-shot
        # against the concept vocabulary. Caption keywords are the fallback,
        # because a caption is written by whoever posted and frequently has
        # nothing to do with the subject of the frame.
        vis_name, _concept_sims = _visual_concept_match(member_emb)
        if vis_name:
            name = _concept_display_name(vis_name)
            if name in seen_names:
                n_variant = 2
                while f"{name} {n_variant}" in seen_names:
                    n_variant += 1
                name = f"{name} {n_variant}"
        else:
            name = " ".join(cap_keywords[:3]) if cap_keywords else f"Visual theme {cid}"
        seen_names.add(name)

        # Photography execution profile (how it is shot, not what is shot)
        if style_scores is not None:
            from src.style_tags import aggregate_styles
            style_tags = aggregate_styles(style_scores, indices=indices)
        else:
            style_tags = []

        summaries.append({
            "cluster_id": int(cid),
            "name": name,
            "keywords": keywords,
            "blip_caption": combined_caption,
            "blip_confidence": round(1.0 if captions else 0.0, 4),
            "style_tags": style_tags,
            "n_posts": len(indices),
            "representative_post_id": best["post_id"],
            "representative_author": best.get("author", ""),
            "example_captions": [
                str(c) for c in members["caption"].fillna("").head(5).tolist() if c
            ],
        })

    print(f"[collector] summarized {len(summaries)} clusters (top-{top_k_caption} centroid captioning)")
    return summaries


# ──────────────────────────────────────────────────────────────────────────
# Stage 6: Temporal trend analysis
# ──────────────────────────────────────────────────────────────────────────
def compute_temporal_trends(
    df: pd.DataFrame,
    labels: np.ndarray,
    clusters: dict[int, list[int]],
    definition: Optional["TrendDefinition"] = None,
) -> dict[int, dict[str, Any]]:
    """
    Temporal analysis per cluster under the formal trend definition.

    Replaces the previous "emerging_score", which was
    ``(... + 0.5 * n_recent + ...) / log1p(total)``. That formula was
    unbounded and not comparable across clusters: a 50-post cluster
    structurally outranked a 10-post one regardless of growth, and on the real
    corpus it produced values from 0.0 to 13.5 for the same underlying
    behaviour. A number that varies by 100x on identical evidence cannot be
    called a score.

    The verdict now comes from ``src.trend_definition``, which applies named,
    individually auditable criteria (support, persistence, author breadth,
    growth margin, significance, engagement floor, coverage) and is willing to
    return ``InsufficientData``. ``emerging_score`` is retained as a strictly
    bounded, documented tie-break for ranking *confirmed* trends — see
    ``_bounded_priority``.
    """
    from src.data_quality import clean_engagement, engagement_summary
    from src.trend_definition import (
        DEFAULT_DEFINITION,
        classify_cluster_trend,
    )

    definition = definition or DEFAULT_DEFINITION
    frame = clean_engagement(df.copy())
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")

    if "timestamp" in frame.columns and frame["timestamp"].notna().any():
        now = frame["timestamp"].max()
    else:
        now = pd.Timestamp.now(tz=timezone.utc)

    corpus_median_likes = (
        float(frame["likes"].median()) if frame["likes"].notna().any() else None
    )

    trends: dict[int, dict[str, Any]] = {}

    for cid, indices in clusters.items():
        members = frame.iloc[indices].copy()
        ts = members["timestamp"]

        # Daily counts (kept verbatim for the history store)
        daily: dict[str, int] = {}
        for t in ts:
            if pd.notna(t):
                day = t.strftime("%Y-%m-%d")
                daily[day] = daily.get(day, 0) + 1

        verdict = classify_cluster_trend(
            members,
            now=now,
            definition=definition,
            corpus_median_likes=corpus_median_likes,
        )
        eng = engagement_summary(members)

        type_counts: dict[str, int] = {}
        if "content_type" in members.columns:
            for ct in members["content_type"].fillna("Image"):
                type_counts[ct] = type_counts.get(ct, 0) + 1

        hashtag_freq: dict[str, int] = {}
        if "hashtags" in members.columns:
            for h in members["hashtags"]:
                if isinstance(h, list):
                    for tag in h:
                        hashtag_freq[tag] = hashtag_freq.get(tag, 0) + 1
        top_hashtags = sorted(hashtag_freq.items(), key=lambda x: -x[1])[:8]

        trends[cid] = {
            "cluster_id": cid,
            "daily_counts": dict(sorted(daily.items())),
            # ---- definition-driven verdict ----
            "classification": verdict["classification"],
            "classification_reason": verdict["reason"],
            "failed_criteria": verdict["failed_criteria"],
            "criteria": verdict["criteria"],
            "n_recent": verdict["n_recent"],
            "n_prior": verdict["n_prior"],
            "recent_active_days": verdict["recent_active_days"],
            "recent_authors": verdict["recent_authors"],
            "recent_rate_per_day": verdict["recent_rate_per_day"],
            "prior_rate_per_day": verdict["prior_rate_per_day"],
            "rate_ratio": verdict["rate_ratio"],
            "relative_growth": verdict["relative_growth"],
            "p_value": verdict["p_value"],
            "significant": verdict["significant"],
            # ---- bounded, comparable priority (0-1) ----
            "emerging_score": round(_bounded_priority(verdict), 4),
            # ---- engagement (median-first, coverage-aware) ----
            "total_posts": verdict["n_total"],
            "recent_posts": verdict["n_recent"],
            "prior_posts": verdict["n_prior"],
            "median_likes": eng["median_likes"],
            "median_comments": eng["median_comments"],
            "avg_likes": eng["mean_likes"],
            "avg_comments": eng["mean_comments"],
            "likes_coverage": eng["likes_coverage"],
            "median_likes_reliable": eng["median_likes_reliable"],
            "content_types": type_counts,
            "top_hashtags": [h for h, _ in top_hashtags],
            "first_seen": str(ts.min()) if len(ts) else "",
            "latest_post": str(ts.max()) if len(ts) else "",
        }

    counts = pd.Series([t["classification"] for t in trends.values()]).value_counts()
    print(
        f"[collector] temporal verdicts for {len(trends)} clusters: "
        + ", ".join(f"{k}={v}" for k, v in counts.items())
    )
    return trends


def _bounded_priority(verdict: dict[str, Any]) -> float:
    """
    Strictly bounded 0-1 ranking value, replacing the old unbounded emerging score.

    This is a *tie-break among clusters the definition already confirmed*; it
    is not the definition and on its own it does not license a trend claim.
    Any non-Rising verdict scores 0, so a cluster cannot rank highly on
    engagement alone while failing the growth criteria.

    Components, each squashed to 0-1 so no single term can dominate:
      strength  how far the rate ratio exceeds the significance floor
      breadth   distinct active days, saturating at 7
      adoption  distinct accounts, saturating at 10
    """
    if verdict.get("classification") != "Rising":
        return 0.0
    p = float(verdict.get("p_value") or 1.0)
    # p in (0, 0.05] -> 1.0 ; p = 0.2 -> ~0.5
    strength = 1.0 - min(1.0, (p / 0.05) ** 0.5) if p > 0 else 1.0
    breadth = min(1.0, verdict.get("recent_active_days", 0) / 7.0)
    adoption = min(1.0, verdict.get("recent_authors", 0) / 10.0)
    return float(0.5 * strength + 0.25 * breadth + 0.25 * adoption)


# ──────────────────────────────────────────────────────────────────────────
# Stage 6b: Hashtag trend analysis
# ──────────────────────────────────────────────────────────────────────────
def compute_hashtag_trends(
    df: pd.DataFrame,
    top_k: int = 20,
) -> list[dict[str, Any]]:
    """Compute hashtag frequency trends across the dataset.

    Returns the top_k most common hashtags with their post counts and
    which clusters they appear in.
    """
    hashtag_freq: dict[str, int] = {}
    hashtag_clusters: dict[str, set[int]] = {}
    hashtag_posts: dict[str, list[str]] = {}

    for _, row in df.iterrows():
        tags = row.get("hashtags", [])
        # parquet round-trips store these as numpy arrays, not python lists
        if not hasattr(tags, "__iter__") or isinstance(tags, (str, bytes)):
            continue
        post_id = row.get("post_id", "")
        cluster = row.get("cluster_id", -1)
        for h in tags:
            h = str(h).lower().strip("#")
            if not h:
                continue
            hashtag_freq[h] = hashtag_freq.get(h, 0) + 1
            if h not in hashtag_clusters:
                hashtag_clusters[h] = set()
                hashtag_posts[h] = []
            if cluster >= 0:
                hashtag_clusters[h].add(cluster)
            if len(hashtag_posts[h]) < 3:
                hashtag_posts[h].append(post_id)

    ranked = sorted(hashtag_freq.items(), key=lambda x: -x[1])[:top_k]
    trends = []
    for h, count in ranked:
        trends.append({
            "hashtag": h,
            "count": count,
            "clusters": sorted(hashtag_clusters[h]),
            "sample_posts": hashtag_posts[h],
        })

    print(f"[collector] hashtag trends: {len(trends)} hashtags tracked")
    return trends


# ──────────────────────────────────────────────────────────────────────────
# Stage 7: Build RAG index
# ──────────────────────────────────────────────────────────────────────────
def build_rag_index(
    cluster_summaries: list[dict[str, Any]],
    temporal_trends: dict[int, dict[str, Any]],
    df: pd.DataFrame,
) -> None:
    """Build FAISS RAG index from cluster summaries + temporal data."""
    import faiss
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(config.RAG_EMBED_MODEL)

    chunks: list[dict[str, Any]] = []
    for s in cluster_summaries:
        cid = s["cluster_id"]
        t = temporal_trends.get(cid, {})
        daily = t.get("daily_counts", {})
        daily_str = ", ".join(f"{d}:{c}" for d, c in list(daily.items())[-10:])

        # Trend wording is taken verbatim from the definition verdict, so the
        # RAG text can never assert a percentage the definition rejected.
        growth = t.get("relative_growth")
        classification = t.get("classification", "InsufficientData")
        reason = t.get("classification_reason", "")
        if classification == "Rising":
            growth_s = (
                f"confirmed rising trend: {t.get('n_recent', 0)} posts in the last "
                f"{t.get('recent_active_days', 0)} active day(s) across "
                f"{t.get('recent_authors', 0)} account(s), "
                f"{(growth * 100):.0f}% above the prior period "
                f"(p={t.get('p_value', 1.0):.3f})"
            )
        else:
            growth_s = f"not a confirmed trend ({reason})"

        examples = " | ".join(s.get("example_captions", [])[:3])

        # Photography execution profile (how it is shot)
        style_tags = s.get("style_tags", [])
        style_str = ""
        if style_tags:
            from src.style_tags import format_style_tags
            style_str = f"Shot like: {format_style_tags(style_tags)}. "

        # Engagement: median first. Means are reported but not indexed as the
        # headline number, because a single viral post moves the mean by orders
        # of magnitude (observed max was 15.5M likes against a 7k median).
        med_l = t.get("median_likes")
        med_c = t.get("median_comments")
        coverage = t.get("likes_coverage")
        eng_parts = []
        if med_l is not None:
            reliability = (
                "" if t.get("median_likes_reliable", True)
                else " (small sample, treat as indicative only)"
            )
            eng_parts.append(f"median {med_l:,.0f} likes{reliability}")
        if med_c is not None:
            eng_parts.append(f"median {med_c:,.0f} comments")
        if coverage is not None and coverage < 1.0:
            eng_parts.append(f"like count known for {coverage:.0%} of posts")
        eng_str = ", ".join(eng_parts) if eng_parts else "engagement unknown"

        # Content types
        ct = t.get("content_types", {})
        ct_str = ", ".join(f"{k}:{v}" for k, v in ct.items()) if ct else "images"

        # Top hashtags
        top_kw = t.get("top_hashtags", [])
        ht_str = ", ".join(f"#{h}" for h in top_kw[:5]) if top_kw else ""

        text = (
            f"Instagram visual trend: \"{s['name']}\". "
            f"Keywords: {', '.join(s['keywords'][:6])}. "
            f"BLIP caption: \"{s['blip_caption']}\". "
            f"{style_str}"
            f"Growth: {growth_s}. "
            f"Posts: {s['n_posts']} total, {t.get('recent_posts', 0)} recent. "
            f"Engagement: {eng_str}. "
            f"Content types: {ct_str}. "
            f"Daily: {daily_str}. "
        )
        if ht_str:
            text += f"Hashtags: {ht_str}. "
        if examples:
            text += f"Examples: {examples}."
        chunks.append({
            "cluster_id": cid,
            "text": text,
            "name": s["name"],
            "keywords": s["keywords"],
            "style_tags": style_tags,
            "classification": classification,
            "classification_reason": reason,
            "relative_growth": growth,
            "rate_ratio": t.get("rate_ratio"),
            "p_value": t.get("p_value"),
            "emerging_score": t.get("emerging_score", 0),
            "median_likes": med_l,
            "median_comments": med_c,
            "likes_coverage": coverage,
            "n_recent": t.get("n_recent"),
            "n_prior": t.get("n_prior"),
            "recent_active_days": t.get("recent_active_days"),
            "recent_authors": t.get("recent_authors"),
            "content_types": t.get("content_types", {}),
            "top_hashtags": t.get("top_hashtags", []),
        })

    if not chunks:
        print("[collector] no chunks to index")
        return

    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype("float32"))

    config.INSTAGRAM_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(config.INSTAGRAM_RAG_INDEX_PATH))
    config.INSTAGRAM_RAG_CHUNKS_PATH.write_text(
        json.dumps(chunks, indent=1, default=_json_default)
    )
    print(f"[collector] RAG index built: {len(chunks)} chunks, dim={dim}")


# ──────────────────────────────────────────────────────────────────────────
# Stage 8: Save final artifacts
# ──────────────────────────────────────────────────────────────────────────
def save_trends_json(
    cluster_summaries: list[dict[str, Any]],
    temporal_trends: dict[int, dict[str, Any]],
    df: pd.DataFrame,
    labels: np.ndarray,
    hashtag_trends: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Save combined trends JSON for the RAG query layer."""
    now = datetime.now(timezone.utc)
    themes = []
    for s in cluster_summaries:
        cid = s["cluster_id"]
        t = temporal_trends.get(cid, {})
        themes.append({
            "name": s["name"],
            "keywords": s["keywords"],
            "blip_caption": s["blip_caption"],
            "blip_confidence": s["blip_confidence"],
            "style_tags": s.get("style_tags", []),
            "n_posts": s["n_posts"],
            # ---- formal trend definition verdict ----
            "classification": t.get("classification", "InsufficientData"),
            "classification_reason": t.get("classification_reason", ""),
            "failed_criteria": t.get("failed_criteria", []),
            "recent_posts": t.get("recent_posts", 0),
            "prior_posts": t.get("prior_posts", 0),
            "recent_active_days": t.get("recent_active_days", 0),
            "recent_authors": t.get("recent_authors", 0),
            "recent_rate_per_day": t.get("recent_rate_per_day", 0.0),
            "prior_rate_per_day": t.get("prior_rate_per_day", 0.0),
            "rate_ratio": t.get("rate_ratio"),
            "relative_growth": t.get("relative_growth"),
            "p_value": t.get("p_value"),
            "emerging_score": t.get("emerging_score", 0),
            "median_likes": t.get("median_likes"),
            "median_comments": t.get("median_comments"),
            "likes_coverage": t.get("likes_coverage"),
            "median_likes_reliable": t.get("median_likes_reliable", False),
            "avg_likes": t.get("avg_likes"),
            "avg_comments": t.get("avg_comments"),
            "content_types": t.get("content_types", {}),
            "top_hashtags": t.get("top_hashtags", []),
            "daily_counts": t.get("daily_counts", {}),
            "example_captions": s.get("example_captions", []),
            "representative_author": s.get("representative_author", ""),
            "representative_post_id": s.get("representative_post_id", ""),
            "first_seen": t.get("first_seen", ""),
            "latest_post": t.get("latest_post", ""),
        })

    # Confirmed trends first; among equals, higher bounded priority; then size.
    themes.sort(
        key=lambda x: (
            x["classification"] == "Rising",
            x["emerging_score"],
            x["n_posts"],
        ),
        reverse=True,
    )

    n_rising = sum(1 for t in themes if t["classification"] == "Rising")
    payload = {
        "disclaimer": (
            "REAL INSTAGRAM DATA: posts, timestamps, likes, comments, views, "
            "hashtags, content types, and images come from public Instagram "
            "accounts via Apify. Not synthetic."
        ),
        "trend_definition": {
            "statement": (
                "A cluster is a Rising trend only if it clears every criterion: "
                "support (>=8 posts in the recent 4d window), persistence "
                "(present on >=3 distinct days), author breadth (>=3 distinct "
                "accounts), growth margin (>=+50% per-day rate vs the prior 4d), "
                "significance (binomial p<=0.05 against a no-growth null), "
                "engagement not depressed (<75% of corpus median likes), and "
                "engagement coverage (>=60% of recent posts have a known like "
                "count). Clusters failing any criterion are reported as "
                "InsufficientData rather than ranked."
            ),
            "parameters": DEFAULT_DEFINITION.as_dict(),
            "note": (
                "This corpus is a single scrape: 87% of posts fall in one 11-day "
                "window, so cross-window growth is weakly identified. Run "
                "src.data_collector daily to build the longitudinal history in "
                "data/trend_history.db."
            ),
        },
        "n_rising": n_rising,
        "n_insufficient": len(themes) - n_rising,
        "generated_at": now.isoformat(),
        "source": "instagram",
        "scan_days": config.INSTAGRAM_SCAN_DAYS,
        "n_posts": len(df),
        "n_themes": len(themes),
        "themes": themes,
        "hashtag_trends": hashtag_trends or [],
    }

    config.INSTAGRAM_TRENDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.INSTAGRAM_TRENDS_PATH.write_text(
        json.dumps(payload, indent=1, default=_json_default)
    )
    print(f"[collector] trends JSON saved: {len(themes)} themes")
    return payload


# ──────────────────────────────────────────────────────────────────────────
# Full pipeline
# ──────────────────────────────────────────────────────────────────────────
def _run_baseline_pipeline(days: int, registry) -> dict[str, Any]:
    """Full HDBSCAN baseline run — creates the initial cluster registry."""
    print(f"\n{'='*60}")
    print(f"  TrendLens Instagram Baseline — {days}-day window")
    print(f"{'='*60}\n")

    # Stage 0: Clean existing data for a fresh start
    print("--- Stage 0: Cleaning existing data ---")
    clean_instagram_data()

    # Stage 1: Fetch
    print("\n--- Stage 1: Fetching Instagram posts via Apify ---")
    df = fetch_and_save(days=days)
    if df.empty:
        return {"n_posts": 0, "error": "No posts fetched"}

    # Stage 2: Download images
    print("\n--- Stage 2: Downloading images ---")
    saved = download_images(df)
    print(f"[collector] downloaded {len(saved)}/{len(df)} new images")

    # Stage 3: Embed
    print("\n--- Stage 3: CLIP image embeddings ---")
    emb, aligned = embed_images(df)
    if emb is None:
        return {"n_posts": len(df), "error": "No images could be embedded"}
    df = aligned

    # Stage 3b: Photography style tagging (CLIP zero-shot, no extra passes)
    print("\n--- Stage 3b: Photography style tagging ---")
    style_scores = _style_scores_for(emb)

    # Stage 4: HDBSCAN clustering
    print("\n--- Stage 4: HDBSCAN clustering ---")
    labels, clusters = cluster_posts(emb, df)

    # Stage 4b: Initialize cluster registry with locked centroids
    print("\n--- Stage 4b: Initializing cluster registry ---")
    # Build per-cluster metadata for the registry
    cluster_meta = {}
    for cid, indices in clusters.items():
        members = df.iloc[indices]
        all_captions = " ".join(members["caption"].fillna("").tolist())
        cap_kw = _title_keywords(all_captions)
        cluster_meta[int(cid)] = {
            "name": " ".join(cap_kw[:3]) if cap_kw else f"Theme {cid}",
            "keywords": cap_kw[:10],
            "blip_caption": "",
        }
    registry.init_from_hdbscan(labels, emb, cluster_meta)
    registry.build_centroid_index()
    registry.save()

    # Stage 5: Summarize
    print("\n--- Stage 5: BLIP captioning + summarization ---")
    summaries = summarize_clusters(df, labels, clusters, emb, style_scores=style_scores)

    # Stage 6: Temporal trends
    print("\n--- Stage 6: Temporal trend analysis ---")
    temporal = compute_temporal_trends(df, labels, clusters)

    # Stage 6b: Hashtag trend analysis
    print("\n--- Stage 6b: Hashtag trend analysis ---")
    hashtag_trends = compute_hashtag_trends(df)

    # Stage 7: RAG index
    print("\n--- Stage 7: Building RAG index ---")
    build_rag_index(summaries, temporal, df)

    # Stage 8: Save
    print("\n--- Stage 8: Saving artifacts ---")
    trends = save_trends_json(summaries, temporal, df, labels, hashtag_trends)

    print(f"\n{'='*60}")
    print("  Baseline complete!")
    print(f"  Posts: {len(df)}")
    print(f"  Clusters: {len(summaries)} (locked in registry)")
    print(f"  Top emerging: {trends['themes'][0]['name'] if trends['themes'] else 'none'}")
    print(f"{'='*60}\n")

    return {
        "n_posts": len(df),
        "n_clusters": len(summaries),
        "trends": trends,
        "baseline": True,
    }


def _run_incremental_pipeline(days: int, registry) -> dict[str, Any]:
    """Incremental run — assign new images to existing clusters via KNN."""
    print(f"\n{'='*60}")
    print(f"  TrendLens Instagram Incremental — {days}-day window")
    print(f"  Existing clusters: {len(registry.clusters)}")
    print(f"{'='*60}\n")

    # Stage 1: Fetch posts (including existing ones — dedup happens inside)
    print("\n--- Stage 1: Fetching Instagram posts via Apify ---")
    new_df = fetch_and_save(days=days)
    if new_df.empty:
        print("[collector] no posts fetched")
        return {"n_posts": 0, "error": "No posts fetched"}

    # Load previously saved posts to find genuinely new ones
    existing_posts_path = config.INSTAGRAM_DIR / "all_posts.parquet"
    if existing_posts_path.exists():
        old_df = pd.read_parquet(existing_posts_path)
        old_ids = set(old_df["post_id"].tolist())
        new_mask = ~new_df["post_id"].isin(old_ids)
        genuinely_new = new_df[new_mask].reset_index(drop=True)
        # Merge for saving
        combined = pd.concat([old_df, genuinely_new], ignore_index=True)
        combined = combined.drop_duplicates(subset=["post_id"], keep="last")
        combined = combined.sort_values("timestamp").reset_index(drop=True)
    else:
        genuinely_new = new_df
        combined = new_df

    n_new = len(genuinely_new)
    print(f"[collector] {n_new} genuinely new posts out of {len(new_df)} fetched")

    # Save combined posts for next incremental run
    combined.to_parquet(existing_posts_path, index=False)

    if n_new == 0:
        print("[collector] no new posts — updating trends from existing data")
        return _rebuild_trends_from_registry(registry, combined)

    # Stage 2: Download only new images
    print("\n--- Stage 2: Downloading new images ---")
    saved = download_images(genuinely_new)
    print(f"[collector] downloaded {len(saved)}/{n_new} new images")

    # Stage 3: Embed new images
    # Save old embeddings + metadata BEFORE embed_images overwrites them
    old_emb_path = config.INSTAGRAM_EMBEDDINGS_PATH
    old_emb = np.load(old_emb_path) if old_emb_path.exists() else None
    old_embed_meta_path = config.INSTAGRAM_DIR / "embed_meta.parquet"
    old_embed_meta = pd.read_parquet(old_embed_meta_path) if old_embed_meta_path.exists() else pd.DataFrame()

    print("\n--- Stage 3: CLIP image embeddings ---")
    new_emb, aligned_new = embed_images(genuinely_new)
    if new_emb is None:
        print("[collector] no new images could be embedded")
        return _rebuild_trends_from_registry(registry, combined)

    # embed_images overwrote the embeddings file with only the new batch —
    # restore the full history immediately so a later-stage crash cannot
    # lose the old per-image embeddings.
    if old_emb is not None:
        np.save(old_emb_path, np.vstack([old_emb, new_emb]))
        pd.concat([old_embed_meta, aligned_new], ignore_index=True).to_parquet(
            old_embed_meta_path, index=False
        )

    # Stage 4: KNN assignment to existing clusters
    print("\n--- Stage 4: KNN assignment to existing clusters ---")
    assignments = registry.assign_new_images(new_emb)

    # Build metadata for unassigned images
    unassigned_meta = []
    for a in assignments:
        if not a["assigned"]:
            idx = a["post_idx"]
            row = aligned_new.iloc[idx] if idx < len(aligned_new) else {}
            unassigned_meta.append({
                "post_id": row.get("post_id", ""),
                "keywords": _title_keywords(str(row.get("caption", ""))),
                "timestamp": str(row.get("timestamp", "")),
            })

    # Add unassigned to pending candidates
    unassigned_embs = new_emb[
        [not a["assigned"] for a in assignments]
    ] if any(not a["assigned"] for a in assignments) else np.array([], dtype="float32").reshape(0, new_emb.shape[1] if len(new_emb) > 0 else 512)

    if len(unassigned_embs) > 0:
        registry.add_pending(unassigned_embs, unassigned_meta)

        # Stage 4b: Detect emerging micro-clusters
        print("\n--- Stage 4b: Detecting emerging micro-clusters ---")
        new_clusters = registry.detect_emerging(new_emb, unassigned_meta, assignments)
        if new_clusters:
            registry.build_centroid_index()

    registry.save()

    # Stage 5: Rebuild summaries from all assigned posts
    print("\n--- Stage 5: Rebuilding cluster summaries ---")
    # Merge new and old embeddings for full trend computation
    if old_emb is not None:
        all_emb = np.vstack([old_emb, new_emb])
    else:
        all_emb = new_emb

    # Build aligned metadata matching all_emb (old embed_meta + new aligned)
    if not old_embed_meta.empty:
        all_meta = pd.concat([old_embed_meta, aligned_new], ignore_index=True)
    else:
        all_meta = aligned_new.copy()

    # Sanity check: embeddings and metadata must be aligned
    assert len(all_emb) == len(all_meta), (
        f"Alignment mismatch: all_emb={len(all_emb)} but all_meta={len(all_meta)}"
    )

    # Assign cluster labels to ALL posts using the registry
    full_labels = _assign_all_to_registry(registry, all_emb)

    # Build clusters dict for downstream functions
    clusters_dict: dict[int, list[int]] = {}
    for i, label in enumerate(full_labels):
        if label >= 0:
            clusters_dict.setdefault(label, []).append(i)

    # Photography style profiles over the full embedding matrix
    print("\n--- Stage 5b: Photography style tagging ---")
    style_profiles = {}
    style_scores = _style_scores_for(all_emb)
    if style_scores is not None:
        from src.style_tags import aggregate_styles
        for cid, indices in clusters_dict.items():
            style_profiles[cid] = aggregate_styles(style_scores, indices=indices)

    # Stage 6: Temporal trends — use all_meta (aligned with all_emb)
    print("\n--- Stage 6: Temporal trend analysis ---")
    temporal = compute_temporal_trends(all_meta, full_labels, clusters_dict)

    # Stage 6b: Hashtag trends
    print("\n--- Stage 6b: Hashtag trend analysis ---")
    hashtag_trends = compute_hashtag_trends(all_meta)

    # Stage 7: RAG index
    print("\n--- Stage 7: Building RAG index ---")
    summaries = _build_summaries_from_registry(
        registry, all_meta, full_labels, clusters_dict,
        style_profiles=style_profiles, emb=all_emb,
    )
    build_rag_index(summaries, temporal, all_meta)

    # Stage 8: Save
    print("\n--- Stage 8: Saving artifacts ---")
    trends = save_trends_json(summaries, temporal, all_meta, full_labels, hashtag_trends)

    print(f"\n{'='*60}")
    print("  Incremental run complete!")
    print(f"  New posts: {n_new}")
    print(f"  Total clusters: {len(registry.clusters)}")
    assigned_count = sum(1 for a in assignments if a["assigned"])
    print(f"  Assigned to existing: {assigned_count}")
    print(f"  New micro-clusters: {len([c for c in registry.clusters.values() if c.get('lifecycle') == 'Emerging'])}")
    print(f"  Top emerging: {trends['themes'][0]['name'] if trends['themes'] else 'none'}")
    print(f"{'='*60}\n")

    return {
        "n_posts": len(combined),
        "n_new_posts": n_new,
        "n_clusters": len(registry.clusters),
        "assigned_to_existing": assigned_count,
        "new_micro_clusters": len([c for c in registry.clusters.values() if c.get('lifecycle') == 'Emerging']),
        "trends": trends,
        "baseline": False,
    }


def _assign_all_to_registry(registry, all_emb: np.ndarray) -> np.ndarray:
    """Assign ALL embeddings (old + new) to nearest cluster via FAISS KNN.

    Returns an array of integer labels (-1 = noise, stable_id mapping).
    """
    index, stable_ids = registry.load_centroid_index()
    if index is None or not stable_ids:
        return np.full(len(all_emb), -1, dtype=int)

    k = min(1, len(stable_ids))
    scores, indices = index.search(
        np.ascontiguousarray(all_emb.astype("float32")), k
    )

    # Map stable_id to a contiguous integer label for downstream functions
    sid_to_int = {sid: i for i, sid in enumerate(stable_ids)}

    labels = np.full(len(all_emb), -1, dtype=int)
    for i in range(len(all_emb)):
        sim = float(scores[i][0])
        idx = int(indices[i][0])
        if idx >= 0 and sim >= registry.assignment_threshold:
            sid = stable_ids[idx]
            labels[i] = sid_to_int[sid]

    return labels


# Curated concept vocabulary for zero-shot visual theme naming. Cluster names
# derived from caption frequency are often caption noise ("very difficult
# focusing"); scoring member IMAGES against these concepts yields a name that
# describes what the photos actually show.
#
# This vocabulary is the naming bottleneck, and it used to be 100% food: with
# only food concepts available, a cluster of street-style photos was still
# named after the nearest food dish, because argmax over 24 food prompts has
# no correct answer to give. The list now spans every subject the tracked
# accounts actually post, so a beauty cluster can be named for beauty.
#
# Two properties matter when editing this list:
#   * phrases stay concrete and visually literal — CLIP matches on what is in
#     the frame, so "a flat lay of makeup products" beats "beauty content";
#   * subjects are balanced across domains, so no one domain wins by count.
_VISUAL_CONCEPTS = [
    # ── beauty / makeup ──
    "a close-up of a made-up face with bold lipstick",
    "a close-up of a made-up face with full eyeshadow",
    "a flat lay of makeup products and brushes",
    "a hand holding a lipstick tube",
    "a person applying makeup with a brush",
    "a glassy highlighted cheek in close-up",
    "a skincare product bottle on a clean surface",
    "a manicure and painted nails in close-up",
    "a perfume bottle with soft studio light",
    # ── food / savoury ──
    "fried chicken", "pizza", "burger", "sushi", "ramen noodle soup",
    "fresh salad bowl", "tacos mexican street food", "ice cream",
    "grilled steak barbecue", "seafood platter", "rice and curry dish",
    "sandwich brunch plate", "cozy home cooked meal",
    "hands preparing food in a kitchen",
    "a plated restaurant dish photographed from above",
    "a bowl of soup or stew on a wooden table",
    # ── baking / sweets ──
    "layered cake dessert", "bakery pastries and bread", "pancakes with syrup",
    "a chocolate cake slice", "cookies and brownies",
    "a bakery counter full of pastries",
    # ── coffee / drinks ──
    "latte art in a coffee cup", "barista pouring espresso coffee",
    "a pour over coffee setup with a kettle",
    "cocktails and drinks", "smoothie bowl with fruit",
    "a matcha or tea drink on a table",
    # ── fashion / beauty editorial ──
    "a street style outfit photographed full length",
    "a fashion model posing against a plain wall",
    "a runwayshow photograph",
    "a clothing rack or flat lay of clothes",
    "a close-up of fabric texture and stitching",
    "a person showing a handbag and shoes",
    # ── travel / outdoors ──
    "a mountain landscape at sunrise",
    "a tropical beach with palm trees",
    "a city street scene with buildings",
    "a forest trail with trees",
    "a lake or river in a wide landscape",
    "a travel landmark or monument",
    "a wildlife animal in its natural habitat",
    # ── fitness / wellness ──
    "a person running outdoors",
    "a gym workout with weights",
    "a yoga pose on a mat",
    # ── home / interior ──
    "a styled living room interior",
    "a bedroom or home decor detail",
    "a workspace desk setup",
    # ── people / events / culture ──
    "a portrait of a person against a blurred background",
    "people gathering at an event",
    "a musician performing on stage",
    "a birthday celebration with candles",
    "a product packaging shot on a plain background",
    "a book or magazine cover photographed flat",
    "text and typography on a sign or title card",
]


def _build_summaries_from_registry(
    registry, df: pd.DataFrame, labels: np.ndarray,
    clusters: dict[int, list[int]],
    style_profiles: Optional[dict[int, list[dict[str, Any]]]] = None,
    emb: Optional[np.ndarray] = None,
) -> list[dict[str, Any]]:
    """Build cluster summaries using registry metadata.

    ``style_profiles`` maps cluster int-label → ranked style tags (from
    src.style_tags) computed over the current embedding matrix.
    ``emb`` (aligned with ``df``/``labels``) enables centroid-based
    representative selection — the member that LOOKS most like the theme,
    instead of the highest-liked outlier.
    """
    index, stable_ids = registry.load_centroid_index()
    int_to_sid = {i: sid for i, sid in enumerate(stable_ids)}

    seen_names: set[str] = set()
    summaries = []
    for cid_int, indices in sorted(clusters.items()):
        sid = int_to_sid.get(cid_int, f"cls_{cid_int}")
        rec = registry.get_cluster(sid) or {}

        members = df.iloc[indices] if indices else pd.DataFrame()
        n_posts = len(indices)

        # Keywords from captions
        if n_posts > 0 and "caption" in members.columns:
            all_captions = " ".join(members["caption"].fillna("").tolist())
            keywords = _title_keywords(all_captions)
        else:
            keywords = rec.get("keywords", [])

        # Representative + display name from what the images actually show:
        # zero-shot CLIP classification of member images against
        # _VISUAL_CONCEPTS. The winning concept becomes the theme name and
        # the text anchor for representative selection, so names, answers,
        # and displayed pictures all agree.
        vis_name: Optional[str] = None
        combined: Optional[np.ndarray] = None
        if emb is not None and n_posts > 0:
            member_mat = emb[np.asarray(indices)].astype("float32")
            vis_name, concept_sims = _visual_concept_match(member_mat)

            if index is not None and stable_ids:
                try:
                    centroid = index.reconstruct(int(cid_int)).reshape(-1).astype("float32")
                    centroid_sims = member_mat @ centroid
                    combined = (
                        0.35 * centroid_sims + 0.65 * concept_sims
                        if vis_name is not None else centroid_sims
                    )
                except Exception:  # noqa: BLE001 — concept sims alone suffice
                    combined = concept_sims if vis_name is not None else None

        rep_id, rep_author = "", ""
        if n_posts > 0:
            best = None
            if combined is not None:
                order = list(np.argsort(-combined)[: min(5, len(indices))])
                likes_vals = (
                    [float(x) for x in members["likes"].tolist()]
                    if "likes" in members.columns else [0.0] * len(members)
                )
                if sum(likes_vals) > 0:
                    best_pos = max(order, key=lambda p: likes_vals[p])
                else:
                    best_pos = order[0]
                best = members.iloc[best_pos]
            elif "likes" in members.columns and members["likes"].sum() > 0:
                best = members.loc[members["likes"].idxmax()]
            else:
                best = members.loc[members["timestamp"].idxmax()]
            rep_id = str(best.get("post_id", ""))
            rep_author = str(best.get("author", ""))
            examples = [
                str(c) for c in members["caption"].fillna("").head(5).tolist() if c
            ]

        # BLIP caption from registry or skip
        caption = rec.get("blip_caption", "")
        conf = 0.0

        display_name = (
            vis_name.capitalize() if vis_name else rec.get("name", f"Theme {sid}")
        )
        base_name, n_variant = display_name, 2
        while display_name in seen_names:
            display_name = f"{base_name} {n_variant}"
            n_variant += 1
        seen_names.add(display_name)
        summaries.append({
            "cluster_id": cid_int,
            "name": display_name,
            "keywords": keywords or rec.get("keywords", []),
            "blip_caption": caption,
            "blip_confidence": conf,
            "style_tags": (style_profiles or {}).get(cid_int, []),
            "n_posts": n_posts,
            "representative_post_id": rep_id,
            "representative_author": rep_author,
            "example_captions": examples,
        })

    return summaries


def _rebuild_trends_from_registry(registry, df: pd.DataFrame) -> dict[str, Any]:
    """Rebuild trends JSON from existing data when no new posts arrive."""
    print("[collector] rebuilding trends from existing data")
    # Load existing embeddings if available
    emb_path = config.INSTAGRAM_EMBEDDINGS_PATH
    if not emb_path.exists():
        return {"n_posts": len(df), "error": "No embeddings available"}

    emb = np.load(emb_path)
    full_labels = _assign_all_to_registry(registry, emb)

    clusters_dict: dict[int, list[int]] = {}
    for i, label in enumerate(full_labels):
        if label >= 0:
            clusters_dict.setdefault(label, []).append(i)

    temporal = compute_temporal_trends(df, full_labels, clusters_dict)
    style_scores = _style_scores_for(emb)
    style_profiles = {}
    if style_scores is not None:
        from src.style_tags import aggregate_styles
        for cid, indices in clusters_dict.items():
            style_profiles[cid] = aggregate_styles(style_scores, indices=indices)
    summaries = _build_summaries_from_registry(
        registry, df, full_labels, clusters_dict,
        style_profiles=style_profiles, emb=emb,
    )
    build_rag_index(summaries, temporal, df)
    trends = save_trends_json(summaries, temporal, df, full_labels)

    return {
        "n_posts": len(df),
        "n_clusters": len(registry.clusters),
        "trends": trends,
        "baseline": False,
    }


def run_pipeline(days: Optional[int] = None, incremental: bool = True) -> dict[str, Any]:
    """Run the Instagram data collection + analysis pipeline.

    Parameters
    ----------
    days : int, optional
        Days to scan (default from config).
    incremental : bool
        If True (default), checks for an existing cluster registry and
        runs incrementally — only new images are embedded, assigned to
        existing clusters via FAISS KNN, and emerging micro-clusters are
        detected.  If False or no baseline exists, runs the full
        HDBSCAN baseline.
    """
    from src.cluster_tracker import ClusterRegistry

    days = days or config.INSTAGRAM_SCAN_DAYS
    registry = ClusterRegistry.load()

    if incremental and registry.clusters:
        return _run_incremental_pipeline(days, registry)

    return _run_baseline_pipeline(days, registry)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TrendLens Instagram data collector")
    parser.add_argument("--days", type=int, default=None, help="Days to scan (default: from config)")
    parser.add_argument("--baseline", action="store_true",
                        help="Force a full baseline run (re-cluster from scratch)")
    args = parser.parse_args()
    run_pipeline(days=args.days, incremental=not args.baseline)
