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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

import config

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
    """Load previously saved Instagram posts."""
    if not config.INSTAGRAM_POSTS_PATH.exists():
        return None
    df = pd.read_parquet(config.INSTAGRAM_POSTS_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
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
    print(f"[collector] embedded {emb.shape[0]} images → {emb.shape}")
    return emb, aligned


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


def _blip_caption(path: Path) -> tuple[str, float]:
    try:
        from PIL import Image
        from src.interpretation import caption_image, load_blip
        model, processor, device = load_blip()
        img = Image.open(path).convert("RGB")
        caption = caption_image(model, processor, img, device=device)
        return caption, 1.0
    except Exception:
        return "", 0.0


def summarize_clusters(
    df: pd.DataFrame,
    labels: np.ndarray,
    clusters: dict[int, list[int]],
    emb: np.ndarray,
    top_k_caption: int = 3,
) -> list[dict[str, Any]]:
    """Generate cluster summaries with BLIP captions and keywords.

    BLIP captioning strategy: find the mathematical centroid of the
    HDBSCAN group and caption only the top *top_k_caption* closest
    vectors. This keeps descriptions concentrated on the core visual
    style while captioning only 3 images instead of the full cluster.
    """
    summaries: list[dict[str, Any]] = []

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

        # Representative: highest engagement among top-k
        if "likes" in closest_members.columns and closest_members["likes"].sum() > 0:
            best_local_idx = closest_members["likes"].idxmax()
        else:
            best_local_idx = closest_members["timestamp"].idxmax()
        best = closest_members.loc[best_local_idx]

        # Keywords from captions of all members
        all_captions_text = " ".join(members["caption"].fillna("").tolist())
        keywords = _title_keywords(all_captions_text)

        # Caption keywords for naming
        cap_keywords = _title_keywords(
            " ".join(members["caption"].fillna("").head(20).tolist())
        )
        name = " ".join(cap_keywords[:3]) if cap_keywords else f"Visual theme {cid}"

        summaries.append({
            "cluster_id": int(cid),
            "name": name,
            "keywords": keywords,
            "blip_caption": combined_caption,
            "blip_confidence": round(1.0 if captions else 0.0, 4),
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
    recent_days: int = 3,
    prior_days: int = 3,
) -> dict[int, dict[str, Any]]:
    """Compute daily post counts, growth rate, and emerging score per cluster."""
    now = datetime.now(timezone.utc)
    recent_cutoff = now - timedelta(days=recent_days)
    prior_cutoff = now - timedelta(days=recent_days + prior_days)

    trends: dict[int, dict[str, Any]] = {}

    for cid, indices in clusters.items():
        members = df.iloc[indices].copy()
        ts = members["timestamp"]
        if ts.dt.tz is not None:
            ts = ts.dt.tz_convert(timezone.utc)

        # Daily counts
        daily: dict[str, int] = {}
        for t in ts:
            if pd.notna(t):
                day = t.strftime("%Y-%m-%d")
                daily[day] = daily.get(day, 0) + 1

        # Recent vs prior
        n_recent = sum(
            1 for t in ts
            if pd.notna(t) and t >= recent_cutoff
        )
        n_prior = sum(
            1 for t in ts
            if pd.notna(t) and prior_cutoff <= t < recent_cutoff
        )

        # Growth rate
        if n_prior > 0:
            growth_rate = (n_recent / n_prior) - 1.0
        elif n_recent > 0:
            growth_rate = None  # brand new
        else:
            growth_rate = 0.0

        # Emerging score: growth velocity + engagement signals + novelty penalty
        total = len(indices)
        if n_recent > 0 and total > 0:
            novelty_penalty = np.log1p(total)

            # Engagement velocity: is engagement per post rising?
            recent_mask = members["timestamp"] >= recent_cutoff
            prior_mask = (members["timestamp"] >= prior_cutoff) & (members["timestamp"] < recent_cutoff)
            eng_score = 0.0
            if "likes" in members and "comments" in members:
                recent_eng = float(members.loc[recent_mask, "likes"].mean() + members.loc[recent_mask, "comments"].mean()) if recent_mask.any() else 0
                prior_eng = float(members.loc[prior_mask, "likes"].mean() + members.loc[prior_mask, "comments"].mean()) if prior_mask.any() else 0
                if prior_eng > 0:
                    eng_score = max(0, (recent_eng - prior_eng) / prior_eng)
                elif recent_eng > 0:
                    eng_score = 0.5

            # Absolute engagement bonus: high engagement = signal worth watching
            abs_eng = 0.0
            if "likes" in members:
                avg_likes = float(members["likes"].mean())
                if avg_likes > 5000:
                    abs_eng = 0.3
                elif avg_likes > 2000:
                    abs_eng = 0.2
                elif avg_likes > 500:
                    abs_eng = 0.1

            # View velocity for video content
            view_score = 0.0
            if "views" in members:
                recent_views = float(members.loc[recent_mask, "views"].mean()) if recent_mask.any() else 0
                prior_views = float(members.loc[prior_mask, "views"].mean()) if prior_mask.any() else 0
                if prior_views > 0:
                    view_score = max(0, (recent_views - prior_views) / prior_views)
                elif recent_views > 0:
                    view_score = 0.3

            if growth_rate is not None and growth_rate > 0:
                score = (2.0 * np.log1p(growth_rate) + 0.5 * n_recent + 1.0 * eng_score + 0.5 * view_score + 1.0 * abs_eng) / novelty_penalty
            elif growth_rate is None:
                score = (0.5 * n_recent + 1.0 * eng_score + 0.5 * view_score + 1.0 * abs_eng) / novelty_penalty
            else:
                score = abs_eng * 0.5 / novelty_penalty  # high engagement even without growth
        else:
            score = 0.0

        # Average engagement
        avg_likes = float(members["likes"].mean()) if "likes" in members else 0.0
        avg_comments = float(members["comments"].mean()) if "comments" in members else 0.0
        avg_views = float(members["views"].mean()) if "views" in members else 0.0
        avg_plays = float(members["plays"].mean()) if "plays" in members else 0.0

        # Content type breakdown
        type_counts: dict[str, int] = {}
        if "content_type" in members:
            for ct in members["content_type"].fillna("Image"):
                type_counts[ct] = type_counts.get(ct, 0) + 1

        # Hashtag frequency across cluster
        all_hashtags: list[str] = []
        if "hashtags" in members:
            for h in members["hashtags"]:
                if isinstance(h, list):
                    all_hashtags.extend(h)
        hashtag_freq: dict[str, int] = {}
        for h in all_hashtags:
            hashtag_freq[h] = hashtag_freq.get(h, 0) + 1
        top_hashtags = sorted(hashtag_freq.items(), key=lambda x: -x[1])[:8]

        trends[cid] = {
            "cluster_id": cid,
            "daily_counts": dict(sorted(daily.items())),
            "recent_posts": n_recent,
            "prior_posts": n_prior,
            "growth_rate": growth_rate,
            "emerging_score": round(float(score), 4),
            "total_posts": total,
            "avg_likes": round(avg_likes, 1),
            "avg_comments": round(avg_comments, 1),
            "avg_views": round(avg_views, 1),
            "avg_plays": round(avg_plays, 1),
            "content_types": type_counts,
            "top_hashtags": [h for h, _ in top_hashtags],
            "first_seen": str(ts.min()) if len(ts) else "",
            "latest_post": str(ts.max()) if len(ts) else "",
        }

    print(f"[collector] temporal trends for {len(trends)} clusters")
    return trends


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
        if not isinstance(tags, list):
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

    model = SentenceTransformer("all-MiniLM-L6-v2")

    chunks: list[dict[str, Any]] = []
    for s in cluster_summaries:
        cid = s["cluster_id"]
        t = temporal_trends.get(cid, {})
        daily = t.get("daily_counts", {})
        daily_str = ", ".join(f"{d}:{c}" for d, c in list(daily.items())[-10:])

        growth = t.get("growth_rate")
        if growth is None:
            growth_s = "brand new this period"
        elif growth is not None:
            growth_s = f"{'+' if growth >= 0 else ''}{growth*100:.0f}% growth"
        else:
            growth_s = "stable"

        examples = " | ".join(s.get("example_captions", [])[:3])

        # Engagement summary
        avg_l = t.get("avg_likes", 0)
        avg_c = t.get("avg_comments", 0)
        avg_v = t.get("avg_views", 0)
        eng_parts = []
        if avg_l > 0:
            eng_parts.append(f"{avg_l:.0f} avg likes")
        if avg_c > 0:
            eng_parts.append(f"{avg_c:.0f} avg comments")
        if avg_v > 0:
            eng_parts.append(f"{avg_v:.0f} avg views")
        eng_str = ", ".join(eng_parts) if eng_parts else "low engagement"

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
            "growth_rate": growth,
            "emerging_score": t.get("emerging_score", 0),
            "avg_likes": t.get("avg_likes", 0),
            "avg_comments": t.get("avg_comments", 0),
            "avg_views": t.get("avg_views", 0),
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
            "n_posts": s["n_posts"],
            "recent_posts": t.get("recent_posts", 0),
            "prior_posts": t.get("prior_posts", 0),
            "growth_rate": t.get("growth_rate"),
            "emerging_score": t.get("emerging_score", 0),
            "avg_likes": t.get("avg_likes", 0),
            "avg_comments": t.get("avg_comments", 0),
            "avg_views": t.get("avg_views", 0),
            "avg_plays": t.get("avg_plays", 0),
            "content_types": t.get("content_types", {}),
            "top_hashtags": t.get("top_hashtags", []),
            "daily_counts": t.get("daily_counts", {}),
            "example_captions": s.get("example_captions", []),
            "representative_author": s.get("representative_author", ""),
            "representative_post_id": s.get("representative_post_id", ""),
            "first_seen": t.get("first_seen", ""),
            "latest_post": t.get("latest_post", ""),
        })

    themes.sort(key=lambda x: x["emerging_score"], reverse=True)

    payload = {
        "disclaimer": (
            "REAL INSTAGRAM DATA: posts, timestamps, likes, comments, views, "
            "hashtags, content types, and images come from public Instagram "
            "accounts via Apify. Not synthetic."
        ),
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
    summaries = summarize_clusters(df, labels, clusters, emb)

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
    print(f"  Baseline complete!")
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
    from src.cluster_tracker import ClusterRegistry

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
    print("\n--- Stage 3: CLIP image embeddings ---")
    new_emb, aligned_new = embed_images(genuinely_new)
    if new_emb is None:
        print("[collector] no new images could be embedded")
        return _rebuild_trends_from_registry(registry, combined)

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
    old_emb_path = config.INSTAGRAM_EMBEDDINGS_PATH
    if old_emb_path.exists():
        old_emb = np.load(old_emb_path)
        all_emb = np.vstack([old_emb, new_emb])
    else:
        all_emb = new_emb

    # Assign cluster labels to ALL posts using the registry
    full_labels = _assign_all_to_registry(registry, all_emb)

    # Build clusters dict for downstream functions
    clusters_dict: dict[int, list[int]] = {}
    for i, label in enumerate(full_labels):
        if label >= 0:
            clusters_dict.setdefault(label, []).append(i)

    # Stage 6: Temporal trends
    print("\n--- Stage 6: Temporal trend analysis ---")
    temporal = compute_temporal_trends(combined, full_labels, clusters_dict)

    # Stage 6b: Hashtag trends
    print("\n--- Stage 6b: Hashtag trend analysis ---")
    hashtag_trends = compute_hashtag_trends(combined)

    # Stage 7: RAG index
    print("\n--- Stage 7: Building RAG index ---")
    summaries = _build_summaries_from_registry(registry, combined, full_labels, clusters_dict)
    build_rag_index(summaries, temporal, combined)

    # Stage 8: Save
    print("\n--- Stage 8: Saving artifacts ---")
    trends = save_trends_json(summaries, temporal, combined, full_labels, hashtag_trends)

    print(f"\n{'='*60}")
    print(f"  Incremental run complete!")
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
    import faiss

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


def _build_summaries_from_registry(
    registry, df: pd.DataFrame, labels: np.ndarray,
    clusters: dict[int, list[int]],
) -> list[dict[str, Any]]:
    """Build cluster summaries using registry metadata."""
    import faiss

    _, stable_ids = registry.load_centroid_index()
    int_to_sid = {i: sid for i, sid in enumerate(stable_ids)}

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

        # Representative: most recent post
        if n_posts > 0:
            if "likes" in members.columns and members["likes"].sum() > 0:
                best_idx = members["likes"].idxmax()
            else:
                best_idx = members["timestamp"].idxmax()
            best = members.loc[best_idx]
            rep_id = best.get("post_id", "")
            rep_author = best.get("author", "")
            examples = [
                str(c) for c in members["caption"].fillna("").head(5).tolist() if c
            ]
        else:
            rep_id = ""
            rep_author = ""
            examples = []

        # BLIP caption from registry or skip
        caption = rec.get("blip_caption", "")
        conf = 0.0

        summaries.append({
            "cluster_id": cid_int,
            "name": rec.get("name", f"Theme {sid}"),
            "keywords": keywords or rec.get("keywords", []),
            "blip_caption": caption,
            "blip_confidence": conf,
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
    summaries = _build_summaries_from_registry(registry, df, full_labels, clusters_dict)
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
