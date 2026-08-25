"""
live.py
-------
Real-time trend ingestion + emerging-trend detection.

Sources (config.LIVE_SOURCE):
  * "reddit"    — official Reddit public feed / OAuth. REAL timestamps + REAL
                  engagement (score = upvotes, num_comments). Often 403-blocked
                  from datacenter IPs; use OAuth (REDDIT_CLIENT_ID / _SECRET)
                  or a residential network.
  * "wikimedia" — Wikimedia Commons API, key-free. REAL upload timestamps +
                  real image files; has NO upvote/comment signal (reported
                  honestly as absent).
  * "auto"      — try Reddit, fall back to Wikimedia Commons automatically.

Pipeline (``python -m src.live``):

  1. fetch_posts()            -> data/live/live_posts.parquet   (deduped, real fields)
  2. download_images()        -> data/live/images/<post_id>.<ext>
  3. embed_live_images()      -> data/live/live_embeddings.npy (+ live_embed_meta.parquet)
  4. detect_trends()          -> artifacts/cluster_metadata/live_trends.json
                                (themes = HDBSCAN groups; growth over recent vs prior window)

A "trend before it has a name" is a theme that emerges from fresh images alone
— it is named later by captioning its representative (BLIP), not by a hashtag.

INTEGRITY
---------
* timestamps are REAL (Reddit created_utc / Commons upload timestamp)
* engagement is REAL for Reddit; honestly ABSENT for Commons (never invented)
* the representative image shown is the highest-engagement real post
  (Reddit) or the newest post (Commons)
* a theme's look is described only from its own images' BLIP caption/title
  keywords — never invented
"""

from __future__ import annotations

import json
import re
import string
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

import config
from src.style_tags import format_style_tags

WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"
WIKIMEDIA_USER_AGENT = "TrendLens/0.1 (visual-trend demo; no engagement signal)"

STOPWORDS = set(
    "the a an and or but of in on for with at by to from this that these those "
    "it is are was were be been being i we you they he she has have had my our "
    "your their its my your your just like get got one two new day night time "
    "post pics pic photo im dont dont. made make use using used first last "
    "best top every more most over under again about into".split()
)


def _local_ext(image_url: str) -> str:
    ext = Path(image_url.split("?")[0]).suffix.lower()
    if ext not in (".jpg", ".jpeg", ".png", ".webp"):
        return ".jpg"
    return ext


def _local_path(post_id: str, image_url: str) -> Path:
    return config.LIVE_IMAGES_DIR / f"{post_id}{_local_ext(image_url)}"


def _resolve_image_url(post: dict[str, Any]) -> str:
    url = (post.get("url") or "").strip()
    low = url.lower()
    if any(url.lower().endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp")):
        return url
    if "i.redd.it" in low or "preview.redd.it" in low:
        return url
    preview = (
        post.get("preview", {})
        .get("images", [{}])[0]
        .get("source", {})
        .get("url", "")
    )
    return preview.replace("&amp;", "&").replace("\\u0026", "&")


# ──────────────────────────────────────────────────────────────────────────
# Reddit fetch
# ──────────────────────────────────────────────────────────────────────────
def _reddit_auth_token(client_id: str, client_secret: str, user_agent: str) -> str:
    import requests

    resp = requests.post(
        "https://www.reddit.com/api/v1/access_token",
        auth=(client_id, client_secret),
        data={"grant_type": "client_credentials"},
        headers={"User-Agent": user_agent},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def fetch_reddit_posts(
    subreddits: Optional[list[str]] = None,
    limit: Optional[int] = None,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    Fetch recent posts from the official Reddit feed.

    Uses OAuth when credentials are given, else the public JSON endpoint with
    a polite user agent. A failed subreddit is skipped, never fatal.
    Returns a list of post dicts with REAL timestamps + engagement.
    """
    import requests

    subreddits = subreddits or config.LIVE_SUBREDDITS
    limit = limit or config.LIVE_PER_SUBREDDIT_LIMIT
    user_agent = user_agent or "TrendLens/0.1 (visual-trend demo)"

    token = None
    if client_id and client_secret:
        token = _reddit_auth_token(client_id, client_secret, user_agent)

    posts: list[dict[str, Any]] = []
    for sub in subreddits:
        headers = {"User-Agent": user_agent}
        if token:
            headers["Authorization"] = f"bearer {token}"
            url = f"https://oauth.reddit.com/r/{sub}/new"
        else:
            url = f"https://www.reddit.com/r/{sub}/new.json"
        try:
            resp = requests.get(url, headers=headers, params={"limit": limit}, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception:  # noqa: BLE001 — one failing sub must not kill the run
            continue

        now = datetime.now(timezone.utc)
        for child in data.get("data", {}).get("children", []):
            d = child.get("data", {})
            img = _resolve_image_url(d)
            if not img:
                continue
            created = d.get("created_utc", 0)
            posts.append(
                {
                    "post_id": d.get("id"),
                    "subreddit": sub,
                    "title": d.get("title", ""),
                    "created_utc": (
                        datetime.fromtimestamp(created, tz=timezone.utc).isoformat()
                        if created else ""
                    ),
                    "score": int(d.get("score") or 0),
                    "num_comments": int(d.get("num_comments") or 0),
                    "permalink": "https://www.reddit.com" + (d.get("permalink") or ""),
                    "image_url": img,
                    "source": "reddit",
                    "fetched_at": now.isoformat(),
                }
            )
    return posts


def fetch_wikimedia_posts(
    query_terms: Optional[list[str]] = None,
    per_term_limit: Optional[int] = None,
    days: Optional[int] = None,
) -> list[dict[str, Any]]:
    """
    Fetch recent images from Wikimedia Commons (key-free, real upload dates).

    For each topic term it prefers the matching Commons *category* sorted by
    upload timestamp (newest first) — genuinely recent, topic-relevant files.
    If the category does not exist, it falls back to a relevance search
    (only results inside the recency window are kept). Commons has no
    upvote/comment signal, so ``score``/``num_comments`` are honestly 0 and
    ``source`` is ``wikimedia-commons``.
    """
    import time

    import requests

    query_terms = query_terms or config.LIVE_WIKIMEDIA_QUERIES
    per_term_limit = per_term_limit or config.LIVE_WIKIMEDIA_LIMIT
    days = days or config.LIVE_WIKIMEDIA_SCAN_DAYS
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    headers = {"User-Agent": WIKIMEDIA_USER_AGENT}

    posts: list[dict[str, Any]] = []
    for i, term in enumerate(query_terms):
        if i > 0:
            time.sleep(1.0)  # be polite to the anonymous rate limit
        try:
            pages = _wikimedia_recent_pages(term, per_term_limit, headers, requests)
            if pages is None:
                pages = _wikimedia_search_pages(term, per_term_limit, headers, requests)
        except Exception:  # noqa: BLE001 — one failing term must not kill the run
            continue
        for p in pages:
            ii = (p.get("imageinfo") or [{}])[0]
            ts = ii.get("timestamp")
            if not ts:
                continue
            try:
                created = datetime.fromisoformat(ts.replace("Z", "+00:00")).isoformat()
            except ValueError:
                continue
            if created < cutoff:
                continue
            url = (ii.get("thumburl") or ii.get("url") or "").replace("&amp;", "&")
            if not url:
                continue
            posts.append(
                {
                    "post_id": f"wm-{p.get('pageid')}",
                    "subreddit": term,  # reuse the channel column with the search term
                    "title": p.get("title", "").replace("File:", "", 1),
                    "created_utc": created,
                    "score": 0,  # Commons has no upvotes — engagement absent, honestly 0
                    "num_comments": 0,
                    "permalink": p.get("descriptionurl", ""),
                    "image_url": url,
                    "source": "wikimedia-commons",
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                }
            )
    return posts


def _api_get(
    requests_mod: Any, params: dict[str, Any], headers: dict[str, str], tries: int = 3
) -> Optional[dict[str, Any]]:
    """Wikimedia API GET with backoff on 429 (anonymous rate limits)."""
    import time

    for attempt in range(tries):
        try:
            resp = requests_mod.get(
                WIKIMEDIA_API, params=params, headers=headers, timeout=30
            )
        except Exception:  # noqa: BLE001
            return None
        if resp.status_code == 429:
            retry_after = int((getattr(resp, "headers", None) or {}).get("Retry-After", "2") or 2)
            time.sleep(min(max(retry_after, 1), 5) * (attempt + 1))
            continue
        resp.raise_for_status()
        return resp.json()
    return None


def _wikimedia_recent_pages(
    term: str,
    limit: int,
    headers: dict[str, str],
    requests_mod: Any,
) -> Optional[list[dict[str, Any]]]:
    """Most recent files in ``Category:<term>`` (newest first), or None if the
    category does not exist."""
    for cat in (term, term[0].upper() + term[1:]):
        check = _api_get(
            requests_mod,
            {"action": "query", "titles": f"Category:{cat}", "format": "json"},
            headers,
        )
        if check is None:
            return None
        page = next(iter(check.get("query", {}).get("pages", {}).values()), None)
        if not page or page.get("missing"):
            continue
        data = _api_get(
            requests_mod,
            {
                "action": "query",
                "generator": "categorymembers",
                "gcmtitle": f"Category:{cat}",
                "gcmtype": "file",
                "gcmsort": "timestamp",
                "gcmdir": "descending",
                "gcmlimit": limit,
                "prop": "imageinfo",
                "iiprop": "url|timestamp|size",
                "iiurlwidth": "320",
                "format": "json",
            },
            headers,
        )
        if data is None:
            return None
        return list(data.get("query", {}).get("pages", {}).values())
    return None


def _wikimedia_search_pages(
    term: str,
    limit: int,
    headers: dict[str, str],
    requests_mod: Any,
) -> list[dict[str, Any]]:
    """Relevance-search fallback (recency filtering still happens upstream)."""
    data = _api_get(
        requests_mod,
        {
            "action": "query",
            "generator": "search",
            "gsrsearch": term,
            "gsrnamespace": "6",
            "gsrlimit": limit,
            "prop": "imageinfo",
            "iiprop": "url|timestamp|size",
            "iiurlwidth": "320",
            "format": "json",
        },
        headers,
    )
    return list((data or {}).get("query", {}).get("pages", {}).values())


def fetch_posts() -> list[dict[str, Any]]:
    """Dispatch on config.LIVE_SOURCE: "auto" → Reddit, then Commons fallback."""
    import os

    mode = config.LIVE_SOURCE
    creds = {
        "client_id": os.environ.get("REDDIT_CLIENT_ID"),
        "client_secret": os.environ.get("REDDIT_CLIENT_SECRET"),
    }
    if mode in ("reddit", "auto"):
        posts = fetch_reddit_posts(**creds)
        if posts or mode == "reddit":
            return posts
    return fetch_wikimedia_posts()


# ──────────────────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────────────────
def save_posts(posts: list[dict[str, Any]], path: Optional[Path] = None) -> pd.DataFrame:
    path = Path(path) if path else config.LIVE_POSTS_PATH
    df = pd.DataFrame(posts)
    if df.empty:
        return df
    old = load_posts(path)
    if old is not None and not old.empty:
        df = pd.concat([old, df], ignore_index=True)
    df = df.drop_duplicates(subset=["post_id"], keep="last")
    df = df.sort_values("created_utc").reset_index(drop=True)
    df.to_parquet(path)
    return df


def load_posts(path: Optional[Path] = None) -> Optional[pd.DataFrame]:
    path = Path(path) if path else config.LIVE_POSTS_PATH
    if not path.exists():
        return None
    return pd.read_parquet(path)


# ──────────────────────────────────────────────────────────────────────────
# Image download + CLIP embedding
# ──────────────────────────────────────────────────────────────────────────
def download_images(
    df: Optional[pd.DataFrame] = None, out_dir: Optional[Path] = None
) -> list[str]:
    """Download any live post images that are not already local. Returns saved names.

    Wikimedia throttles anonymous downloads (429) — retries with backoff and
    spaces requests politely.
    """
    import time

    import requests

    out_dir = Path(out_dir) if out_dir else config.LIVE_IMAGES_DIR
    df = df if df is not None else (load_posts() or pd.DataFrame())
    saved: list[str] = []
    if df.empty:
        return saved
    headers = {"User-Agent": WIKIMEDIA_USER_AGENT}
    for i, (_, p) in enumerate(df.iterrows()):
        if i > 0:
            time.sleep(1.0)
        dest = _local_path(p["post_id"], p["image_url"])
        if dest.exists():
            continue
        for attempt in range(4):
            try:
                r = requests.get(p["image_url"], headers=headers, timeout=30)
            except Exception:  # noqa: BLE001
                break
            if r.status_code == 429:
                time.sleep(3 * (attempt + 1))
                continue
            if r.status_code == 200 and len(r.content) > 500:
                dest.write_bytes(r.content)
                saved.append(dest.name)
            break
    return saved


def embed_live_images(
    df: Optional[pd.DataFrame] = None,
) -> tuple[Optional[np.ndarray], pd.DataFrame]:
    """
    CLIP-embed the live images (CPU-friendly, small sets). Saves the
    embeddings + an aligned metadata parquet. Returns (embeddings, aligned_df).
    """
    import torch
    from PIL import Image

    from src.embeddings import l2_normalize, load_clip

    df = df if df is not None else (load_posts() or pd.DataFrame())
    if df.empty:
        return None, df

    model, processor, device = load_clip()
    embs: list[np.ndarray] = []
    keep: list[str] = []
    with torch.no_grad():
        for _, p in df.iterrows():
            path = _local_path(p["post_id"], p["image_url"])
            try:
                img = Image.open(path).convert("RGB")
            except Exception:  # noqa: BLE001 — skip unreadable image
                continue
            inputs = processor(images=img, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out = model.get_image_features(**inputs)
            if hasattr(out, "pooler_output"):  # transformers >= 4.x full output
                out = out.pooler_output
            feats = out.detach().cpu().numpy()
            embs.append(feats[0])
            keep.append(p["post_id"])

    if not embs:
        return None, df.iloc[0:0]
    emb = l2_normalize(np.vstack(embs).astype("float32"))
    aligned = df[df["post_id"].isin(keep)].reset_index(drop=True)
    np.save(config.LIVE_EMBEDDINGS_PATH, emb)
    aligned.to_parquet(config.LIVE_DIR / "live_embed_meta.parquet")
    return emb, aligned


def _title_keywords(title: str) -> list[str]:
    tokens = re.sub(f"[{re.escape(string.punctuation)}]", " ", title.lower()).split()
    seen: list[str] = []
    for t in tokens:
        if t not in STOPWORDS and len(t) > 2 and t not in seen:
            seen.append(t)
    return seen[:8]


def _blip_caption(path: Path) -> tuple[str, float]:
    """Caption a single image with BLIP (optional — falls back gracefully)."""
    try:
        from src.interpretation import caption_image, load_blip
    except Exception:  # noqa: BLE001
        return "", 0.0
    try:
        model = load_blip()
        return caption_image(model, str(path))
    except Exception:  # noqa: BLE001
        return "", 0.0


# ──────────────────────────────────────────────────────────────────────────
# Emerging-trend detection
# ──────────────────────────────────────────────────────────────────────────
def detect_trends(
    emb: Optional[np.ndarray] = None,
    df: Optional[pd.DataFrame] = None,
    recent_days: Optional[int] = None,
    prior_days: Optional[int] = None,
) -> dict[str, Any]:
    """
    Group live posts into themes (HDBSCAN on CLIP embeddings), compute real
    growth over the recent vs prior window, and write ``live_trends.json``.
    """
    import hdbscan

    recent_days = recent_days or config.LIVE_RECENT_WINDOW_DAYS
    prior_days = prior_days or recent_days

    if emb is None and config.LIVE_EMBEDDINGS_PATH.exists():
        emb = np.load(config.LIVE_EMBEDDINGS_PATH, mmap_mode="r")
    if df is None:
        meta = config.LIVE_DIR / "live_embed_meta.parquet"
        df = pd.read_parquet(meta) if meta.exists() else None
    if emb is None or df is None or df.empty or len(emb) != len(df):
        return {
            "disclaimer": config.LIVE_DATA_WARNING,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": config.LIVE_SOURCE,
            "subreddits": config.LIVE_SUBREDDITS,
            "recent_window_days": recent_days,
            "n_posts": 0,
            "n_themes": 0,
            "themes": [],
            "note": "No live posts embedded yet — run `python -m src.live` first.",
        }

    n = len(df)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=max(3, min(8, n // 5)),
        min_samples=2,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(np.asarray(emb, dtype="float32"))

    now = datetime.now(timezone.utc)
    recent_cutoff = now - timedelta(days=recent_days)
    prior_cutoff = now - timedelta(days=recent_days + prior_days)

    # Zero-shot photography-style scores for every embedded image (how it
    # is shot — framing, lighting, grading, process — not what is shot).
    style_scores = None
    try:
        from src.style_tags import aggregate_styles, compute_style_scores
        style_scores = compute_style_scores(np.asarray(emb, dtype="float32"))
    except Exception as e:  # noqa: BLE001 — styling must never break detection
        print(f"style tagging skipped ({e})")

    themes: list[dict[str, Any]] = []
    for label in sorted(set(labels.tolist())):
        if label < 0:
            continue  # HDBSCAN noise — not a theme
        mask = labels == label
        members = df[mask].reset_index(drop=True)
        recent = members[members["created_utc"] >= recent_cutoff.isoformat()]
        prior = members[
            (members["created_utc"] >= prior_cutoff.isoformat())
            & (members["created_utc"] < recent_cutoff.isoformat())
        ]
        n_recent, n_prior = len(recent), len(prior)
        if n_recent == 0:
            continue
        growth = (
            (n_recent / n_prior - 1.0) if n_prior > 0
            else None  # brand-new theme this window
        )

        # Source-aware channel label: Reddit subs are channels; Commons search
        # terms are just the query that found the images.
        theme_source = (
            members["source"].value_counts().idxmax()
            if "source" in members and len(members["source"].value_counts())
            else "reddit"
        )
        terms = sorted(members["subreddit"].unique().tolist())
        if theme_source == "reddit":
            channel_label = "r/" + ", r/".join(terms)
        else:
            channel_label = "wikimedia search: " + ", ".join(terms)

        # Representative: highest-engagement real post (Reddit) or the newest
        # real post (sources without engagement).
        if theme_source == "reddit" and int(members["score"].max()) > 0:
            best_idx = members["score"].idxmax()
        else:
            best_idx = members["created_utc"].idxmax()
        best = members.loc[best_idx]
        local = _local_path(best["post_id"], best["image_url"])
        caption, conf = ("", 0.0)
        if local.exists():
            caption, conf = _blip_caption(local)

        keywords = _title_keywords(best["title"])
        kw_joined = ", ".join(keywords[:4]).lower() or "the theme's look"

        theme_style: list[dict[str, Any]] = []
        if style_scores is not None:
            from src.style_tags import aggregate_styles
            theme_style = aggregate_styles(
                style_scores, indices=[int(i) for i in np.flatnonzero(mask)]
            )

        themes.append(
            {
                "name": (keywords[0] if keywords else "unnamed theme")
                + (" visual theme" if keywords else ""),
                "keywords": keywords,
                "keywords_emoji": "🔥",
                "blip_caption": caption,
                "blip_confidence": round(float(conf), 4),
                "style_tags": theme_style,
                "subreddits": sorted(members["subreddit"].unique().tolist()),
                "channel_label": channel_label,
                "source": theme_source,
                "has_engagement": bool(
                    int(members["score"].sum()) > 0
                    or int(members["num_comments"].sum()) > 0
                ),
                "n_posts": int(len(members)),
                "recent_posts": int(n_recent),
                "prior_posts": int(n_prior),
                "growth_rate": growth,
                "avg_engagement": float(members["score"].mean()),
                "total_engagement": int(members["score"].sum()),
                "total_comments": int(members["num_comments"].sum()),
                "representative_image_url": (
                    f"/api/live-images?name={local.name}" if local.exists()
                    else best["image_url"]
                ),
                "representative_post": best["permalink"],
                "first_seen": str(members["created_utc"].min()),
                "latest_post": str(members["created_utc"].max()),
                "replicate": (
                    f"Shoot images that feature {kw_joined}"
                    + (f"; the representative post shows “{caption}”" if caption else "")
                    + (
                        f", typically shot as {format_style_tags(theme_style)}"
                        if theme_style
                        else ""
                    )
                    + "."
                ),
            }
        )

    themes.sort(
        key=lambda t: (
            t["growth_rate"] is not None,
            t["growth_rate"] if t["growth_rate"] is not None else 0.0,
            t["has_engagement"],
            t["avg_engagement"],
        ),
        reverse=True,
    )

    source_overall = (
        df["source"].value_counts().idxmax()
        if "source" in df and len(df["source"].value_counts())
        else config.LIVE_SOURCE
    )
    payload = {
        "disclaimer": config.LIVE_DATA_WARNING,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": source_overall,
        "subreddits": sorted(df["subreddit"].unique().tolist()),
        "recent_window_days": recent_days,
        "n_posts": int(n),
        "n_themes": len(themes),
        "themes": themes,
    }
    config.LIVE_TRENDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.LIVE_TRENDS_PATH.write_text(json.dumps(payload, indent=1))
    return payload


def load_trends(path: Optional[Path] = None) -> Optional[dict[str, Any]]:
    path = Path(path) if path else config.LIVE_TRENDS_PATH
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


# ──────────────────────────────────────────────────────────────────────────
# CLI driver
# ──────────────────────────────────────────────────────────────────────────
def run_pipeline() -> dict[str, Any]:
    posts = fetch_posts()
    if not posts:
        print("no posts fetched from any source.")
        print("  - Reddit's public feed is frequently 403-blocked from datacenter/cloud")
        print("    IPs. To use Reddit: set REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET")
        print("    (create an app at reddit.com/prefs/apps) or run from a residential")
        print("    network.")
        print("  - Wikimedia Commons (key-free) is used automatically as a fallback;")
        print("    force it with TRENDLENS_LIVE_SOURCE=wikimedia.")
        return {"n_posts": 0}
    df = save_posts(posts)
    print(f"live posts on disk: {len(df)} (source: {df['source'].iloc[0] if len(df) else 'none'})")
    df = df.sort_values("created_utc").tail(config.LIVE_MAX_EMBED_POSTS).reset_index(drop=True)
    print(f"using most recent {len(df)} posts for this run")
    saved = download_images(df)
    print(f"downloaded images: {len(saved)}/{len(df)} (throttled hosts may need re-runs; downloads persist)")
    emb, aligned = embed_live_images(df)
    if emb is None:
        print("no images could be embedded")
        return {"n_posts": int(len(df)), "embedded": 0}
    print(f"embedded: {emb.shape[0]}")
    src = df["source"].iloc[0] if len(df) else ""
    if src == "wikimedia-commons":
        recent_days = config.LIVE_WIKIMEDIA_RECENT_DAYS
    else:
        recent_days = config.LIVE_RECENT_WINDOW_DAYS
    trends = detect_trends(emb, aligned, recent_days=recent_days)
    print(f"themes detected: {trends['n_themes']} from {trends['n_posts']} posts "
          f"(source: {trends.get('source')})")
    for t in trends["themes"][:5]:
        g = t["growth_rate"]
        eng = f"avg={round(t['avg_engagement'], 1)}" if t["has_engagement"] else "no engagement signal"
        print(f"  {t['name']}: recent={t['recent_posts']} prior={t['prior_posts']} "
              f"growth={g if g is None else round(g, 2)} {eng}")
    return trends


if __name__ == "__main__":
    run_pipeline()
