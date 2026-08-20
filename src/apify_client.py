"""
apify_client.py
---------------
Instagram data fetching via the Apify REST API.

Uses the ``apify/instagram-post-scraper`` actor to fetch posts from public
Instagram accounts. No SDK dependency — raw ``requests`` calls only.

INTEGRITY
---------
* timestamps, likes, comments, views come from Instagram via Apify
* images are Instagram CDN-hosted (downloaded locally for CLIP embedding)
* no data is fabricated — missing fields are honestly 0 or empty
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any, Optional

import requests

APIFY_BASE = "https://api.apify.com/v2"
ACTOR_ID = "apify~instagram-post-scraper"
POLL_INTERVAL = 5  # seconds between status checks
MAX_WAIT = 600  # 10-minute timeout for the actor run


def _apify_token() -> str:
    token = os.environ.get("APIFY_API_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "APIFY_API_TOKEN not set. Add it to your .env file."
        )
    return token


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {_apify_token()}"}


def _extract_username(url_or_name: str) -> str:
    """Extract a clean username from an Instagram URL or bare username."""
    s = url_or_name.strip().rstrip("/")
    if "instagram.com/" in s:
        s = s.split("instagram.com/")[-1]
    s = s.lstrip("@").split("?")[0].split("/")[0]
    return s


def _normalize_post(item: dict[str, Any]) -> dict[str, Any]:
    """Map Apify's raw output to our canonical post format.

    Extracts images, engagement (likes, comments, views), metadata
    (hashtags, mentions, tagged users, content type), and media info
    (dimensions, alt text, video duration, music). Missing fields are
    honestly 0 / empty — never fabricated.
    """
    ts = item.get("timestamp") or item.get("takenAt") or ""
    if ts and isinstance(ts, str):
        ts = ts.replace("Z", "+00:00")

    image_url = (
        item.get("imageUrl")
        or item.get("displayUrl")
        or item.get("thumbnailUrl")
        or ""
    )

    caption = item.get("caption") or item.get("text") or ""
    hashtags = item.get("hashtags") or []
    if not hashtags and caption:
        hashtags = [w.lstrip("#") for w in caption.split() if w.startswith("#")]

    likes = item.get("likesCount") or item.get("likes") or 0
    comments = item.get("commentsCount") or item.get("comments") or 0
    views = item.get("videoViewCount") or item.get("views") or 0
    plays = item.get("videoPlayCount") or 0

    post_id = item.get("id") or item.get("shortCode") or ""

    # Mentions (extracted as username strings)
    raw_mentions = item.get("mentions") or []
    mentions = [
        m if isinstance(m, str) else m.get("username", "")
        for m in raw_mentions
    ]

    # Tagged users (extracted as username strings)
    raw_tagged = item.get("taggedUsers") or []
    tagged_users = [
        t if isinstance(t, str) else t.get("username", "")
        for t in raw_tagged
    ]

    # Music / audio info
    music = item.get("musicInfo") or {}
    audio_artist = music.get("artist_name") or ""
    audio_song = music.get("song_name") or ""
    uses_original_audio = bool(music.get("uses_original_audio", False))

    # Dimensions
    width = int(item.get("dimensionsWidth") or 0)
    height = int(item.get("dimensionsHeight") or 0)

    return {
        "post_id": str(post_id),
        "author": item.get("ownerUsername") or item.get("username") or "",
        "full_name": item.get("ownerFullName") or "",
        "owner_id": item.get("ownerId") or "",
        "image_url": image_url,
        "caption": caption,
        "alt_text": item.get("alt") or "",
        "timestamp": ts,
        # Engagement
        "likes": int(likes),
        "comments": int(comments),
        "views": int(views),
        "plays": int(plays),
        # Metadata
        "hashtags": hashtags,
        "mentions": mentions,
        "tagged_users": tagged_users,
        "post_url": item.get("url") or "",
        # Content type
        "content_type": item.get("type") or "",
        "product_type": item.get("productType") or "",
        # Media info
        "width": width,
        "height": height,
        "video_duration": float(item.get("videoDuration") or 0),
        "audio_artist": audio_artist,
        "audio_song": audio_song,
        "uses_original_audio": uses_original_audio,
        # Flags
        "is_comments_disabled": bool(item.get("isCommentsDisabled", False)),
        "source": "instagram",
    }


def fetch_instagram_posts(
    accounts: Optional[list[str]] = None,
    days: int = 10,
    posts_per_account: int = 50,
    timeout: int = MAX_WAIT,
) -> list[dict[str, Any]]:
    """
    Fetch recent Instagram posts from the given accounts via Apify.

    Parameters
    ----------
    accounts : list[str]
        Instagram URLs or usernames. If None, reads from ``account.txt``.
    days : int
        How many days back to fetch.
    posts_per_account : int
        Max posts to fetch per account.
    timeout : int
        Max seconds to wait for the Apify actor run.

    Returns
    -------
    list[dict]
        Normalized post dicts with image_url, caption, timestamp, likes,
        comments, views, hashtags, source.
    """
    import config

    if accounts is None:
        accounts_file = config.INSTAGRAM_ACCOUNTS_FILE
        if not accounts_file.exists():
            raise FileNotFoundError(
                f"Accounts file not found: {accounts_file}. "
                "Create it with one Instagram URL/username per line."
            )
        raw = accounts_file.read_text().strip().splitlines()
        accounts = [a.strip() for a in raw if a.strip()]

    usernames = [_extract_username(a) for a in accounts]

    actor_input = {
        "username": usernames,
        "resultsLimit": posts_per_account,
        "skipPinnedPosts": False,
        "onlyPostsNewerThan": f"{days} days",
    }

    print(f"[apify] starting instagram-post-scraper for {len(usernames)} accounts")

    resp = requests.post(
        f"{APIFY_BASE}/acts/{ACTOR_ID}/runs",
        json=actor_input,
        headers=_headers(),
        timeout=30,
    )
    resp.raise_for_status()
    run_data = resp.json().get("data", {})
    run_id = run_data.get("id")
    if not run_id:
        raise RuntimeError(f"No run ID returned: {resp.json()}")
    print(f"[apify] run started: {run_id}")

    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(POLL_INTERVAL)
        status_resp = requests.get(
            f"{APIFY_BASE}/actor-runs/{run_id}",
            headers=_headers(),
            timeout=30,
        )
        status_resp.raise_for_status()
        run_info = status_resp.json().get("data", {})
        status = run_info.get("status")
        if status == "SUCCEEDED":
            break
        if status in ("FAILED", "ABORTED", "TIMED-OUT"):
            raise RuntimeError(f"Apify run {status}: {run_info.get('statusMessage', '')}")
        print(f"[apify] status: {status}...")
    else:
        raise TimeoutError(f"Apify run did not complete within {timeout}s")

    dataset_id = run_info.get("defaultDatasetId")
    if not dataset_id:
        raise RuntimeError("No dataset ID in completed run")

    print(f"[apify] fetching results from dataset {dataset_id}")
    items_resp = requests.get(
        f"{APIFY_BASE}/datasets/{dataset_id}/items",
        headers=_headers(),
        params={"format": "json"},
        timeout=60,
    )
    items_resp.raise_for_status()
    raw_items = items_resp.json()

    posts = [_normalize_post(item) for item in raw_items]
    posts = [p for p in posts if p["image_url"] and p["post_id"]]

    seen = set()
    deduped = []
    for p in posts:
        if p["post_id"] not in seen:
            seen.add(p["post_id"])
            deduped.append(p)

    print(f"[apify] fetched {len(deduped)} unique posts from {len(usernames)} accounts")
    return deduped
